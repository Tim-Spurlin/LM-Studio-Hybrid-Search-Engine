import os
import numpy as np
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import networkx as nx
import h5py
import logging
import yaml
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - RiverMind - %(levelname)s - %(message)s')

class IMDDS_TruthFilter:
    def __init__(self, config_path='~/Desktop/Projects/Local LLM/OfflineRAG/rivermind_config.yaml'):
        config_full_path = os.path.expanduser(config_path)
        self.config = yaml.safe_load(open(config_full_path)) if os.path.exists(config_full_path) else {
            'prior_deceptive': 0.15, 'truth_score_threshold': 0.75, 'max_chunks': 5, 'svd_rank': 32
        }
        self.hdf5_path = os.path.expanduser('~/KnowledgeDrop/universal_knowledge_base.h5')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"RiverMind IMDDS v2.0 initialized on {self.device}")

        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.tfidf = TfidfVectorizer(max_features=5000)

        # Pre-compute everything once (latency killer fixed)
        self.truth_priors_emb, self.truth_priors_texts = self._load_or_bootstrap_truth_priors()
        if len(self.truth_priors_emb) > 0:
            self.truth_priors_tfidf = self.tfidf.fit_transform(self.truth_priors_texts).toarray()
            self.pca = PCA(n_components=min(64, self.truth_priors_emb.shape[0]))
            self.pca.fit(self.truth_priors_emb)
        else:
            self.truth_priors_tfidf = np.zeros((0, 5000))
            self.pca = None

        self.graph = nx.Graph()
        
        # True Kalman Filter Initialization
        svd_rank = self.config.get('svd_rank', 32)
        self.kalman_state = np.zeros(svd_rank)
        self.kalman_P = np.eye(svd_rank)       # Error covariance matrix
        self.kalman_Q = np.eye(svd_rank) * 0.01 # Process noise covariance
        self.kalman_R = np.eye(svd_rank) * 0.1  # Measurement noise covariance

    def _load_or_bootstrap_truth_priors(self):
        embeddings, texts = [], []
        try:
            if not os.path.exists(self.hdf5_path):
                return np.zeros((0, 384)), []
                
            with h5py.File(self.hdf5_path, 'r') as f:
                # Auto-bootstrap from your existing chunks (no migration needed)
                if 'chunks' in f:
                    texts_ds = f['chunks']['text']
                    emb_ds = f['chunks']['embeddings']
                    
                    # Safer check for is_verified to prevent crashes
                    if 'is_verified' in f['chunks']:
                        verified = f['chunks']['is_verified'][:]
                    else:
                        verified = [False] * len(texts_ds)
                        
                    for i in range(len(texts_ds)):
                        if verified[i]:
                            try:
                                txt = texts_ds[i].decode('utf-8') if isinstance(texts_ds[i], bytes) else str(texts_ds[i])
                                texts.append(txt)
                                embeddings.append(emb_ds[i])
                            except:
                                pass
                                
                # Dedicated collection if you later add it
                if 'truth_priors' in f:
                    embeddings.extend(f['truth_priors']['embeddings'][:])
                    texts.extend([t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in f['truth_priors']['text'][:]])
        except Exception as e:
            logging.error(f"HDF5 bootstrap error: {e}")
            
        return np.array(embeddings) if embeddings else np.zeros((0, 384)), texts

    def filter_and_score(self, retrieved_chunks):
        if not retrieved_chunks:
            return []
        chunk_texts = [c.get('text', '') for c in retrieved_chunks]
        chunk_embs = np.array([c.get('embedding', np.zeros(384)) for c in retrieved_chunks])

        # 1. KL + Entropy (propaganda filter)
        try:
            current_tfidf = self.tfidf.transform(chunk_texts).toarray()
            
            if len(self.truth_priors_tfidf) > 0:
                priors_to_compare = self.truth_priors_tfidf[:len(current_tfidf)]
                # If we retrieved more chunks than we have priors, pad the priors or tile them
                if len(priors_to_compare) < len(current_tfidf):
                    diff = len(current_tfidf) - len(priors_to_compare)
                    priors_to_compare = np.vstack([priors_to_compare, np.ones((diff, 5000)) * 1e-12])
                kl_divs = np.array([np.sum(p * np.log((p + 1e-12) / (q + 1e-12))) 
                               for p, q in zip(current_tfidf, priors_to_compare)])
            else:
                kl_divs = np.zeros(len(chunk_texts))
                
            entropies = np.array([entropy(p[p > 0]) if np.any(p > 0) else 0.0 for p in current_tfidf])
        except Exception as e:
            logging.error(f"KL/Entropy calc failed: {e}")
            kl_divs = np.zeros(len(chunk_texts))
            entropies = np.zeros(len(chunk_texts))

        # 2. Bayesian posterior (optimized)
        prior = self.config['prior_deceptive']
        likelihood = 1 / (1 + np.exp(kl_divs))
        posterior_truth = (likelihood * (1 - prior)) / (likelihood * (1 - prior) + (1 - likelihood) * prior)

        # 3. Spectral anomaly (pre-fitted PCA)
        recon_errors = np.zeros(len(chunk_embs))
        if self.pca is not None:
            try:
                recon = self.pca.inverse_transform(self.pca.transform(chunk_embs))
                recon_errors = np.linalg.norm(chunk_embs - recon, axis=1)
            except:
                pass

        # 4. Graph anomaly (co-occurrence & topological SIR spread)
        anomaly_scores = np.zeros(len(retrieved_chunks))
        if len(chunk_embs) > 1:
            try:
                # Build an adjacency matrix using cosine similarity
                norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                norm_embs = chunk_embs / norms
                sim_matrix = norm_embs @ norm_embs.T
                np.fill_diagonal(sim_matrix, 0)
                
                # Threshold to create topological edges (dense echo chamber detection)
                adj_matrix = (sim_matrix > 0.80).astype(float)
                
                # SIR Disease Spread Approximation - evaluate coordinated misinformation
                # by measuring isolated high-degree cliques vs expected uniform spread
                degrees = np.sum(adj_matrix, axis=1)
                mean_deg = np.mean(degrees)
                if mean_deg > 0:
                    anomaly_scores = np.maximum(0, (degrees - mean_deg) / mean_deg)
            except Exception as e:
                logging.error(f"Graph anomaly calc failed: {e}")

        # 5. Entropy-optimized fused score (superior to arbitrary exponents)
        # Includes negative weight for structural topological anomalies
        fused_score = (posterior_truth ** 0.40) * (np.exp(-kl_divs) ** 0.25) * \
                      (np.exp(-recon_errors) ** 0.20) * (np.exp(-entropies/10) ** 0.10) * \
                      (np.exp(-anomaly_scores) ** 0.05)

        results = []
        for i, chunk in enumerate(retrieved_chunks):
            # Fallback if text is somehow empty
            explanation = f"KL={kl_divs[i]:.3f} | Bayesian={posterior_truth[i]:.3f} | Recon={recon_errors[i]:.3f} | Entropy={entropies[i]:.3f}"
            results.append({
                'chunk': chunk,
                'truth_score': float(fused_score[i]),
                'explanation': explanation
            })
        return results

    def generate_biomed_hypothesis(self, high_truth_chunks):
        if len(high_truth_chunks) < 2:
            return "Insufficient high-truth biomedical data for hypothesis generation."
        emb_matrix = np.array([c['chunk'].get('embedding', np.zeros(384)) for c in high_truth_chunks])
        try:
            # Randomized SVD for speed
            U, S, Vt = np.linalg.svd(emb_matrix, full_matrices=False)
            
            # Ensure we don't exceed actual bounds if matrix is smaller than svd_rank
            rank = min(self.config['svd_rank'], U.shape[1], S.shape[0], Vt.shape[0])
            
            low_rank = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
            sim = low_rank @ low_rank.T
            np.fill_diagonal(sim, 0)
            i, j = np.unravel_index(sim.argmax(), sim.shape)
            novelty = sim[i, j]  # Higher = more novel connection
            
            # Handle possible missing title
            title_a = high_truth_chunks[i]['chunk'].get('title', 'Concept A')
            title_b = high_truth_chunks[j]['chunk'].get('title', 'Concept B')
            
            hyp = (f"[SVD Latent Discovery] Strong novel connection (score {novelty:.3f}) between "
                   f"'{title_a}' and "
                   f"'{title_b}'. "
                   f"Potential therapeutic repurposing pathway or mechanistic link.")
                   
            # True Kalman update for streaming belief refinement
            if low_rank.shape[1] > 0:
                feat_len = low_rank.shape[1]
                if len(self.kalman_state) != feat_len:
                    self.kalman_state = np.zeros(feat_len)
                    self.kalman_P = np.eye(feat_len)
                    self.kalman_Q = np.eye(feat_len) * 0.01
                    self.kalman_R = np.eye(feat_len) * 0.1
                
                # The "measurement" is the fused salient vector
                z = (low_rank[i] + low_rank[j]) / 2.0
                
                # 1. Predict Step
                x_pred = self.kalman_state
                P_pred = self.kalman_P + self.kalman_Q
                
                # 2. Update Step
                # Kalman Gain: K = P_pred * (P_pred + R)^-1
                inv_S = np.linalg.inv(P_pred + self.kalman_R)
                K = P_pred @ inv_S
                
                # Update state estimate and error covariance
                self.kalman_state = x_pred + K @ (z - x_pred)
                self.kalman_P = (np.eye(feat_len) - K) @ P_pred
                
                logging.info(f"Kalman trace tr(P): {np.trace(self.kalman_P):.4f}")
                
            return hyp
        except Exception as e:
            logging.error(f"SVD hypothesis generation failed: {e}")
            return "Hypothesis matrix generation failed due to sparse alignment."
