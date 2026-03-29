use hdf5::{file::File, Result, H5Type};
use ndarray::{s, Array1};
use std::path::Path;

#[derive(H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct ChunkMetadata {
    pub source_file: hdf5::types::VarLenUnicode,
    pub domain:      hdf5::types::VarLenUnicode,
    pub timestamp:   i64,
}

pub struct H5KnowledgeBase {
    file: File,
}

impl H5KnowledgeBase {
    pub fn open_or_create(path: &str) -> hdf5::Result<Self> {
        let file = if Path::new(path).exists() {
            File::append(path)?
        } else {
            let file = File::create(path)?;
            file.create_group("chunks")?;
            file.create_group("domains")?;
            file.create_group("indexes")?;

            let chunks_group = file.group("chunks")?;

            chunks_group
                .new_dataset::<hdf5::types::VarLenUnicode>()
                .chunk(1000)
                .shape(hdf5::Extent::resizable(0))
                .create("text")?;

            chunks_group
                .new_dataset::<f32>()
                .chunk(1000 * 768)
                .shape(hdf5::Extent::resizable(0))
                .create("embeddings")?;

            chunks_group
                .new_dataset::<ChunkMetadata>()
                .chunk(1000)
                .shape(hdf5::Extent::resizable(0))
                .create("metadata")?;

            chunks_group
                .new_dataset::<bool>()
                .chunk(1000)
                .shape(hdf5::Extent::resizable(0))
                .create("is_verified")?;
                
            chunks_group
                .new_dataset::<hdf5::types::VarLenUnicode>()
                .chunk(1000)
                .shape(hdf5::Extent::resizable(0))
                .create("topic")?;
                
            chunks_group
                .new_dataset::<hdf5::types::VarLenUnicode>()
                .chunk(1000)
                .shape(hdf5::Extent::resizable(0))
                .create("structure")?;

            file
        };

        // Guarantee is_verified, topic, and structure exist in previously created files
        if let Ok(chunks_group) = file.group("chunks") {
            let rows = if let Ok(text_ds) = chunks_group.dataset("text") { text_ds.shape()[0] } else { 0 };
            
            if chunks_group.dataset("is_verified").is_err() {
                if let Ok(ds) = chunks_group
                    .new_dataset::<bool>()
                    .chunk(1000)
                    .shape(hdf5::Extent::resizable(0))
                    .create("is_verified")
                {
                    if rows > 0 {
                        let _ = ds.resize(rows);
                        let false_vec = vec![false; rows];
                        let _ = ds.write_slice(
                            &ndarray::Array1::from_vec(false_vec),
                            ndarray::s![0..rows],
                        );
                    }
                }
            }
            
            if chunks_group.dataset("topic").is_err() {
                if let Ok(ds) = chunks_group
                    .new_dataset::<hdf5::types::VarLenUnicode>()
                    .chunk(1000)
                    .shape(hdf5::Extent::resizable(0))
                    .create("topic")
                {
                    if rows > 0 {
                        let _ = ds.resize(rows);
                        let empty_str: hdf5::types::VarLenUnicode = "Unknown".parse().unwrap();
                        let empty_vec = vec![empty_str; rows];
                        let _ = ds.write_slice(
                            &ndarray::Array1::from_vec(empty_vec),
                            ndarray::s![0..rows],
                        );
                    }
                }
            }
            
            if chunks_group.dataset("structure").is_err() {
                if let Ok(ds) = chunks_group
                    .new_dataset::<hdf5::types::VarLenUnicode>()
                    .chunk(1000)
                    .shape(hdf5::Extent::resizable(0))
                    .create("structure")
                {
                    if rows > 0 {
                        let _ = ds.resize(rows);
                        let empty_str: hdf5::types::VarLenUnicode = "Unknown".parse().unwrap();
                        let empty_vec = vec![empty_str; rows];
                        let _ = ds.write_slice(
                            &ndarray::Array1::from_vec(empty_vec),
                            ndarray::s![0..rows],
                        );
                    }
                }
            }
        }

        Ok(H5KnowledgeBase { file })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn append_chunk(
        &self,
        text:        &str,
        embedding:   &[f32],
        source_file: &str,
        domain:      &str,
        timestamp:   i64,
        is_verified: bool,
        topic:       &str,
        structure:   &str,
    ) -> hdf5::Result<()> {
        let chunks_group  = self.file.group("chunks")?;
        let ds_text       = chunks_group.dataset("text")?;
        let ds_embeddings = chunks_group.dataset("embeddings")?;
        let ds_metadata   = chunks_group.dataset("metadata")?;

        let current_rows = ds_text.shape()[0];
        let new_rows     = current_rows + 1;

        ds_text.resize(new_rows)?;
        ds_metadata.resize(new_rows)?;

        // Write text
        let safe_text = text.replace('\0', "");
        let hl_str: hdf5::types::VarLenUnicode = safe_text.parse().unwrap();
        ds_text.write_slice(&[hl_str], s![current_rows..new_rows])?;

        // Write embedding (flat 1D)
        let current_emb = ds_embeddings.shape()[0];
        let new_emb     = current_emb + 768; // Or 384 based on your model size
        // Note: from the logs, it looks like it's a 384-dimensional embedding, but the DB was initialized with 768. 
        // FAISS proxy returned 384 lengths. We'll use embedding.len() dynamically for safety if it's 1D, 
        // but the proxy_server usually emits 384 and HDF5 was chunked at 768.
        let emb_len = embedding.len();
        let exact_new_emb = current_emb + emb_len;
        ds_embeddings.resize(exact_new_emb)?;
        let emb_array = Array1::from_vec(embedding.to_vec());
        ds_embeddings.write_slice(&emb_array, s![current_emb..exact_new_emb])?;

        // Write metadata 
        let meta = ChunkMetadata {
            source_file: source_file.parse().unwrap(),
            domain:      domain.parse().unwrap(),
            timestamp,
        };
        ds_metadata.write_slice(&[meta], s![current_rows..new_rows])?;

        // Write is_verified
        if let Ok(ds_verified) = chunks_group.dataset("is_verified") {
            ds_verified.resize(new_rows)?;
            let v = Array1::from_vec(vec![is_verified]);
            ds_verified.write_slice(&v, s![current_rows..new_rows])?;
        }
        
        // Write topic
        if let Ok(ds_topic) = chunks_group.dataset("topic") {
            ds_topic.resize(new_rows)?;
            let t: hdf5::types::VarLenUnicode = topic.parse().unwrap_or_else(|_| "Unknown".parse().unwrap());
            let v = Array1::from_vec(vec![t]);
            ds_topic.write_slice(&v, s![current_rows..new_rows])?;
        }
        
        // Write structure
        if let Ok(ds_struct) = chunks_group.dataset("structure") {
            ds_struct.resize(new_rows)?;
            let s_val: hdf5::types::VarLenUnicode = structure.parse().unwrap_or_else(|_| "Unknown".parse().unwrap());
            let v = Array1::from_vec(vec![s_val]);
            ds_struct.write_slice(&v, s![current_rows..new_rows])?;
        }

        Ok(())
    }

    pub fn get_chunk_text(&self, index: usize) -> Result<String> {
        let chunks_group = self.file.group("chunks")?;
        let ds_text      = chunks_group.dataset("text")?;
        let data: Array1<hdf5::types::VarLenUnicode> =
            ds_text.read_slice_1d(s![index..index + 1])?;
        Ok(data[0].as_str().to_string())
    }
}
