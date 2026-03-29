mod retrieval;
mod knowledge_h5;
use notify::{Watcher, RecursiveMode, watcher};
use std::sync::mpsc::channel;
use std::time::Duration;
use std::path::Path;
use std::process::Command;
use std::fs;

use text_splitter::TextSplitter;
// Fastembed removed in favor of proxy_server.py

// Helper function to extract text after OCR/audio transcription
fn extract_text_content(path: &Path) -> Option<String> {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
    let text_exts = ["txt", "md", "csv", "py", "rs", "js", "ts", "jsx", "tsx", "html", "css", "json", "xml", "yml", "yaml", "toml", "ini", "sh", "bat", "ps1", "c", "cpp", "h", "hpp", "java", "go", "rb", "php", "sql"];
    
    if text_exts.contains(&ext.as_str()) {
        if let Ok(content) = fs::read_to_string(path) {
            return Some(content);
        }
    }
    
    let path_str = path.to_str().unwrap();
    
    // Check if Tesseract/Whisper appended .txt (e.g. image.png.txt or video.mp4.wav.txt)
    let appended_txt = format!("{}.txt", path_str);
    if Path::new(&appended_txt).exists() {
        if let Ok(content) = fs::read_to_string(&appended_txt) {
            return Some(content);
        }
    }
    
    let appended_wav_txt = format!("{}.wav.txt", path_str);
    if Path::new(&appended_wav_txt).exists() {
        if let Ok(content) = fs::read_to_string(&appended_wav_txt) {
            return Some(content);
        }
    }
    
    // Check if pdftotext replaced the extension (e.g. docs.pdf -> docs.txt)
    let replaced_txt = path.with_extension("txt");
    if replaced_txt.exists() {
        if let Ok(content) = fs::read_to_string(&replaced_txt) {
            return Some(content);
        }
    }
    
    None
}

fn process_file(path: &Path) {
    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    let path_str = path.to_str().unwrap();

    let output_file = format!("{}.txt", path_str);

    match extension.to_lowercase().as_str() {
        "txt" | "md" | "csv" | "py" | "rs" | "js" | "ts" | "jsx" | "tsx" | "html" | "css" | "json" | "xml" | "yml" | "yaml" | "toml" | "ini" | "sh" | "bat" | "ps1" | "c" | "cpp" | "h" | "hpp" | "java" | "go" | "rb" | "php" | "sql" => {
            println!("Text/Code file detected: {}", path_str);
        }
        "png" | "jpg" | "jpeg" | "webp" | "bmp" | "tiff" | "gif" => {
            println!("Image detected, running Tesseract OCR: {}", path_str);
            let status = Command::new("tesseract")
                .arg(path_str)
                .arg(path_str) // tesseract appends .txt automatically
                .status()
                .expect("Failed to execute tesseract");
            if !status.success() { eprintln!("OCR failed for {}", path_str); return; }
        }
        "pdf" => {
            println!("PDF detected, running pdftotext or pdftoppm+tesseract: {}", path_str);
            let status = Command::new("pdftotext")
                .arg(path_str)
                .status()
                .unwrap_or_else(|_| {
                    Command::new("tesseract").status().expect("Fallback failed")
                });
            if !status.success() { return; }
        }
        "mp3" | "wav" | "mp4" | "mkv" | "avi" | "mov" | "webm" | "flv" | "m4a" | "aac" | "ogg" | "flac" => {
            println!("Audio/Video detected, running whisper.cpp: {}", path_str);
            let wav_path = format!("{}.wav", path_str);
            let ffmpeg_status = Command::new("ffmpeg")
                .arg("-y")
                .arg("-i")
                .arg(path_str)
                .arg("-ar").arg("16000")
                .arg("-ac").arg("1")
                .arg("-c:a").arg("pcm_s16le")
                .arg(&wav_path)
                .status()
                .expect("Failed to execute ffmpeg");

            if ffmpeg_status.success() {
                 let whisper_status = Command::new("whisper-cli")
                    .arg("-m").arg("/path/to/ggml-base.en.bin")
                    .arg("-f").arg(&wav_path)
                    .arg("-otxt")
                    .status()
                    .expect("Failed to execute whisper");
                 let _ = fs::remove_file(wav_path);
                 if !whisper_status.success() { return; }
            }
        }
        "zim" => {
            println!("Native ZIM Archive extraction initiated: {}", path_str);
            let ext_dir = path.parent().unwrap().join(format!("{}_extracted", path.file_stem().unwrap().to_str().unwrap()));
            let _ = fs::create_dir_all(&ext_dir);
            
            if let Ok(zim) = zim::Zim::new(path_str) {
                let mut extracted = 0;
                println!("ZIM Archive loaded natively. Contains {} total articles...", zim.article_count());
                
                for entry in zim.iterate_by_urls() {
                    // Extract only articles
                    if !matches!(entry.namespace, zim::Namespace::Articles) { continue; }
                    
                    if let Some(zim::Target::Cluster(c_idx, b_idx)) = entry.target {
                        if let Ok(cluster) = zim.get_cluster(c_idx) {
                            if let Ok(blob) = cluster.get_blob(b_idx) {
                                if let Ok(html) = std::str::from_utf8(blob.as_ref()) {
                                    if !html.is_empty() {
                                        let plain_text = html2text::from_read(html.as_bytes(), 120).unwrap_or_default();
                                        let mut safe_title = entry.title.replace("/", "_").replace("\\", "_");
                                        if safe_title.is_empty() {
                                            safe_title = entry.url.replace("/", "_").replace("\\", "_");
                                        }
                                        let out_path = ext_dir.join(format!("{}.txt", safe_title));
                                        let _ = fs::write(out_path, plain_text);
                                        extracted += 1;
                                    }
                                }
                            }
                        }
                    }
                    
                    if extracted > 0 && extracted % 5000 == 0 {
                        println!("Natively extracted {} ZIM articles so far...", extracted);
                    }
                }
                println!("ZIM extraction complete. {} articles natively parsed to text. Watcher will naturally ingest the output texts.", extracted);
                let converted_dir = path.parent().unwrap().join("Converted");
                let _ = fs::create_dir_all(&converted_dir);
                let _ = fs::rename(path, converted_dir.join(path.file_name().unwrap()));
            } else {
                eprintln!("Native ZIM extraction failed for {} (Corrupt archive or unsupported format)", path_str);
            }
            return; // Exit because we do not directly ingest the .zim, we let the watcher catch the generated texts
        }
        "epub" => {
            println!("EPUB detected, extracting text structure via python backend: {}", path_str);
            let status = Command::new("/home/saphyre-solutions/Desktop/Projects/Local LLM/OfflineRAG/.venv/bin/python")
                .arg("/home/saphyre-solutions/Desktop/Projects/Local LLM/OfflineRAG/epub_to_txt.py")
                .arg(path_str)
                .arg(&output_file)
                .status()
                .expect("Failed to execute EPUB text extraction subsystem");
            
            if !status.success() { 
                eprintln!("EPUB processing failed for {}", path_str);
                return; 
            }
        }
        _ => {
            println!("Unsupported file format: {}", path_str);
            return;
        }
    }

    // Now extract the final text content
    if let Some(content) = extract_text_content(path) {
        println!("Ingesting content from {} into HDF5 database...", path_str);
        if ingest_into_database(path_str, &content) {
            let converted_dir = path.parent().unwrap().join("Converted");
            let _ = fs::create_dir_all(&converted_dir);
            
            let filename = path.file_name().unwrap();
            let dest_path = converted_dir.join(filename);
            
            match fs::rename(path, &dest_path) {
                Ok(_) => println!("Moved {} to Converted folder.", path_str),
                Err(e) => eprintln!("Failed to move {}: {}", path_str, e),
            }
            
            let txt_path = Path::new(&output_file);
            if txt_path.exists() && txt_path != path {
                let txt_filename = txt_path.file_name().unwrap();
                let _ = fs::rename(txt_path, converted_dir.join(txt_filename));
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LLM LABELING — called once per file at ingestion time
// Returns (topic, structure). One HTTP call to LM Studio per document.
// ─────────────────────────────────────────────────────────────────────────────
fn get_ingestion_label(chunks: &[String], source_file: &str) -> (String, String) {
    let n = chunks.len();
    let step = (n / 3).max(1);
    let samples_text: String = chunks.iter().step_by(step).take(3).enumerate()
        .map(|(i, t)| format!("Sample {}:\n{}", i + 1, &t.chars().take(400).collect::<String>()))
        .collect::<Vec<_>>().join("\n\n---\n\n");

    let filename = std::path::Path::new(source_file)
        .file_name().and_then(|n| n.to_str()).unwrap_or(source_file);

    let prompt = format!(
        "You are a knowledge taxonomist. Analyze these text samples and classify the document.\n\
         \nFilename: {filename}\nSAMPLES:\n{samples_text}\n\
         \nAnswer EXACTLY in this two-line format with no other text:\n\
         TOPIC: [2-6 word maximally specific subject label]\n\
         STRUCTURE: [single word]\n\
         \nFor TOPIC: be specific. Include person's full name if biographical.\n\
         Examples: \"Timothy Spurlin Welding Career\" | \"Ayurvedic Panchakarma Detoxification\" | \"MITRE ATT&CK Lateral Movement\"\n\
         \nFor STRUCTURE pick exactly one: Instructional Reference Narrative Research Data Dialogue Legal Creative Technical Philosophical"
    );

    let body = serde_json::json!({"model": "local-model", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": 100});

    let content_str = reqwest::blocking::Client::new()
        .post("http://127.0.0.1:8888/v1/chat/completions")
        .header("Authorization", "Bearer your_api_key_here")
        .json(&body)
        .send()
        .ok()
        .and_then(|r| r.json::<serde_json::Value>().ok())
        .map(|json| {
            let c = json["choices"][0]["message"]["content"].as_str().unwrap_or("").trim().to_string();
            if c.is_empty() { json["choices"][0]["message"]["reasoning_content"].as_str().unwrap_or("").trim().to_string() } else { c }
        })
        .unwrap_or_default();

    let mut topic = format!("Document: {}", filename);
    let mut structure = "Reference".to_string();
    for line in content_str.lines() {
        let line = line.trim();
        if line.to_uppercase().starts_with("TOPIC:") {
            let raw = line["TOPIC:".len()..].trim().trim_matches('"').trim_matches('\'').trim();
            let w: Vec<&str> = raw.split_whitespace().take(7).collect();
            if !w.is_empty() { topic = w.join(" "); }
        } else if line.to_uppercase().starts_with("STRUCTURE:") {
            structure = line["STRUCTURE:".len()..].trim().split_whitespace().next().unwrap_or("Reference").to_string();
        }
    }
    println!("[Label] {} → topic='{}' structure='{}'", filename, topic, structure);
    (topic, structure)
}

// Ingest content into the HDF5 universal database
fn ingest_into_database(source_file: &str, content: &str) -> bool {
    let db_path = os.environ.get("HDF5_PATH", "./universal_knowledge_base.h5");
    
    // Open the DB
    let h5_db = match knowledge_h5::H5KnowledgeBase::open_or_create(db_path) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("Failed to open HDF5 database: {}", e);
            return false;
        }
    };

    // Check if it's a JSON array or a CSV of QA pairs (from Structured_Prompt_Conversion)
    let mut chunks: Vec<String> = Vec::new();
    
    if source_file.ends_with(".csv") {
        let mut rdr = csv::ReaderBuilder::new().has_headers(true).flexible(true).trim(csv::Trim::All).from_reader(content.as_bytes());
        let headers = rdr.headers().map(|h| h.clone()).unwrap_or_else(|_| csv::StringRecord::new());
        let has_qa_headers = headers.iter().any(|h| h.eq_ignore_ascii_case("question")) && headers.iter().any(|h| h.eq_ignore_ascii_case("answer"));
        
        // Resilience Fallback: If no Q/A headers, restart map natively as generic rows
        let mut data_rdr = if has_qa_headers {
            rdr
        } else {
            csv::ReaderBuilder::new().has_headers(false).flexible(true).trim(csv::Trim::All).from_reader(content.as_bytes())
        };

        for result in data_rdr.records() {
            if let Ok(record) = result {
                if record.len() >= 2 {
                    let q = &record[0];
                    let a = &record[1];
                    if q.is_empty() || a.is_empty() { continue; }
                    chunks.push(format!("Question: {}\nAnswer: {}", q, a));
                } else if record.len() == 1 {
                    let text = &record[0];
                    if !text.is_empty() { chunks.push(text.to_string()); }
                }
            }
        }
    } else if source_file.ends_with(".json") {
        if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(content) {
            if let Some(arr) = json_val.as_array() {
                for item in arr {
                    if let (Some(q), Some(a)) = (item.get("question").and_then(|v| v.as_str()), item.get("answer").and_then(|v| v.as_str())) {
                        chunks.push(format!("Question: {}\nAnswer: {}", q, a));
                    }
                }
            }
        }
    }

    // Fallback to Semantic text chunking (1000 chars per chunk roughly) if not a valid QA JSON
    if chunks.is_empty() {
        let splitter = TextSplitter::new(1000);
        chunks = splitter.chunks(content).map(|s| s.to_string()).collect();
    }

    if chunks.is_empty() { return false; }

    // One LLM call per file — topic+structure applied to all chunks of this file
    let (topic, structure) = get_ingestion_label(&chunks, source_file);

    println!("Created {} chunks. Generating local embeddings in batches of 50...", chunks.len());

    // Make HTTP POSTs to our local FAISS Python proxy to get 'all-mpnet-base-v2' embeddings
    // Set explicit timeout to 5 minutes to prevent FAISS server processing from hitting the default Rust Reqwest timeout bounds
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .unwrap_or_else(|_| reqwest::blocking::Client::new());
        
    let proxy_url = "http://localhost:8000/embed";
    
    let mut raw_embs: Vec<serde_json::Value> = Vec::with_capacity(chunks.len());
    let mut batch_size = 50usize;
    let min_batch = 10usize;
    let max_batch = 150usize;
    let mut current_idx = 0;

    // Adaptive batch logic (Real-Time GGUF hardware tuning)
    while current_idx < chunks.len() {
        let end_idx = (current_idx + batch_size).min(chunks.len());
        let chunk_slice = &chunks[current_idx..end_idx];

        let body = serde_json::json!({
            "texts": chunk_slice
        });

        let start_time = std::time::Instant::now();
        let res = match client.post(proxy_url)
            .header("Authorization", "Bearer your_api_key_here")
            .json(&body)
            .send() {
            Ok(res) => res,
            Err(e) => {
                if e.is_timeout() {
                    batch_size = (batch_size / 2).max(min_batch);
                    println!("Timeout on batch (size {}), backing off to {}", chunk_slice.len(), batch_size);
                    continue; // Retry same window
                }
                eprintln!("Failed to connect to local FAISS proxy at {}: {:?}", proxy_url, e);
                return false;
            }
        };

        if !res.status().is_success() {
             eprintln!("Proxy HTTP Error: {:?}", res.status());
             return false;
        }

        let elapsed = start_time.elapsed().as_millis();
        if elapsed < 2000 && batch_size < max_batch {
            batch_size = (batch_size + 10).min(max_batch);
        } else if elapsed > 8000 && batch_size > min_batch {
            batch_size = (batch_size - 10).max(min_batch);
        }

        let json_data: serde_json::Value = res.json().unwrap_or_default();
        if let Some(embeddings_array) = json_data.get("embeddings").and_then(|v| v.as_array()) {
             raw_embs.extend(embeddings_array.iter().cloned());
             current_idx = end_idx;
        } else {
             eprintln!("Proxy returned empty or malformed embeddings for batch.");
             return false;
        }
    }

    // Write metadata, text, and embeddings atomically to HDF5
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    
    // Extract a domain from the filepath (e.g. parent folder name)
    let domain = Path::new(source_file)
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("general");

    let mut success_count = 0;
    
    // Check if path indicates verified natural remedies
    let is_verified = source_file.contains("NaturalRemedies");
    
    if raw_embs.len() != chunks.len() {
        eprintln!("Fatal Error: Proxy returned {} embeddings, but we sent {} chunks. Aborting insertion to avoid database corruption.", raw_embs.len(), chunks.len());
        return false;
    }

fn sanitize_for_hdf5(text: &str) -> String {
    text.chars()
        .filter(|c| {
            let cp = *c as u32;
            match cp {
                0x0000 => false,
                0x0001..=0x0008 => false,
                0x000B..=0x000C => false,
                0x000E..=0x001F => false,
                0xFFFD => false,
                0xFFFE..=0xFFFF => false,
                _ => c.is_alphanumeric() || c.is_whitespace() || c.is_ascii_punctuation() || cp > 0x007F,
            }
        })
        .collect::<String>()
        .trim()
        .to_string()
}

    for (i, chunk_text) in chunks.iter().enumerate() {
        let safe_text = sanitize_for_hdf5(chunk_text);
        if safe_text.is_empty() {
            println!("Chunk sanitized to empty string, skipping.");
            continue;
        }

        let arr = raw_embs[i].as_array().expect("Expected a JSON array for the embedding");
        let embedding_vec: Vec<f32> = arr.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect();

        match h5_db.append_chunk(&safe_text, &embedding_vec, source_file, domain, timestamp, is_verified, &topic, &structure) {
            Ok(_) => success_count += 1,
            Err(e) => eprintln!("Failed to write chunk to HDF5: {:?}", e),
        }
    }
    
    println!("Successfully appended {} chunks with 384-dimensional embeddings natively to {}.", success_count, db_path);
    success_count > 0
}

use clap::{Parser, Subcommand};
use reqwest::blocking::Client;
use serde_json::json;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the local ingest watcher for the drop folder
    Watch {
        /// The folder to watch containing text, PDFs, or media
        #[arg(short, long, default_value = os.environ.get("KNOWLEDGE_DROP", "./KnowledgeDrop"))]
        folder: String,
    },
    /// Query the knowledge base and generate an answer using LM Studio
    Ask {
        /// The question to ask
        query: String,
        
        /// Port for the LM studio local server
        #[arg(short, long, default_value = "1234")]
        port: u16,
    },
    /// Process a single file directly
    Process {
        /// The file to process
        file: String,
    }
}

fn query_lm_studio(context: &str, question: &str, port: u16) {
    let client = Client::new();
    let url = format!("http://localhost:{}/v1/chat/completions", port);
    
    // Support hot-swappable model selection via LLM Integration Layer
    let model_name = std::env::var("PRIMARY_MODEL").unwrap_or_else(|_| "liquid/lfm2.5-1.2b".to_string());

    let system_prompt = format!(
        "You are a helpful AI assistant running offline. Use the following context to answer the user's question.\n\nContext:\n{}\n",
        context
    );

    let body = json!({
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
    });

    println!("Sending query to LM Studio at {} using model {}...", url, model_name);

    match client.post(&url)
        .header("Authorization", "Bearer your_api_key_here")
        .json(&body)
        .send() {
        Ok(res) => {
            if res.status().is_success() {
                let json: serde_json::Value = res.json().unwrap();
                if let Some(content) = json["choices"][0]["message"]["content"].as_str() {
                    println!("\n=== Answer ===\n{}\n", content);
                }
            } else {
                eprintln!("LM Studio returned an error: {:?}", res.status());
            }
        }
        Err(e) => eprintln!("Failed to connect to LM Studio: {}", e),
    }
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Watch { folder } => {
            let drop_folder = folder.as_str();
            if !Path::new(drop_folder).exists() {
                fs::create_dir_all(drop_folder).expect("Failed to create drop folder");
            }
            
            // Resolve symlinks so the notification engine watches the actual folder
            let watch_path = fs::canonicalize(drop_folder).unwrap_or_else(|_| std::path::PathBuf::from(drop_folder));

            println!("Watching for new files in: {}", watch_path.display());

            let (tx, rx) = channel();
            let mut watcher = watcher(tx, Duration::from_secs(2)).unwrap();

            watcher.watch(&watch_path, RecursiveMode::Recursive).unwrap();

            loop {
                match rx.recv() {
                    Ok(event) => {
                        match event {
                            notify::DebouncedEvent::Create(path) | notify::DebouncedEvent::Write(path) => {
                                if path.exists() && path.is_file() {
                                    let path_str = path.to_str().unwrap();
                                    
                                    // Make sure we're not infinitely reacting to our own Converted files
                                    if !path_str.contains("Converted") && !path_str.ends_with(".tmp") {
                                        println!("New file detected: {:?}", path);
                                        process_file(&path);
                                    }
                                }
                            }
                            _ => {}
                        }
                    },
                    Err(e) => println!("Watch error: {:?}", e),
                }
            }
        },
        Commands::Ask { query, port } => {
            println!("Searching local knowledge base for: '{}'", query);
            
            // NOTE:
            // In a fully flushed out BM25 + Vector integration, here you would:
            // 1. Query the locally built HDF5 blobs or tantivy arrays.
            // 2. Load the corresponding row IDs from HDF5
            // 3. fetch the pristine chunks from H5KnowledgeBase::get_chunk_text()
            
            // For now, testing retrieving our single chunk directly from the DB.
            let db_path = os.environ.get("HDF5_PATH", "./universal_knowledge_base.h5");
            let retrieved_context = if Path::new(db_path).exists() {
                match knowledge_h5::H5KnowledgeBase::open_or_create(db_path) {
                    Ok(db) => {
                        // Assuming the most recent appended text chunk is the context 
                        // representing our highest-ranking semantic search match
                        db.get_chunk_text(0).unwrap_or_else(|err| { eprintln!("HDF5 Read Error: {}", err); String::from("No documents found in the offline knowledge base.") })
                    },
                    Err(_) => String::from("Failed to connect to HDF5 database.")
                }
            } else {
                String::from("Database not found. Please drop files into KnowledgeDrop first!")
            };
            
            println!("Simulated Hybrid Retrieval complete. Generating answer via LM Studio from exact HDF5 chunks...");
            query_lm_studio(&retrieved_context, query, *port);
        },
        Commands::Process { file } => {
            let path = Path::new(file);
            println!("--- OfflineRAG Ingestion Interface ---");
            if path.exists() {
                process_file(path);
                println!("\nIngestion routine complete. You can close this window now.");
            } else {
                eprintln!("File not found or no longer exists: {}", file);
            }
        }
    }
}
