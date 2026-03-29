use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::Index;
use std::path::Path;
use std::fs;
use tantivy::TantivyDocument;

pub struct Bm25Index {
    index: Index,
    schema: Schema,
}

impl Bm25Index {
    pub fn new(index_path: &str) -> Self {
        let mut schema_builder = Schema::builder();
        let body = schema_builder.add_text_field("body", TEXT | STORED);
        let path_field = schema_builder.add_text_field("filepath", STRING | STORED);
        let schema = schema_builder.build();

        if !Path::new(index_path).exists() {
            fs::create_dir_all(index_path).unwrap();
        }

        let index = Index::create_in_dir(index_path, schema.clone())
            .or_else(|_| Index::open_in_dir(index_path))
            .expect("Failed to create or open index");

        Bm25Index { index, schema }
    }

    pub fn add_document(&self, filepath: &str, content: &str) {
        let mut index_writer = self.index.writer(50_000_000).unwrap();
        let body = self.schema.get_field("body").unwrap();
        let path_field = self.schema.get_field("filepath").unwrap();

        let mut doc = TantivyDocument::default();
        doc.add_text(body, content);
        doc.add_text(path_field, filepath);

        index_writer.add_document(doc).unwrap();
        index_writer.commit().unwrap();
    }

    pub fn search(&self, query_str: &str, top_k: usize) -> Vec<(f32, String, String)> {
        let reader = self.index.reader().unwrap();
        let searcher = reader.searcher();
        
        let body = self.schema.get_field("body").unwrap();
        let path_field = self.schema.get_field("filepath").unwrap();

        let query_parser = QueryParser::for_index(&self.index, vec![body]);
        let query = query_parser.parse_query(query_str).unwrap();

        let top_docs = searcher.search(&query, &TopDocs::with_limit(top_k)).unwrap();

        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let retrieved_doc: TantivyDocument = searcher.doc(doc_address).unwrap();
            let text = retrieved_doc.get_first(body).unwrap().as_str().unwrap().to_string();
            let path = retrieved_doc.get_first(path_field).unwrap().as_str().unwrap().to_string();
            results.push((score, path, text));
        }

        results
    }
}
