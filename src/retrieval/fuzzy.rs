use rapidfuzz::fuzz;

pub struct FuzzyIndex {
    documents: Vec<(String, String)>, // (filepath, content)
}

impl FuzzyIndex {
    pub fn new() -> Self {
        FuzzyIndex {
            documents: Vec::new(),
        }
    }

    pub fn add_document(&mut self, filepath: &str, content: &str) {
        self.documents.push((filepath.to_string(), content.to_string()));
    }

    pub fn search(&self, query: &str, top_k: usize) -> Vec<(f64, String, String)> {
        let mut results: Vec<(f64, String, String)> = self.documents.iter().map(|(path, content)| {
            let score = fuzz::ratio(query.chars(), content.chars());
            (score, path.clone(), content.clone())
        }).collect();

        // Sort by highest score first
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        results.into_iter().take(top_k).collect()
    }
}
