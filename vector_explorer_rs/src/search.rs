// search.rs — live search and filter logic

use crate::data::Point;
use std::collections::HashSet;

pub struct SearchState {
    pub query:           String,
    pub matched_indices: HashSet<usize>,
    pub match_count:     usize,
    pub filter_column:   FilterColumn,
    pub filter_value:    String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FilterColumn {
    None,
    Topic,
    Structure,
    Domain,
    SourceFile,
    Verified,
}

impl SearchState {
    pub fn new() -> Self {
        Self {
            query:           String::new(),
            matched_indices: HashSet::new(),
            match_count:     0,
            filter_column:   FilterColumn::None,
            filter_value:    String::new(),
        }
    }

    /// Run search across all text columns
    pub fn run_search(&mut self, dataset: &crate::data::Dataset) {
        let q = self.query.to_lowercase();
        self.matched_indices.clear();

        if q.is_empty() {
            self.match_count = 0;
            return;
        }

        for (idx, point) in dataset.points.iter().enumerate() {
            if point.topic.to_lowercase().contains(&q)
                || point.structure.to_lowercase().contains(&q)
                || point.domain.to_lowercase().contains(&q)
                || point.source_file.to_lowercase().contains(&q)
                || point.text_preview.to_lowercase().contains(&q)
            {
                self.matched_indices.insert(idx);
            }
        }

        self.match_count = self.matched_indices.len();
    }

    /// Returns a closure that filters points based on active filter
    pub fn filter_fn(&self) -> impl Fn(&Point) -> bool + '_ {
        let col = self.filter_column.clone();
        let val = self.filter_value.to_lowercase();

        move |point: &Point| -> bool {
            if val.is_empty() { return true; }
            match col {
                FilterColumn::None       => true,
                FilterColumn::Topic      => point.topic.to_lowercase()       == val,
                FilterColumn::Structure  => point.structure.to_lowercase()   == val,
                FilterColumn::Domain     => point.domain.to_lowercase()      == val,
                FilterColumn::SourceFile => point.source_file.to_lowercase() == val,
                FilterColumn::Verified   => {
                    if val == "yes" { point.verified }
                    else            { !point.verified }
                },
            }
        }
    }
}
