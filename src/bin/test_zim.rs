#[allow(unused_imports)]
use zim::{Zim, DirectoryEntry};

fn main() {
    let zim = Zim::new("/home/saphyre-solutions/Desktop/Projects/Local LLM/OfflineRAG/test.zim");
    if let Ok(z) = zim {
        println!("Articles: {}", z.article_count());
        for entry in z.iterate_by_urls().take(5) {
            println!("Entry: {:?}", entry.url);
            if let Some(target) = entry.target {
                println!("Target: {:?}", target);
            }
        }
    }
}
