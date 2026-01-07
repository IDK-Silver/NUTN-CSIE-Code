use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

const MNIST_BASE_URL: &str = "https://ossci-datasets.s3.amazonaws.com/mnist";
const DATA_DIR: &str = "blob/mnist/data";

const FILES: [(&str, u64); 4] = [
    ("train-images-idx3-ubyte.gz", 9912422),
    ("train-labels-idx1-ubyte.gz", 28881),
    ("t10k-images-idx3-ubyte.gz", 1648877),
    ("t10k-labels-idx1-ubyte.gz", 4542),
];

fn download_file(filename: &str, expected_size: u64) -> Result<(), Box<dyn std::error::Error>> {
    let filepath = format!("{}/{}", DATA_DIR, filename);
    let path = Path::new(&filepath);

    // Check if file already exists with correct size
    if path.exists() {
        let metadata = fs::metadata(path)?;
        if metadata.len() == expected_size {
            println!("  {} already exists ({}B), skipping", filename, expected_size);
            return Ok(());
        }
    }

    let url = format!("{}/{}", MNIST_BASE_URL, filename);
    println!("  Downloading {} from {}...", filename, url);

    let response = reqwest::blocking::get(&url)?;
    if !response.status().is_success() {
        return Err(format!("HTTP error: {}", response.status()).into());
    }

    let bytes = response.bytes()?;
    let mut file = File::create(path)?;
    file.write_all(&bytes)?;

    println!("  Saved {} ({}B)", filename, bytes.len());
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MNIST Dataset Downloader ===\n");

    // Create data directory
    fs::create_dir_all(DATA_DIR)?;
    println!("Data directory: {}\n", DATA_DIR);

    // Download all files
    println!("Downloading files:");
    for (filename, expected_size) in FILES.iter() {
        download_file(filename, *expected_size)?;
    }

    println!("\nDownload complete!");
    println!("\nFiles:");
    for (filename, _) in FILES.iter() {
        let filepath = format!("{}/{}", DATA_DIR, filename);
        if Path::new(&filepath).exists() {
            let metadata = fs::metadata(&filepath)?;
            println!("  {} ({}B)", filepath, metadata.len());
        }
    }

    Ok(())
}
