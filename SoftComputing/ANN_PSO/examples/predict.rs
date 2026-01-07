use ann_pso::{Mat, XorNetwork, SavedModel, xor_data, test_cases};
use std::env;
use std::fs::{self, File};
use std::io::Write;

fn get_model_path(optimizer: &str) -> String {
    match optimizer {
        "pso" => "blob/train/pso/model.json".to_string(),
        _ => format!("blob/train/gradient-descent/{}/model.json", optimizer),
    }
}

fn get_output_dir(optimizer: &str) -> String {
    match optimizer {
        "pso" => "blob/predict/pso".to_string(),
        _ => format!("blob/predict/gradient-descent/{}", optimizer),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let optimizer = args.get(1).map(|s| s.as_str()).unwrap_or("pso");

    let model_path = get_model_path(optimizer);
    let output_dir = get_output_dir(optimizer);

    println!("=== Prediction with {} Model ===\n", optimizer.to_uppercase());

    // Load model
    let model = match SavedModel::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model from {}: {}", model_path, e);
            eprintln!("\nMake sure you have trained the model first:");
            match optimizer {
                "pso" => eprintln!("  cargo run --example train_pso"),
                _ => eprintln!("  cargo run --example train_gd -- {}", optimizer),
            }
            std::process::exit(1);
        }
    };

    println!("Model loaded from: {}", model_path);
    println!("  Architecture: {}", model.architecture);
    println!("  Optimizer: {}", model.optimizer);
    println!("  Final loss: {:.6}", model.final_loss);
    println!("  Iterations: {}", model.iterations);
    println!();

    // Create network and load weights
    let network = XorNetwork::from_saved_model(&model);

    // Verify on training data
    let (x, y) = xor_data();
    println!("--- Training Data Verification ---");
    println!("Input     | Target | Prediction");
    println!("----------|--------|------------");
    for i in 0..4 {
        let input = Mat::from_slice(&[&[x.get(i, 0), x.get(i, 1)]]);
        let pred = network.forward(&input);
        println!("({}, {})   | {}      | {:.4}",
                 x.get(i, 0) as i32, x.get(i, 1) as i32,
                 y.get(i, 0) as i32, pred.get(0, 0));
    }

    // Test cases from assignment
    println!("\n--- Test Results ---");
    println!("Input         | Prediction");
    println!("--------------|------------");

    let mut results = Vec::new();
    for (a, b) in test_cases() {
        let input = Mat::from_slice(&[&[a, b]]);
        let pred = network.forward(&input);
        let prediction = pred.get(0, 0);
        println!("({}, {})    | {:.4}", a, b, prediction);
        results.push((a, b, prediction));
    }

    // Save results to CSV
    fs::create_dir_all(&output_dir).expect("Failed to create output directory");
    let csv_path = format!("{}/results.csv", output_dir);
    let mut file = File::create(&csv_path).expect("Failed to create results file");
    writeln!(file, "x1,x2,prediction").unwrap();
    for (a, b, pred) in &results {
        writeln!(file, "{},{},{}", a, b, pred).unwrap();
    }
    println!("\nResults saved to: {}", csv_path);
}
