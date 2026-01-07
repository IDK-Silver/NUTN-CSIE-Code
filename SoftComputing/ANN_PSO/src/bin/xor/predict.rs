use ann_pso::{
    Dataset, XorDataset, Model, XorNetwork, SavedModel,
};
use std::env;
use std::fs::{self, File};
use std::io::Write;

fn get_model_path(optimizer: &str) -> String {
    match optimizer {
        "pso" => "blob/xor/train/pso/model.json".to_string(),
        _ => format!("blob/xor/train/gradient-descent/{}/model.json", optimizer),
    }
}

fn get_output_dir(optimizer: &str) -> String {
    match optimizer {
        "pso" => "blob/xor/predict/pso".to_string(),
        _ => format!("blob/xor/predict/gradient-descent/{}", optimizer),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let optimizer = args.get(1).map(|s| s.as_str()).unwrap_or("pso");

    let model_path = get_model_path(optimizer);
    let output_dir = get_output_dir(optimizer);

    println!("=== XOR: Prediction with {} Model ===\n", optimizer.to_uppercase());

    // Load model
    let model = match SavedModel::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model from {}: {}", model_path, e);
            eprintln!("\nMake sure you have trained the model first:");
            match optimizer {
                "pso" => eprintln!("  cargo run --bin xor-train-pso"),
                _ => eprintln!("  cargo run --bin xor-train-gd -- {}", optimizer),
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

    // Load dataset
    let dataset = XorDataset::new();
    let train = dataset.train_data();

    // Verify on training data
    println!("--- Training Data Verification ---");
    println!("Input     | Target | Prediction");
    println!("----------|--------|------------");
    for i in 0..train.len() {
        let pred = network.forward(&train.x);
        println!("({}, {})   | {}      | {:.4}",
                 train.x.get(i, 0) as i32, train.x.get(i, 1) as i32,
                 train.y.get(i, 0) as i32, pred.get(i, 0));
    }

    // Test data from assignment
    if let Some(test) = dataset.test_data() {
        println!("\n--- Test Results ---");
        println!("Input         | Prediction");
        println!("--------------|------------");

        let pred = network.forward(&test.x);
        let mut results = Vec::new();

        for i in 0..test.len() {
            let x1 = test.x.get(i, 0);
            let x2 = test.x.get(i, 1);
            let prediction = pred.get(i, 0);
            println!("({}, {})    | {:.4}", x1, x2, prediction);
            results.push((x1, x2, prediction));
        }

        // Save results to CSV
        fs::create_dir_all(&output_dir).expect("Failed to create output directory");
        let csv_path = format!("{}/results.csv", output_dir);
        let mut file = File::create(&csv_path).expect("Failed to create results file");
        writeln!(file, "x1,x2,prediction").unwrap();
        for (x1, x2, pred) in &results {
            writeln!(file, "{},{},{}", x1, x2, pred).unwrap();
        }
        println!("\nResults saved to: {}", csv_path);
    }
}
