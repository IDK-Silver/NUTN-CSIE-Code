use ann_pso::{
    Dataset, MnistDataset, Model, MnistNetwork, MnistSavedModel,
};
use std::env;
use std::fs::{self, File};
use std::io::Write;

fn get_model_path(optimizer: &str) -> String {
    match optimizer {
        "pso" => "blob/mnist/train/pso/model.json".to_string(),
        _ => format!("blob/mnist/train/gradient-descent/{}/model.json", optimizer),
    }
}

fn get_output_dir(optimizer: &str) -> String {
    match optimizer {
        "pso" => "blob/mnist/predict/pso".to_string(),
        _ => format!("blob/mnist/predict/gradient-descent/{}", optimizer),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let optimizer = args.get(1).map(|s| s.as_str()).unwrap_or("sgd");

    let model_path = get_model_path(optimizer);
    let output_dir = get_output_dir(optimizer);

    println!("=== MNIST: Prediction with {} Model ===\n", optimizer.to_uppercase());

    // Load model
    let model = match MnistSavedModel::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model from {}: {}", model_path, e);
            eprintln!("\nMake sure you have trained the model first:");
            match optimizer {
                "pso" => eprintln!("  cargo run --bin mnist-train-pso"),
                _ => eprintln!("  cargo run --bin mnist-train-gd -- {}", optimizer),
            }
            std::process::exit(1);
        }
    };

    println!("Model loaded from: {}", model_path);
    println!("  Architecture: {}", model.architecture);
    println!("  Hidden size: {}", model.weights.hidden_size);
    println!("  Optimizer: {}", model.optimizer);
    println!("  Final loss: {:.6}", model.final_loss);
    println!("  Iterations: {}", model.iterations);
    println!();

    // Create network and load weights
    let network = MnistNetwork::from_saved_model(&model);

    // Load dataset
    println!("Loading MNIST dataset...");
    let dataset = MnistDataset::load()
        .expect("Failed to load MNIST. Run 'cargo run --bin mnist-download' first.");

    // Evaluate on test set
    let test = dataset.test_data().unwrap();
    println!("Test set: {} samples\n", test.x.rows);

    println!("Evaluating...");
    let pred = network.forward(&test.x);
    let pred_labels = pred.argmax_rows();
    let true_labels = test.y.argmax_rows();

    // Calculate accuracy
    let correct: usize = pred_labels.iter().zip(true_labels.iter())
        .filter(|(p, t)| p == t).count();
    let accuracy = 100.0 * correct as f64 / test.x.rows as f64;

    println!("\n--- Test Results ---");
    println!("Accuracy: {:.2}% ({}/{})", accuracy, correct, test.x.rows);

    // Calculate per-class accuracy
    println!("\n--- Per-Class Accuracy ---");
    println!("Digit | Correct | Total | Accuracy");
    println!("------|---------|-------|----------");

    let mut class_correct = [0usize; 10];
    let mut class_total = [0usize; 10];

    for (pred, true_label) in pred_labels.iter().zip(true_labels.iter()) {
        class_total[*true_label] += 1;
        if pred == true_label {
            class_correct[*true_label] += 1;
        }
    }

    for digit in 0..10 {
        let acc = if class_total[digit] > 0 {
            100.0 * class_correct[digit] as f64 / class_total[digit] as f64
        } else {
            0.0
        };
        println!("  {}   | {:7} | {:5} | {:7.2}%",
                 digit, class_correct[digit], class_total[digit], acc);
    }

    // Confusion matrix summary (errors)
    println!("\n--- Common Errors (Top 10) ---");
    let mut errors: Vec<(usize, usize, usize)> = Vec::new(); // (true, pred, count)
    let mut error_counts = [[0usize; 10]; 10];

    for (pred, true_label) in pred_labels.iter().zip(true_labels.iter()) {
        if pred != true_label {
            error_counts[*true_label][*pred] += 1;
        }
    }

    for true_label in 0..10 {
        for pred_label in 0..10 {
            if error_counts[true_label][pred_label] > 0 {
                errors.push((true_label, pred_label, error_counts[true_label][pred_label]));
            }
        }
    }

    errors.sort_by(|a, b| b.2.cmp(&a.2));
    println!("True -> Predicted | Count");
    println!("------------------|------");
    for (true_label, pred_label, count) in errors.iter().take(10) {
        println!("   {} -> {}          | {}",
                 true_label, pred_label, count);
    }

    // Save results
    fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    let csv_path = format!("{}/results.csv", output_dir);
    let mut file = File::create(&csv_path).expect("Failed to create results file");
    writeln!(file, "index,true_label,predicted_label,correct").unwrap();
    for (i, (pred, true_label)) in pred_labels.iter().zip(true_labels.iter()).enumerate() {
        writeln!(file, "{},{},{},{}", i, true_label, pred, if pred == true_label { 1 } else { 0 }).unwrap();
    }
    println!("\nResults saved to: {}", csv_path);

    // Save summary
    let summary_path = format!("{}/summary.txt", output_dir);
    let mut file = File::create(&summary_path).expect("Failed to create summary file");
    writeln!(file, "MNIST Prediction Summary").unwrap();
    writeln!(file, "========================").unwrap();
    writeln!(file, "Model: {}", model_path).unwrap();
    writeln!(file, "Architecture: {}", model.architecture).unwrap();
    writeln!(file, "Optimizer: {}", model.optimizer).unwrap();
    writeln!(file, "Training loss: {:.6}", model.final_loss).unwrap();
    writeln!(file, "").unwrap();
    writeln!(file, "Test Results").unwrap();
    writeln!(file, "------------").unwrap();
    writeln!(file, "Accuracy: {:.2}% ({}/{})", accuracy, correct, test.x.rows).unwrap();
    println!("Summary saved to: {}", summary_path);
}
