use ann_pso::{
    MnistSgdYamlConfig,
    Dataset, MnistDataset, Model, GradientModel, MnistNetwork,
    cross_entropy, cross_entropy_softmax_grad, save_loss_history, save_accuracy_history,
};
use rand::seq::SliceRandom;
use std::env;

fn get_output_dir(optimizer: &str) -> String {
    format!("blob/mnist/train/gradient-descent/{}", optimizer)
}

fn get_config_path(optimizer: &str) -> String {
    format!("config/mnist/gradient-descent/{}/default.yaml", optimizer)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let optimizer = args.get(1).map(|s| s.as_str()).unwrap_or("sgd");

    match optimizer {
        "sgd" => train_sgd(),
        _ => {
            eprintln!("Unknown optimizer: {}", optimizer);
            eprintln!("Usage: cargo run --bin mnist-train-gd -- <optimizer>");
            eprintln!("Available optimizers: sgd");
            std::process::exit(1);
        }
    }
}

fn train_sgd() {
    println!("=== MNIST: SGD Training ===\n");

    let config_path = get_config_path("sgd");
    let output_dir = get_output_dir("sgd");

    // Load config
    let config = MnistSgdYamlConfig::load(&config_path)
        .expect(&format!("Failed to load {}", config_path));

    println!("Configuration loaded from: {}", config_path);
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Learning rate: {}", config.lr);
    println!("  Epochs: {}", config.max_iter);
    println!("  Batch size: {}", config.batch_size);
    println!();

    // Load dataset
    println!("Loading MNIST dataset...");
    let dataset = MnistDataset::load()
        .expect("Failed to load MNIST. Run 'cargo run --bin mnist-download' first.");

    let train_len = dataset.train_len();
    println!("Dataset: {} (train: {}, test: {})",
             dataset.name(), train_len, dataset.test_len());

    // Create network
    let mut network = MnistNetwork::new(config.hidden_size);
    println!("Network: {} ({} parameters)", network.name(), network.param_count());
    println!("Activation: ReLU (hidden), Softmax (output)");
    println!("Loss: Cross-Entropy\n");

    println!("Epoch | Train Loss | Train Acc | Test Acc");
    println!("------|------------|-----------|----------");

    let mut loss_history = Vec::new();
    let mut train_acc_history = Vec::new();
    let mut test_acc_history = Vec::new();
    let mut final_epoch = 0;
    let mut final_loss = 0.0;

    // Create index array for shuffling
    let mut indices: Vec<usize> = (0..train_len).collect();
    let mut rng = rand::thread_rng();

    let num_batches = train_len / config.batch_size;

    for epoch in 0..config.max_iter {
        // Shuffle training data each epoch
        indices.shuffle(&mut rng);

        let mut epoch_loss = 0.0;
        let mut epoch_correct = 0;
        let mut epoch_samples = 0;

        for batch_idx in 0..num_batches {
            let start = batch_idx * config.batch_size;

            // Get batch using shuffled indices
            let batch = dataset.train_subset(start, config.batch_size);

            // Forward pass
            let (pred, cache) = network.forward_with_cache(&batch.x);
            let loss = cross_entropy(&pred, &batch.y);
            epoch_loss += loss;

            // Calculate batch accuracy
            let pred_labels = pred.argmax_rows();
            let true_labels = batch.y.argmax_rows();
            epoch_correct += pred_labels.iter().zip(true_labels.iter())
                .filter(|(p, t)| p == t).count();
            epoch_samples += batch.x.rows;

            // Backward pass
            let grad = cross_entropy_softmax_grad(&pred, &batch.y);
            network.backward(&cache, &grad);
            network.apply_grads(config.lr);
        }

        let avg_loss = epoch_loss / num_batches as f64;
        loss_history.push(avg_loss);
        final_epoch = epoch + 1;
        final_loss = avg_loss;

        let train_acc = 100.0 * epoch_correct as f64 / epoch_samples as f64;
        train_acc_history.push(train_acc);

        // Evaluate on test set
        let test = dataset.test_data().unwrap();
        let pred = network.forward(&test.x);
        let pred_labels = pred.argmax_rows();
        let true_labels = test.y.argmax_rows();
        let test_correct: usize = pred_labels.iter().zip(true_labels.iter())
            .filter(|(p, t)| p == t).count();
        let test_acc = 100.0 * test_correct as f64 / test.x.rows as f64;
        test_acc_history.push(test_acc);

        // Print every 5 epochs
        if epoch % 5 == 0 || epoch == config.max_iter - 1 {
            println!("{:5} | {:10.6} | {:8.2}% | {:7.2}%",
                     epoch + 1, avg_loss, train_acc, test_acc);
        }
    }

    // Save outputs
    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    // Save loss history
    let csv_path = format!("{}/loss.csv", output_dir);
    save_loss_history(&loss_history, &csv_path)
        .expect("Failed to save loss history");
    println!("\nLoss history saved to: {}", csv_path);

    // Save accuracy history
    let acc_csv_path = format!("{}/accuracy.csv", output_dir);
    save_accuracy_history(&train_acc_history, &test_acc_history, &acc_csv_path)
        .expect("Failed to save accuracy history");
    println!("Accuracy history saved to: {}", acc_csv_path);

    // Save model
    let model = network.to_saved_model("sgd", final_loss, final_epoch);
    let model_path = format!("{}/model.json", output_dir);
    model.save(&model_path).expect("Failed to save model");
    println!("Model saved to: {}", model_path);

    // Print final results
    println!("\n--- Results ---");
    println!("Final loss: {:.6}", final_loss);
    println!("Epochs: {}", final_epoch);
    println!("Parameters: {}", network.param_count());

    // Final test evaluation
    let test = dataset.test_data().unwrap();
    let pred = network.forward(&test.x);
    let pred_labels = pred.argmax_rows();
    let true_labels = test.y.argmax_rows();
    let correct: usize = pred_labels.iter().zip(true_labels.iter())
        .filter(|(p, t)| p == t).count();
    let accuracy = 100.0 * correct as f64 / test.x.rows as f64;

    println!("\n--- Test Set Evaluation ---");
    println!("Test accuracy: {:.2}% ({}/{})", accuracy, correct, test.x.rows);
}
