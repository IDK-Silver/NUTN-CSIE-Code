use ann_pso::{
    SgdYamlConfig,
    Dataset, XorDataset, Model, GradientModel, XorNetwork,
    mse, mse_grad, save_loss_history, plot_loss_curve,
};
use std::env;

fn get_output_dir(optimizer: &str) -> String {
    format!("blob/xor/train/gradient-descent/{}", optimizer)
}

fn get_config_path(optimizer: &str) -> String {
    format!("config/xor/gradient-descent/{}/default.yaml", optimizer)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let optimizer = args.get(1).map(|s| s.as_str()).unwrap_or("sgd");

    match optimizer {
        "sgd" => train_sgd(),
        _ => {
            eprintln!("Unknown optimizer: {}", optimizer);
            eprintln!("Usage: cargo run --bin xor-train-gd -- <optimizer>");
            eprintln!("Available optimizers: sgd");
            std::process::exit(1);
        }
    }
}

fn train_sgd() {
    println!("=== XOR: SGD Training ===\n");

    let config_path = get_config_path("sgd");
    let output_dir = get_output_dir("sgd");

    // Load config
    let config = SgdYamlConfig::load(&config_path)
        .expect(&format!("Failed to load {}", config_path));

    println!("Configuration loaded from: {}", config_path);
    println!("  Learning rate: {}", config.lr);
    println!("  Max iterations: {}", config.max_iter);
    println!("  Target loss: {}", config.target_loss);
    println!();

    // Load dataset
    let dataset = XorDataset::new();
    let train = dataset.train_data();
    println!("Dataset: {} ({} samples)", dataset.name(), train.len());

    // Create network
    let mut network = XorNetwork::new();
    println!("Network: {} ({} parameters)", network.name(), network.param_count());
    println!("Activation: Sigmoid");
    println!("Loss: MSE = 0.5 * sum((y - y_hat)^2)\n");

    println!("Iteration | Loss");
    println!("----------|--------");

    let mut loss_history = Vec::new();
    let mut final_iter = 0;
    let mut final_loss = 0.0;

    for iter in 0..config.max_iter {
        let (pred, cache) = network.forward_with_cache(&train.x);
        let loss = mse(&pred, &train.y);
        loss_history.push(loss);
        final_iter = iter + 1;
        final_loss = loss;

        if iter % 100 == 0 || iter == config.max_iter - 1 {
            println!("{:9} | {:.6}", iter + 1, loss);
        }

        if loss < config.target_loss {
            println!("{:9} | {:.6} (converged!)", iter + 1, loss);
            break;
        }

        let grad = mse_grad(&pred, &train.y);
        network.backward(&cache, &grad);
        network.apply_grads(config.lr);
    }

    // Save outputs
    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    // Save loss history
    let csv_path = format!("{}/loss.csv", output_dir);
    save_loss_history(&loss_history, &csv_path)
        .expect("Failed to save loss history");
    println!("\nLoss history saved to: {}", csv_path);

    // Plot loss curve
    let png_path = format!("{}/loss.png", output_dir);
    plot_loss_curve(&loss_history, &png_path, "XOR SGD Loss Curve")
        .expect("Failed to plot loss curve");
    println!("Loss curve saved to: {}", png_path);

    // Save model
    let model = network.to_saved_model("sgd", final_loss, final_iter);
    let model_path = format!("{}/model.json", output_dir);
    model.save(&model_path).expect("Failed to save model");
    println!("Model saved to: {}", model_path);

    // Print results
    println!("\n--- Results ---");
    println!("Final loss: {:.6}", final_loss);
    println!("Iterations: {}", final_iter);
    println!("\nBest weights:");
    let params = network.get_params();
    println!("  linear1.weight: [{:.4}, {:.4}, {:.4}, {:.4}]",
             params[0], params[1], params[2], params[3]);
    println!("  linear1.bias: [{:.4}, {:.4}]", params[4], params[5]);
    println!("  linear2.weight: [{:.4}, {:.4}]", params[6], params[7]);
    println!("  linear2.bias: [{:.4}]", params[8]);

    // Verify on training data
    println!("\n--- Training Data Predictions ---");
    println!("Input  | Target | Prediction");
    println!("-------|--------|------------");
    for i in 0..train.len() {
        let pred = network.forward(&train.x);
        println!("({}, {}) | {:6} | {:.4}",
                 train.x.get(i, 0) as i32, train.x.get(i, 1) as i32,
                 train.y.get(i, 0) as i32, pred.get(i, 0));
    }
}
