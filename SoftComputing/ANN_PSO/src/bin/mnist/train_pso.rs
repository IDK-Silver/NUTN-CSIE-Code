use ann_pso::{
    Pso, PsoConfig, MnistPsoYamlConfig,
    Dataset, MnistDataset, Model, MnistNetwork,
    cross_entropy, save_loss_history, plot_loss_curve,
};
use rand::seq::SliceRandom;

const OUTPUT_DIR: &str = "blob/mnist/train/pso";

fn main() {
    println!("=== MNIST: PSO Training ===\n");

    // Load config
    let config = MnistPsoYamlConfig::load_default()
        .expect("Failed to load config/mnist/pso/default.yaml");

    println!("Configuration loaded from: config/mnist/pso/default.yaml");
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Particles: {}", config.num_particles);
    println!("  Inertia (w): {}", config.w);
    println!("  Cognitive (c1): {}", config.c1);
    println!("  Social (c2): {}", config.c2);
    println!("  Position range: [{}, {}]", config.pos_min, config.pos_max);
    println!("  Max velocity: {}", config.vel_max);
    println!("  Max iterations: {}", config.max_iter);
    println!("  Target loss: {}", config.target_loss);
    println!("  Batch size: {}", config.batch_size);
    println!();

    // Load dataset
    println!("Loading MNIST dataset...");
    let dataset = MnistDataset::load()
        .expect("Failed to load MNIST. Run 'cargo run --bin mnist-download' first.");

    println!("Dataset: {} (train: {}, test: {})",
             dataset.name(), dataset.train_len(), dataset.test_len());

    // Create network
    let mut network = MnistNetwork::new(config.hidden_size);
    println!("Network: {} ({} parameters)", network.name(), network.param_count());
    println!("Activation: ReLU (hidden), Softmax (output)");
    println!("Loss: Cross-Entropy\n");

    println!("WARNING: PSO with {} parameters is slow. Consider using SGD.\n",
             network.param_count());

    // Create PSO
    let pso_config = PsoConfig {
        num_particles: config.num_particles,
        dim: network.param_count(),
        w: config.w,
        c1: config.c1,
        c2: config.c2,
        pos_min: config.pos_min,
        pos_max: config.pos_max,
        vel_max: config.vel_max,
    };

    let mut pso = Pso::new(pso_config);

    // Create index array for random sampling
    let mut indices: Vec<usize> = (0..dataset.train_len()).collect();
    let mut rng = rand::thread_rng();

    // Fitness function using mini-batch
    let fitness_fn = |params: &[f64], indices: &[usize], batch_size: usize,
                      network: &mut MnistNetwork, dataset: &MnistDataset| -> f64 {
        network.set_params(params);
        let batch = dataset.train_subset(indices[0], batch_size.min(dataset.train_len()));
        let pred = network.forward(&batch.x);
        cross_entropy(&pred, &batch.y)
    };

    // Initialize PSO
    indices.shuffle(&mut rng);
    pso.init(|params| {
        fitness_fn(params, &indices, config.batch_size, &mut network, &dataset)
    });

    println!("Iteration | Loss     | Accuracy");
    println!("----------|----------|----------");

    let mut loss_history = Vec::new();
    let mut final_iter = 0;

    for iter in 0..config.max_iter {
        // Shuffle indices for each iteration
        indices.shuffle(&mut rng);
        let start_idx = indices[0];

        pso.step(|params| {
            network.set_params(params);
            let batch = dataset.train_subset(start_idx, config.batch_size.min(dataset.train_len() - start_idx));
            let pred = network.forward(&batch.x);
            cross_entropy(&pred, &batch.y)
        });

        let best_loss = pso.best_fitness();
        loss_history.push(best_loss);
        final_iter = iter + 1;

        if iter % 50 == 0 || iter == config.max_iter - 1 {
            // Evaluate accuracy on a subset
            network.set_params(pso.best_position());
            let eval_batch = dataset.train_subset(0, 1000.min(dataset.train_len()));
            let pred = network.forward(&eval_batch.x);
            let pred_labels = pred.argmax_rows();
            let true_labels = eval_batch.y.argmax_rows();
            let correct: usize = pred_labels.iter().zip(true_labels.iter())
                .filter(|(p, t)| p == t).count();
            let accuracy = 100.0 * correct as f64 / pred_labels.len() as f64;

            println!("{:9} | {:.6} | {:5.2}%", iter + 1, best_loss, accuracy);
        }

        if best_loss < config.target_loss {
            println!("{:9} | {:.6} (converged!)", iter + 1, best_loss);
            break;
        }
    }

    // Apply best weights
    network.set_params(pso.best_position());
    let final_loss = pso.best_fitness();

    // Save outputs
    std::fs::create_dir_all(OUTPUT_DIR).expect("Failed to create output directory");

    // Save loss history
    let csv_path = format!("{}/loss.csv", OUTPUT_DIR);
    save_loss_history(&loss_history, &csv_path)
        .expect("Failed to save loss history");
    println!("\nLoss history saved to: {}", csv_path);

    // Plot loss curve
    let png_path = format!("{}/loss.png", OUTPUT_DIR);
    plot_loss_curve(&loss_history, &png_path, "MNIST PSO Loss Curve")
        .expect("Failed to plot loss curve");
    println!("Loss curve saved to: {}", png_path);

    // Save model
    let model = network.to_saved_model("pso", final_loss, final_iter);
    let model_path = format!("{}/model.json", OUTPUT_DIR);
    model.save(&model_path).expect("Failed to save model");
    println!("Model saved to: {}", model_path);

    // Print final results
    println!("\n--- Results ---");
    println!("Final loss: {:.6}", final_loss);
    println!("Iterations: {}", final_iter);
    println!("Parameters: {}", network.param_count());

    // Evaluate on test set
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
