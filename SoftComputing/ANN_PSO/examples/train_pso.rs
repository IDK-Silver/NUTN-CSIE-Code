use ann_pso::{
    Mat, Pso, PsoConfig, XorNetwork,
    PsoYamlConfig, mse, xor_data,
    save_loss_history, plot_loss_curve,
};

const OUTPUT_DIR: &str = "blob/train/pso";

fn main() {
    println!("=== PSO Training for XOR Problem ===\n");

    // Load config
    let config = PsoYamlConfig::load_default()
        .expect("Failed to load config/pso/default.yaml");

    println!("Configuration loaded from: config/pso/default.yaml");
    println!("  Particles: {}", config.num_particles);
    println!("  Inertia (w): {}", config.w);
    println!("  Cognitive (c1): {}", config.c1);
    println!("  Social (c2): {}", config.c2);
    println!("  Position range: [{}, {}]", config.pos_min, config.pos_max);
    println!("  Max velocity: {}", config.vel_max);
    println!("  Max iterations: {}", config.max_iter);
    println!("  Target loss: {}", config.target_loss);
    println!();

    // XOR data
    let (x, y) = xor_data();

    // Create network
    let mut network = XorNetwork::new();
    println!("Network: 2-2-1 ({} parameters)", network.param_count());
    println!("Activation: Sigmoid");
    println!("Loss: MSE = 0.5 * sum((y - y_hat)^2)\n");

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

    // Initialize PSO
    pso.init(|params| {
        network.set_params(params);
        let pred = network.forward(&x);
        mse(&pred, &y)
    });

    println!("Iteration | Loss");
    println!("----------|--------");

    let mut loss_history = Vec::new();
    let mut final_iter = 0;

    for iter in 0..config.max_iter {
        pso.step(|params| {
            network.set_params(params);
            let pred = network.forward(&x);
            mse(&pred, &y)
        });

        let best_loss = pso.best_fitness();
        loss_history.push(best_loss);
        final_iter = iter + 1;

        if iter % 500 == 0 || iter == config.max_iter - 1 {
            println!("{:9} | {:.6}", iter + 1, best_loss);
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
    plot_loss_curve(&loss_history, &png_path, "PSO Loss Curve")
        .expect("Failed to plot loss curve");
    println!("Loss curve saved to: {}", png_path);

    // Save model
    let model = network.to_saved_model("pso", final_loss, final_iter);
    let model_path = format!("{}/model.json", OUTPUT_DIR);
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
    println!("\n--- Predictions on Training Data ---");
    println!("Input     | Target | Prediction");
    println!("----------|--------|------------");
    for i in 0..4 {
        let input = Mat::from_slice(&[&[x.get(i, 0), x.get(i, 1)]]);
        let pred = network.forward(&input);
        println!("({}, {})   | {}      | {:.4}",
                 x.get(i, 0) as i32, x.get(i, 1) as i32,
                 y.get(i, 0) as i32, pred.get(0, 0));
    }
}
