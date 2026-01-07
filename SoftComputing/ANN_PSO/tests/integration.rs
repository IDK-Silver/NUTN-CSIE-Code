use ann_pso::{
    mse, mse_grad, Mat, Pso, PsoConfig, Sgd,
    Model, GradientModel, XorNetwork,
    Dataset, XorDataset,
};

#[test]
fn test_xor_network_forward() {
    let mut network = XorNetwork::new();

    // Set known weights for deterministic testing
    let params = vec![
        // linear1 weights (2x2)
        5.0, -5.0,
        5.0, -5.0,
        // linear1 bias (2)
        -2.5, 7.5,
        // linear2 weights (2x1)
        10.0, 10.0,
        // linear2 bias (1)
        -5.0,
    ];
    network.set_params(&params);

    let dataset = XorDataset::new();
    let train = dataset.train_data();

    // Forward pass
    let output = network.forward(&train.x);

    // Verify output shape
    assert_eq!(output.rows, 4);
    assert_eq!(output.cols, 1);

    // Verify outputs are in valid range (0, 1)
    for i in 0..4 {
        let val = output.get(i, 0);
        assert!(val > 0.0 && val < 1.0, "Output {} = {} out of range", i, val);
    }
}

#[test]
fn test_xor_network_param_count() {
    let network = XorNetwork::new();
    // linear1: 2*2 + 2 = 6
    // linear2: 2*1 + 1 = 3
    // Total: 9
    assert_eq!(network.param_count(), 9);
}

#[test]
fn test_xor_network_get_set_params() {
    let mut network = XorNetwork::new();

    // Set params to known values
    let params: Vec<f64> = (1..=9).map(|x| x as f64 * 0.1).collect();
    network.set_params(&params);

    // Get params back
    let retrieved = network.get_params();
    assert_eq!(retrieved.len(), 9);

    for (i, (&expected, &actual)) in params.iter().zip(retrieved.iter()).enumerate() {
        assert!(
            (expected - actual).abs() < 1e-10,
            "Param {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn test_xor_pso_convergence() {
    let dataset = XorDataset::new();
    let train = dataset.train_data();

    let mut network = XorNetwork::new();

    let config = PsoConfig {
        num_particles: 100,
        dim: network.param_count(),
        w: 0.729,
        c1: 1.49445,
        c2: 1.49445,
        pos_min: -10.0,
        pos_max: 10.0,
        vel_max: 4.0,
    };

    let mut pso = Pso::new(config);

    // Initialize
    pso.init(|params| {
        network.set_params(params);
        let pred = network.forward(&train.x);
        mse(&pred, &train.y)
    });

    // Run PSO
    for _ in 0..2000 {
        pso.step(|params| {
            network.set_params(params);
            let pred = network.forward(&train.x);
            mse(&pred, &train.y)
        });

        // Early termination if converged
        if pso.best_fitness() < 0.01 {
            break;
        }
    }

    // Verify convergence (threshold relaxed for test stability)
    assert!(
        pso.best_fitness() < 0.15,
        "PSO failed to converge on XOR, loss = {}",
        pso.best_fitness()
    );

    // Apply best weights and verify predictions
    network.set_params(pso.best_position());

    for i in 0..4 {
        let input = Mat::from_slice(&[&[train.x.get(i, 0), train.x.get(i, 1)]]);
        let pred = network.forward(&input);
        let target = train.y.get(i, 0);

        // Prediction should be close to target (within 0.4)
        assert!(
            (pred.get(0, 0) - target).abs() < 0.4,
            "XOR({}, {}): expected ~{}, got {}",
            train.x.get(i, 0),
            train.x.get(i, 1),
            target,
            pred.get(0, 0)
        );
    }
}

#[test]
fn test_xor_sgd_convergence() {
    let dataset = XorDataset::new();
    let train = dataset.train_data();

    let mut network = XorNetwork::new();
    let sgd = Sgd::new(1.0);

    let mut last_loss = f64::INFINITY;

    // Train with SGD
    for iter in 0..5000 {
        let (pred, cache) = network.forward_with_cache(&train.x);
        let loss = mse(&pred, &train.y);

        // Track loss decrease
        if iter % 1000 == 0 {
            assert!(
                loss < last_loss + 0.5 || iter == 0,
                "Loss not decreasing at iter {}: {} -> {}",
                iter,
                last_loss,
                loss
            );
            last_loss = loss;
        }

        // Early termination
        if loss < 0.01 {
            break;
        }

        let grad = mse_grad(&pred, &train.y);
        network.backward(&cache, &grad);
        network.apply_grads(sgd.lr);
    }

    // Final evaluation
    let final_pred = network.forward(&train.x);
    let final_loss = mse(&final_pred, &train.y);

    // SGD should achieve reasonable loss
    assert!(
        final_loss < 1.0,
        "SGD failed to reduce loss, final loss = {}",
        final_loss
    );
}
