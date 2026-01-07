/// Simple SGD optimizer without momentum
/// Usage is manual: user calls layer.apply_grads(sgd.lr) for each layer.
/// This keeps SGD decoupled from network structure.
pub struct Sgd {
    pub lr: f64,
}

impl Sgd {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::Layer;
    use crate::layer::Linear;
    use crate::mat::Mat;

    #[test]
    fn test_sgd_new() {
        let sgd = Sgd::new(0.01);
        assert!((sgd.lr - 0.01).abs() < 1e-10);

        let sgd2 = Sgd::new(0.5);
        assert!((sgd2.lr - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sgd_with_linear() {
        // Test SGD integration with Linear layer
        let mut layer = Linear::new(2, 2);
        layer.weight = Mat::from_slice(&[&[1.0, 2.0], &[3.0, 4.0]]);
        layer.bias = Mat::from_slice(&[&[0.0, 0.0]]);

        let sgd = Sgd::new(0.1);

        // Perform forward and backward
        let input = Mat::from_slice(&[&[1.0, 1.0]]);
        let _ = layer.forward(&input);

        // Set gradient manually via backward
        let grad_output = Mat::from_slice(&[&[1.0, 1.0]]);
        layer.backward(&input, &grad_output);

        // Get weights before update
        let w00_before = layer.weight.get(0, 0);

        // Apply SGD update
        layer.apply_grads(sgd.lr);

        // Verify weights changed
        let w00_after = layer.weight.get(0, 0);
        assert!(
            (w00_before - w00_after).abs() > 1e-10,
            "Weight should have changed after SGD update"
        );

        // grad_weight[0,0] = input[0,0] * grad_output[0,0] = 1.0 * 1.0 = 1.0
        // new_weight[0,0] = old_weight[0,0] - lr * grad = 1.0 - 0.1 * 1.0 = 0.9
        assert!(
            (w00_after - 0.9).abs() < 1e-10,
            "Expected 0.9, got {}",
            w00_after
        );
    }
}
