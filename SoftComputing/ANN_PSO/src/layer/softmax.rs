use crate::layer::Layer;
use crate::mat::Mat;

/// Softmax activation function for multi-class classification
/// softmax(x_i) = exp(x_i) / sum(exp(x_j))
/// Applied row-wise (each row is a sample)
pub struct Softmax;

impl Layer for Softmax {
    /// Forward: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    /// Subtracting max for numerical stability
    fn forward(&self, input: &Mat) -> Mat {
        let mut result = Mat::zeros(input.rows, input.cols);

        for i in 0..input.rows {
            // Find max for numerical stability
            let row = input.row(i);
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            // Compute exp(x - max) and sum
            let exp_vals: Vec<f64> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f64 = exp_vals.iter().sum();

            // Normalize
            for j in 0..input.cols {
                result.set(i, j, exp_vals[j] / sum);
            }
        }

        result
    }

    /// Backward: When combined with cross-entropy loss, the gradient simplifies to (pred - target)
    /// This is handled in the loss function, so we just pass through the gradient here.
    /// For standalone softmax backprop, we would need the full Jacobian.
    fn backward(&mut self, _input: &Mat, grad_output: &Mat) -> Mat {
        // When used with cross_entropy_softmax_grad, this is identity
        grad_output.clone()
    }

    fn param_count(&self) -> usize {
        0
    }

    fn get_params(&self) -> Vec<f64> {
        vec![]
    }

    fn set_params(&mut self, _params: &[f64]) -> usize {
        0
    }

    fn get_grads(&self) -> Vec<f64> {
        vec![]
    }

    fn apply_grads(&mut self, _lr: f64) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_forward() {
        let softmax = Softmax;
        let input = Mat::from_slice(&[&[1.0, 2.0, 3.0]]);
        let output = softmax.forward(&input);

        // Check sum = 1
        let sum: f64 = output.row(0).iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Check monotonicity (larger input -> larger output)
        assert!(output.get(0, 0) < output.get(0, 1));
        assert!(output.get(0, 1) < output.get(0, 2));
    }

    #[test]
    fn test_softmax_batch() {
        let softmax = Softmax;
        let input = Mat::from_slice(&[
            &[1.0, 2.0, 3.0],
            &[0.0, 0.0, 0.0],
        ]);
        let output = softmax.forward(&input);

        // Check sum = 1 for each row
        for i in 0..output.rows {
            let sum: f64 = output.row(i).iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "Row {} sum = {}", i, sum);
        }

        // Second row should be uniform (all equal)
        let val = output.get(1, 0);
        for j in 1..output.cols {
            assert!((output.get(1, j) - val).abs() < 1e-10);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let softmax = Softmax;
        // Large values that would overflow without max subtraction
        let input = Mat::from_slice(&[&[1000.0, 1001.0, 1002.0]]);
        let output = softmax.forward(&input);

        // Should still be valid probabilities
        let sum: f64 = output.row(0).iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(output.data.iter().all(|&x| x.is_finite() && x > 0.0));
    }
}
