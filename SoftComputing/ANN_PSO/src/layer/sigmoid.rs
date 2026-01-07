use crate::layer::Layer;
use crate::mat::Mat;

/// Sigmoid activation function
/// No parameters, stateless.
pub struct Sigmoid;

impl Layer for Sigmoid {
    /// Forward: σ(x) = 1 / (1 + e^(-x))
    fn forward(&self, input: &Mat) -> Mat {
        input.map(|x| 1.0 / (1.0 + (-x).exp()))
    }

    /// Backward: dL/dX = dL/dY ⊙ σ(X) ⊙ (1 - σ(X))
    fn backward(&mut self, input: &Mat, grad_output: &Mat) -> Mat {
        let sig = self.forward(input); // Recompute sigmoid
        let one_minus_sig = sig.map(|s| 1.0 - s);
        grad_output.mul(&sig).mul(&one_minus_sig)
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
    fn test_sigmoid_forward() {
        let sigmoid = Sigmoid;
        let input = Mat::from_slice(&[&[0.0, 1.0, -1.0]]);
        let output = sigmoid.forward(&input);

        assert!((output.get(0, 0) - 0.5).abs() < 1e-10);
        assert!(output.get(0, 1) > 0.5);
        assert!(output.get(0, 2) < 0.5);
    }

    #[test]
    fn test_sigmoid_output_range() {
        let sigmoid = Sigmoid;
        let input = Mat::from_slice(&[&[-10.0, -5.0, 0.0, 5.0, 10.0]]);
        let output = sigmoid.forward(&input);

        for &val in &output.data {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_sigmoid_backward() {
        let mut sigmoid = Sigmoid;

        // Test at x = 0 where sigmoid(0) = 0.5
        // Derivative: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        // At x = 0: sigmoid'(0) = 0.5 * 0.5 = 0.25
        let input = Mat::from_slice(&[&[0.0]]);
        let grad_output = Mat::from_slice(&[&[1.0]]); // upstream gradient = 1

        let grad_input = sigmoid.backward(&input, &grad_output);

        // grad_input = grad_output * sigmoid(x) * (1 - sigmoid(x))
        // = 1.0 * 0.5 * 0.5 = 0.25
        assert!(
            (grad_input.get(0, 0) - 0.25).abs() < 1e-10,
            "Expected 0.25, got {}",
            grad_input.get(0, 0)
        );
    }

    #[test]
    fn test_sigmoid_derivative_at_zero() {
        let mut sigmoid = Sigmoid;

        // Verify derivative at various points
        let inputs = vec![0.0, 1.0, -1.0, 2.0, -2.0];

        for x in inputs {
            let input = Mat::from_slice(&[&[x]]);
            let grad_output = Mat::from_slice(&[&[1.0]]);

            let grad_input = sigmoid.backward(&input, &grad_output);

            // Manually compute expected derivative
            let sig_x = 1.0 / (1.0 + (-x).exp());
            let expected = sig_x * (1.0 - sig_x);

            assert!(
                (grad_input.get(0, 0) - expected).abs() < 1e-10,
                "Derivative at x={}: expected {}, got {}",
                x,
                expected,
                grad_input.get(0, 0)
            );
        }
    }
}
