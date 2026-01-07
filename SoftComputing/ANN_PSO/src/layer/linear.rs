use crate::layer::Layer;
use crate::mat::Mat;

/// Fully connected (linear) layer
/// Y = X @ W + B
pub struct Linear {
    /// Weight matrix (in_features, out_features)
    pub weight: Mat,
    /// Bias vector (1, out_features)
    pub bias: Mat,
    /// Gradient for weights (same shape as weight)
    grad_weight: Mat,
    /// Gradient for bias (same shape as bias)
    grad_bias: Mat,
}

impl Linear {
    /// Create a layer with random weights in [-1, 1]
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self::new_with_range(in_features, out_features, -1.0, 1.0)
    }

    /// Create a layer with specified weight range
    pub fn new_with_range(in_features: usize, out_features: usize, min: f64, max: f64) -> Self {
        Self {
            weight: Mat::random(in_features, out_features, min, max),
            bias: Mat::random(1, out_features, min, max),
            grad_weight: Mat::zeros(in_features, out_features),
            grad_bias: Mat::zeros(1, out_features),
        }
    }
}

impl Layer for Linear {
    /// Forward: Y = X @ W + B
    fn forward(&self, input: &Mat) -> Mat {
        input.matmul(&self.weight).add(&self.bias)
    }

    /// Backward propagation
    /// Given dL/dY (grad_output), compute:
    /// dL/dX = dL/dY @ W^T
    /// dL/dW = X^T @ dL/dY
    /// dL/dB = sum(dL/dY, axis=0)
    fn backward(&mut self, input: &Mat, grad_output: &Mat) -> Mat {
        // grad_input = grad_output @ weight.T
        let grad_input = grad_output.matmul(&self.weight.transpose());

        // grad_weight = input.T @ grad_output
        self.grad_weight = input.transpose().matmul(grad_output);

        // grad_bias = sum over batch (axis=0)
        self.grad_bias = grad_output.sum_axis(0);

        grad_input
    }

    /// Parameter count: in_features * out_features + out_features
    fn param_count(&self) -> usize {
        self.weight.data.len() + self.bias.data.len()
    }

    /// Get parameters: [weight row-major..., bias...]
    fn get_params(&self) -> Vec<f64> {
        let mut params = self.weight.data.clone();
        params.extend(&self.bias.data);
        params
    }

    /// Set parameters from slice
    fn set_params(&mut self, params: &[f64]) -> usize {
        let weight_len = self.weight.data.len();
        let bias_len = self.bias.data.len();

        self.weight.data.copy_from_slice(&params[..weight_len]);
        self.bias.data.copy_from_slice(&params[weight_len..weight_len + bias_len]);

        weight_len + bias_len
    }

    /// Get gradients (same order as get_params)
    fn get_grads(&self) -> Vec<f64> {
        let mut grads = self.grad_weight.data.clone();
        grads.extend(&self.grad_bias.data);
        grads
    }

    /// Update parameters: param -= lr * grad
    fn apply_grads(&mut self, lr: f64) {
        for i in 0..self.weight.data.len() {
            self.weight.data[i] -= lr * self.grad_weight.data[i];
        }
        for i in 0..self.bias.data.len() {
            self.bias.data[i] -= lr * self.grad_bias.data[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        let mut layer = Linear::new(2, 3);
        // Set known weights for testing
        layer.weight = Mat::from_slice(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        layer.bias = Mat::from_slice(&[&[0.1, 0.2, 0.3]]);

        let input = Mat::from_slice(&[&[1.0, 2.0]]);
        let output = layer.forward(&input);

        assert_eq!(output.rows, 1);
        assert_eq!(output.cols, 3);
        // [1, 2] @ [[1,2,3],[4,5,6]] + [0.1, 0.2, 0.3] = [9.1, 12.2, 15.3]
        assert!((output.get(0, 0) - 9.1).abs() < 1e-10);
        assert!((output.get(0, 1) - 12.2).abs() < 1e-10);
        assert!((output.get(0, 2) - 15.3).abs() < 1e-10);
    }

    #[test]
    fn test_linear_param_count() {
        let layer = Linear::new(2, 3);
        // 2*3 weights + 3 biases = 9
        assert_eq!(layer.param_count(), 9);
    }

    #[test]
    fn test_linear_backward() {
        let mut layer = Linear::new(2, 3);
        // Set known weights: W = [[1, 2, 3], [4, 5, 6]]
        layer.weight = Mat::from_slice(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        layer.bias = Mat::from_slice(&[&[0.1, 0.2, 0.3]]);

        let input = Mat::from_slice(&[&[1.0, 2.0]]);
        let grad_output = Mat::from_slice(&[&[1.0, 1.0, 1.0]]);

        // Backward: grad_input = grad_output @ W^T
        // W^T = [[1, 4], [2, 5], [3, 6]]
        // grad_input = [1, 1, 1] @ [[1, 4], [2, 5], [3, 6]] = [6, 15]
        let grad_input = layer.backward(&input, &grad_output);

        assert_eq!(grad_input.rows, 1);
        assert_eq!(grad_input.cols, 2);
        assert!((grad_input.get(0, 0) - 6.0).abs() < 1e-10);
        assert!((grad_input.get(0, 1) - 15.0).abs() < 1e-10);

        // Check grad_weight = input^T @ grad_output
        // input^T = [[1], [2]]
        // grad_weight = [[1], [2]] @ [[1, 1, 1]] = [[1, 1, 1], [2, 2, 2]]
        let grads = layer.get_grads();
        assert!((grads[0] - 1.0).abs() < 1e-10); // grad_weight[0,0]
        assert!((grads[1] - 1.0).abs() < 1e-10); // grad_weight[0,1]
        assert!((grads[2] - 1.0).abs() < 1e-10); // grad_weight[0,2]
        assert!((grads[3] - 2.0).abs() < 1e-10); // grad_weight[1,0]
        assert!((grads[4] - 2.0).abs() < 1e-10); // grad_weight[1,1]
        assert!((grads[5] - 2.0).abs() < 1e-10); // grad_weight[1,2]

        // Check grad_bias = sum(grad_output, axis=0) = [1, 1, 1]
        assert!((grads[6] - 1.0).abs() < 1e-10); // grad_bias[0]
        assert!((grads[7] - 1.0).abs() < 1e-10); // grad_bias[1]
        assert!((grads[8] - 1.0).abs() < 1e-10); // grad_bias[2]
    }

    #[test]
    fn test_linear_get_set_params() {
        let mut layer = Linear::new(2, 3);

        // Set specific params
        let params: Vec<f64> = (1..=9).map(|x| x as f64).collect();
        layer.set_params(&params);

        // Get params back
        let retrieved = layer.get_params();
        assert_eq!(retrieved.len(), 9);

        for (i, &val) in retrieved.iter().enumerate() {
            assert!(
                (val - (i + 1) as f64).abs() < 1e-10,
                "Param {} mismatch: expected {}, got {}",
                i,
                i + 1,
                val
            );
        }

        // Verify weights and bias are correctly set
        // weights: 1-6, bias: 7-9
        assert!((layer.weight.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((layer.weight.get(1, 2) - 6.0).abs() < 1e-10);
        assert!((layer.bias.get(0, 0) - 7.0).abs() < 1e-10);
        assert!((layer.bias.get(0, 2) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_apply_grads() {
        let mut layer = Linear::new(2, 2);

        // Set initial weights
        layer.weight = Mat::from_slice(&[&[1.0, 2.0], &[3.0, 4.0]]);
        layer.bias = Mat::from_slice(&[&[0.5, 0.5]]);

        // Perform backward to set gradients
        let input = Mat::from_slice(&[&[1.0, 1.0]]);
        let grad_output = Mat::from_slice(&[&[1.0, 1.0]]);
        layer.backward(&input, &grad_output);

        // Apply gradients with lr = 0.1
        let lr = 0.1;
        layer.apply_grads(lr);

        // grad_weight = input^T @ grad_output = [[1], [1]] @ [[1, 1]] = [[1, 1], [1, 1]]
        // new_weight = weight - lr * grad_weight
        // new_weight[0,0] = 1.0 - 0.1 * 1.0 = 0.9
        assert!((layer.weight.get(0, 0) - 0.9).abs() < 1e-10);
        assert!((layer.weight.get(0, 1) - 1.9).abs() < 1e-10);
        assert!((layer.weight.get(1, 0) - 2.9).abs() < 1e-10);
        assert!((layer.weight.get(1, 1) - 3.9).abs() < 1e-10);

        // grad_bias = [1, 1]
        // new_bias = bias - lr * grad_bias = [0.5 - 0.1, 0.5 - 0.1] = [0.4, 0.4]
        assert!((layer.bias.get(0, 0) - 0.4).abs() < 1e-10);
        assert!((layer.bias.get(0, 1) - 0.4).abs() < 1e-10);
    }
}
