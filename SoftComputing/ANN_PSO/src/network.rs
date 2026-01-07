use crate::{Layer, Linear, Mat, Sigmoid};
use crate::config::{ModelWeights, SavedModel};

/// XOR Network: 2-2-1 architecture
/// 2 inputs -> 2 hidden neurons -> 1 output
pub struct XorNetwork {
    pub linear1: Linear,   // 2 -> 2
    pub sigmoid1: Sigmoid,
    pub linear2: Linear,   // 2 -> 1
    pub sigmoid2: Sigmoid,
}

/// Cache for backpropagation
pub struct XorCache {
    pub x: Mat,      // Original input
    pub z1: Mat,     // linear1 output (before sigmoid)
    pub h1: Mat,     // sigmoid1 output
    pub z2: Mat,     // linear2 output (before sigmoid)
}

impl XorNetwork {
    pub fn new() -> Self {
        Self {
            linear1: Linear::new(2, 2),
            sigmoid1: Sigmoid,
            linear2: Linear::new(2, 1),
            sigmoid2: Sigmoid,
        }
    }

    pub fn forward(&self, x: &Mat) -> Mat {
        let x = self.linear1.forward(x);
        let x = self.sigmoid1.forward(&x);
        let x = self.linear2.forward(&x);
        self.sigmoid2.forward(&x)
    }

    pub fn forward_with_cache(&self, x: &Mat) -> (Mat, XorCache) {
        let z1 = self.linear1.forward(x);
        let h1 = self.sigmoid1.forward(&z1);
        let z2 = self.linear2.forward(&h1);
        let y = self.sigmoid2.forward(&z2);

        let cache = XorCache {
            x: x.clone(),
            z1,
            h1,
            z2,
        };
        (y, cache)
    }

    pub fn backward(&mut self, cache: &XorCache, grad_output: &Mat) {
        let grad = self.sigmoid2.backward(&cache.z2, grad_output);
        let grad = self.linear2.backward(&cache.h1, &grad);
        let grad = self.sigmoid1.backward(&cache.z1, &grad);
        let _ = self.linear1.backward(&cache.x, &grad);
    }

    pub fn apply_grads(&mut self, lr: f64) {
        self.linear1.apply_grads(lr);
        self.linear2.apply_grads(lr);
    }

    pub fn param_count(&self) -> usize {
        self.linear1.param_count() + self.linear2.param_count()
        // sigmoid has 0 parameters
    }

    pub fn get_params(&self) -> Vec<f64> {
        let mut params = self.linear1.get_params();
        params.extend(self.linear2.get_params());
        params
    }

    pub fn set_params(&mut self, params: &[f64]) {
        let consumed = self.linear1.set_params(params);
        self.linear2.set_params(&params[consumed..]);
    }

    /// Get weights as ModelWeights struct for saving
    pub fn get_model_weights(&self) -> ModelWeights {
        let params = self.get_params();
        ModelWeights {
            linear1_weight: params[0..4].to_vec(),
            linear1_bias: params[4..6].to_vec(),
            linear2_weight: params[6..8].to_vec(),
            linear2_bias: params[8..9].to_vec(),
        }
    }

    /// Set weights from ModelWeights struct (for loading)
    pub fn set_model_weights(&mut self, weights: &ModelWeights) {
        let mut params = Vec::new();
        params.extend(&weights.linear1_weight);
        params.extend(&weights.linear1_bias);
        params.extend(&weights.linear2_weight);
        params.extend(&weights.linear2_bias);
        self.set_params(&params);
    }

    /// Create SavedModel for serialization
    pub fn to_saved_model(&self, optimizer: &str, final_loss: f64, iterations: usize) -> SavedModel {
        SavedModel {
            architecture: "2-2-1".to_string(),
            optimizer: optimizer.to_string(),
            final_loss,
            iterations,
            weights: self.get_model_weights(),
        }
    }

    /// Load weights from SavedModel
    pub fn from_saved_model(model: &SavedModel) -> Self {
        let mut network = Self::new();
        network.set_model_weights(&model.weights);
        network
    }
}

impl Default for XorNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// XOR training data
pub fn xor_data() -> (Mat, Mat) {
    let x = Mat::from_slice(&[
        &[0.0, 0.0],
        &[0.0, 1.0],
        &[1.0, 0.0],
        &[1.0, 1.0],
    ]);
    let y = Mat::from_slice(&[&[0.0], &[1.0], &[1.0], &[0.0]]);
    (x, y)
}

/// Test cases from assignment
pub fn test_cases() -> Vec<(f64, f64)> {
    vec![
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),
    ]
}
