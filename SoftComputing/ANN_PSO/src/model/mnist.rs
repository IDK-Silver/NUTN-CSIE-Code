use crate::{Layer, Linear, Mat, ReLU, Softmax};
use crate::config::{MnistModelWeights, MnistSavedModel};
use super::{Model, GradientModel};

/// MNIST Network: 784 -> hidden -> 10 MLP
/// 784 inputs (28x28 flattened) -> hidden neurons (ReLU) -> 10 outputs (Softmax)
pub struct MnistNetwork {
    pub linear1: Linear,    // 784 -> hidden
    pub relu1: ReLU,
    pub linear2: Linear,    // hidden -> 10
    pub softmax: Softmax,
    pub hidden_size: usize,
}

/// Cache for backpropagation
pub struct MnistCache {
    pub x: Mat,       // Original input
    pub z1: Mat,      // linear1 output (before ReLU)
    pub h1: Mat,      // ReLU output
    pub z2: Mat,      // linear2 output (before softmax)
}

impl MnistNetwork {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            linear1: Linear::new(784, hidden_size),
            relu1: ReLU,
            linear2: Linear::new(hidden_size, 10),
            softmax: Softmax,
            hidden_size,
        }
    }

    /// Get weights as MnistModelWeights struct for saving
    pub fn get_model_weights(&self) -> MnistModelWeights {
        let params1 = self.linear1.get_params();
        let params2 = self.linear2.get_params();

        // linear1: 784*hidden weights + hidden biases
        let l1_weights_end = 784 * self.hidden_size;
        let l1_bias_end = l1_weights_end + self.hidden_size;

        // linear2: hidden*10 weights + 10 biases
        let l2_weights_end = self.hidden_size * 10;

        MnistModelWeights {
            hidden_size: self.hidden_size,
            linear1_weight: params1[0..l1_weights_end].to_vec(),
            linear1_bias: params1[l1_weights_end..l1_bias_end].to_vec(),
            linear2_weight: params2[0..l2_weights_end].to_vec(),
            linear2_bias: params2[l2_weights_end..].to_vec(),
        }
    }

    /// Set weights from MnistModelWeights struct
    pub fn set_model_weights(&mut self, weights: &MnistModelWeights) {
        let mut params1 = Vec::new();
        params1.extend(&weights.linear1_weight);
        params1.extend(&weights.linear1_bias);
        self.linear1.set_params(&params1);

        let mut params2 = Vec::new();
        params2.extend(&weights.linear2_weight);
        params2.extend(&weights.linear2_bias);
        self.linear2.set_params(&params2);
    }

    /// Create saved model for serialization
    pub fn to_saved_model(&self, optimizer: &str, final_loss: f64, iterations: usize) -> MnistSavedModel {
        MnistSavedModel {
            architecture: self.name().to_string(),
            optimizer: optimizer.to_string(),
            final_loss,
            iterations,
            weights: self.get_model_weights(),
        }
    }

    /// Load from saved model
    pub fn from_saved_model(model: &MnistSavedModel) -> Self {
        let mut network = Self::new(model.weights.hidden_size);
        network.set_model_weights(&model.weights);
        network
    }
}

impl Model for MnistNetwork {
    fn forward(&self, x: &Mat) -> Mat {
        let z1 = self.linear1.forward(x);
        let h1 = self.relu1.forward(&z1);
        let z2 = self.linear2.forward(&h1);
        self.softmax.forward(&z2)
    }

    fn param_count(&self) -> usize {
        self.linear1.param_count() + self.linear2.param_count()
    }

    fn get_params(&self) -> Vec<f64> {
        let mut params = self.linear1.get_params();
        params.extend(self.linear2.get_params());
        params
    }

    fn set_params(&mut self, params: &[f64]) {
        let consumed = self.linear1.set_params(params);
        self.linear2.set_params(&params[consumed..]);
    }

    fn name(&self) -> &str {
        match self.hidden_size {
            32 => "784-32-10",
            64 => "784-64-10",
            128 => "784-128-10",
            256 => "784-256-10",
            _ => "784-h-10",
        }
    }
}

impl GradientModel for MnistNetwork {
    type Cache = MnistCache;

    fn forward_with_cache(&self, x: &Mat) -> (Mat, MnistCache) {
        let z1 = self.linear1.forward(x);
        let h1 = self.relu1.forward(&z1);
        let z2 = self.linear2.forward(&h1);
        let y = self.softmax.forward(&z2);

        let cache = MnistCache {
            x: x.clone(),
            z1,
            h1,
            z2,
        };
        (y, cache)
    }

    fn backward(&mut self, cache: &MnistCache, grad_output: &Mat) {
        // Note: grad_output is already (pred - target) from cross_entropy_softmax_grad
        // Softmax backward is identity when combined with cross-entropy
        let grad = self.softmax.backward(&cache.z2, grad_output);
        let grad = self.linear2.backward(&cache.h1, &grad);
        let grad = self.relu1.backward(&cache.z1, &grad);
        let _ = self.linear1.backward(&cache.x, &grad);
    }

    fn apply_grads(&mut self, lr: f64) {
        self.linear1.apply_grads(lr);
        self.linear2.apply_grads(lr);
    }
}
