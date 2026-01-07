mod mnist;
mod xor;

pub use mnist::{MnistNetwork, MnistCache};
pub use xor::{XorNetwork, XorCache};

use crate::Mat;

/// Base model trait for all neural networks
///
/// This trait is used by PSO optimizer which doesn't need gradients
pub trait Model {
    /// Forward pass
    fn forward(&self, x: &Mat) -> Mat;

    /// Get total parameter count
    fn param_count(&self) -> usize;

    /// Get all parameters as a flat vector
    fn get_params(&self) -> Vec<f64>;

    /// Set all parameters from a flat vector
    fn set_params(&mut self, params: &[f64]);

    /// Model name/architecture description
    fn name(&self) -> &str;
}

/// Extended model trait for gradient-based training
///
/// This trait is used by SGD, Adam, etc.
pub trait GradientModel: Model {
    /// Cache type for storing intermediate values during forward pass
    type Cache;

    /// Forward pass with cache for backpropagation
    fn forward_with_cache(&self, x: &Mat) -> (Mat, Self::Cache);

    /// Backward pass to compute gradients
    fn backward(&mut self, cache: &Self::Cache, grad_output: &Mat);

    /// Apply accumulated gradients with learning rate
    fn apply_grads(&mut self, lr: f64);
}
