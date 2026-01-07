mod linear;
mod relu;
mod sigmoid;
mod softmax;

pub use linear::Linear;
pub use relu::ReLU;
pub use sigmoid::Sigmoid;
pub use softmax::Softmax;

use crate::mat::Mat;

/// Layer trait for neural network layers
pub trait Layer {
    /// Forward propagation
    /// input: (batch_size, in_features)
    /// output: (batch_size, out_features)
    fn forward(&self, input: &Mat) -> Mat;

    /// Backward propagation
    /// input: the input passed to forward
    /// grad_output: gradient from upstream (batch_size, out_features)
    /// Returns: gradient with respect to input (batch_size, in_features)
    /// Side effect: stores gradients for parameters internally
    fn backward(&mut self, input: &Mat, grad_output: &Mat) -> Mat;

    /// Number of trainable parameters
    fn param_count(&self) -> usize;

    /// Flatten parameters into a vector
    /// Order must be deterministic and documented
    fn get_params(&self) -> Vec<f64>;

    /// Load parameters from a slice
    /// Returns the number of elements consumed
    fn set_params(&mut self, params: &[f64]) -> usize;

    /// Get gradients (same order as get_params)
    fn get_grads(&self) -> Vec<f64>;

    /// Update parameters: param -= lr * grad
    fn apply_grads(&mut self, lr: f64);
}
