use crate::layer::Layer;
use crate::mat::Mat;

/// ReLU activation function
/// f(x) = max(0, x)
pub struct ReLU;

impl Layer for ReLU {
    /// Forward: ReLU(x) = max(0, x)
    fn forward(&self, input: &Mat) -> Mat {
        input.map(|x| if x > 0.0 { x } else { 0.0 })
    }

    /// Backward: dL/dX = dL/dY * (x > 0 ? 1 : 0)
    fn backward(&mut self, input: &Mat, grad_output: &Mat) -> Mat {
        let mask = input.map(|x| if x > 0.0 { 1.0 } else { 0.0 });
        grad_output.mul(&mask)
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
    fn test_relu_forward() {
        let relu = ReLU;
        let input = Mat::from_slice(&[&[-2.0, -1.0, 0.0, 1.0, 2.0]]);
        let output = relu.forward(&input);

        assert_eq!(output.get(0, 0), 0.0);
        assert_eq!(output.get(0, 1), 0.0);
        assert_eq!(output.get(0, 2), 0.0);
        assert_eq!(output.get(0, 3), 1.0);
        assert_eq!(output.get(0, 4), 2.0);
    }

    #[test]
    fn test_relu_backward() {
        let mut relu = ReLU;
        let input = Mat::from_slice(&[&[-1.0, 0.0, 1.0, 2.0]]);
        let grad_output = Mat::from_slice(&[&[1.0, 1.0, 1.0, 1.0]]);

        let grad_input = relu.backward(&input, &grad_output);

        assert_eq!(grad_input.get(0, 0), 0.0); // x < 0
        assert_eq!(grad_input.get(0, 1), 0.0); // x = 0
        assert_eq!(grad_input.get(0, 2), 1.0); // x > 0
        assert_eq!(grad_input.get(0, 3), 1.0); // x > 0
    }
}
