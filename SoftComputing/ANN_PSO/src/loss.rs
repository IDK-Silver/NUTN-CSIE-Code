use crate::mat::Mat;

/// Compute MSE loss (0.5 * sum of squared errors)
/// L = (1/2) * sum((y - y_hat)^2)
/// Note: This is sum, not mean. The 1/2 simplifies gradient.
pub fn mse(pred: &Mat, target: &Mat) -> f64 {
    assert_eq!(pred.rows, target.rows);
    assert_eq!(pred.cols, target.cols);

    let mut sum = 0.0;
    for i in 0..pred.data.len() {
        let diff = pred.data[i] - target.data[i];
        sum += diff * diff;
    }
    0.5 * sum
}

/// Compute gradient of MSE loss with respect to predictions
/// dL/dy_hat = y_hat - y
pub fn mse_grad(pred: &Mat, target: &Mat) -> Mat {
    pred.sub(target)
}

/// Cross-entropy loss for multi-class classification
/// L = -mean(sum(target * log(pred + epsilon)))
pub fn cross_entropy(pred: &Mat, target: &Mat) -> f64 {
    assert_eq!(pred.rows, target.rows);
    assert_eq!(pred.cols, target.cols);

    let epsilon = 1e-15;
    let mut sum = 0.0;
    for i in 0..pred.data.len() {
        sum -= target.data[i] * (pred.data[i] + epsilon).ln();
    }
    sum / pred.rows as f64
}

/// Gradient of cross-entropy loss combined with softmax
/// When softmax output is used with cross-entropy, the gradient simplifies to:
/// dL/dz = softmax(z) - target = pred - target
pub fn cross_entropy_softmax_grad(pred: &Mat, target: &Mat) -> Mat {
    pred.sub(target).scale(1.0 / pred.rows as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse() {
        let pred = Mat::from_slice(&[&[1.0], &[2.0]]);
        let target = Mat::from_slice(&[&[1.0], &[1.0]]);
        // 0.5 * ((1-1)^2 + (2-1)^2) = 0.5 * 1 = 0.5
        assert!((mse(&pred, &target) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_mse_grad() {
        let pred = Mat::from_slice(&[&[1.0], &[2.0]]);
        let target = Mat::from_slice(&[&[1.0], &[1.0]]);
        let grad = mse_grad(&pred, &target);
        assert!((grad.get(0, 0) - 0.0).abs() < 1e-10);
        assert!((grad.get(1, 0) - 1.0).abs() < 1e-10);
    }
}
