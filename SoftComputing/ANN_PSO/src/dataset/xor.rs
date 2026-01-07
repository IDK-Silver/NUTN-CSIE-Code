use crate::Mat;
use super::{Dataset, DataSplit};

/// XOR Dataset
///
/// Training data: 4 XOR truth table samples
/// Test data: Assignment test cases (0.7, 0.3), (0.6, 0.4), (0.5, 0.5)
pub struct XorDataset;

impl XorDataset {
    pub fn new() -> Self {
        Self
    }
}

impl Default for XorDataset {
    fn default() -> Self {
        Self::new()
    }
}

impl Dataset for XorDataset {
    fn train_data(&self) -> DataSplit {
        let x = Mat::from_slice(&[
            &[0.0, 0.0],
            &[0.0, 1.0],
            &[1.0, 0.0],
            &[1.0, 1.0],
        ]);
        let y = Mat::from_slice(&[&[0.0], &[1.0], &[1.0], &[0.0]]);
        DataSplit::new(x, y)
    }

    fn val_data(&self) -> Option<DataSplit> {
        // XOR is too small for validation split
        None
    }

    fn test_data(&self) -> Option<DataSplit> {
        // Test cases from assignment
        let x = Mat::from_slice(&[
            &[0.7, 0.3],
            &[0.6, 0.4],
            &[0.5, 0.5],
        ]);
        // No ground truth for test cases
        let y = Mat::from_slice(&[&[0.0], &[0.0], &[0.0]]);
        Some(DataSplit::new(x, y))
    }

    fn name(&self) -> &str {
        "xor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_dataset_train() {
        let dataset = XorDataset::new();
        let train = dataset.train_data();
        assert_eq!(train.len(), 4);
    }

    #[test]
    fn test_xor_dataset_test() {
        let dataset = XorDataset::new();
        let test = dataset.test_data().unwrap();
        assert_eq!(test.len(), 3);
    }

    #[test]
    fn test_xor_dataset_no_val() {
        let dataset = XorDataset::new();
        assert!(!dataset.has_val());
    }
}
