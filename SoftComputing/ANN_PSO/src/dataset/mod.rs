mod mnist;
mod xor;

pub use mnist::MnistDataset;
pub use xor::XorDataset;

use crate::Mat;

/// Data split containing input and target matrices
#[derive(Clone)]
pub struct DataSplit {
    pub x: Mat,
    pub y: Mat,
}

impl DataSplit {
    pub fn new(x: Mat, y: Mat) -> Self {
        Self { x, y }
    }

    /// Number of samples in this split
    pub fn len(&self) -> usize {
        self.x.rows
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Dataset trait for train/val/test splits
pub trait Dataset {
    /// Get training data
    fn train_data(&self) -> DataSplit;

    /// Get validation data (optional)
    fn val_data(&self) -> Option<DataSplit> {
        None
    }

    /// Get test data (optional)
    fn test_data(&self) -> Option<DataSplit> {
        None
    }

    /// Dataset name
    fn name(&self) -> &str;

    /// Check if dataset has validation split
    fn has_val(&self) -> bool {
        self.val_data().is_some()
    }

    /// Check if dataset has test split
    fn has_test(&self) -> bool {
        self.test_data().is_some()
    }
}
