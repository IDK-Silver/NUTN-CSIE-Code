use crate::dataset::{Dataset, DataSplit};
use crate::Mat;
use byteorder::{BigEndian, ReadBytesExt};
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::Read;

const DATA_DIR: &str = "blob/mnist/data";

/// MNIST handwritten digit dataset
/// 60,000 training images + 10,000 test images
/// Each image is 28x28 grayscale, flattened to 784-dim vector
pub struct MnistDataset {
    train_images: Mat,  // (60000, 784)
    train_labels: Mat,  // (60000, 10) one-hot
    test_images: Mat,   // (10000, 784)
    test_labels: Mat,   // (10000, 10) one-hot
}

impl MnistDataset {
    /// Load MNIST dataset from data/mnist/ directory
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let train_images = Self::load_images(&format!("{}/train-images-idx3-ubyte.gz", DATA_DIR))?;
        let train_labels = Self::load_labels(&format!("{}/train-labels-idx1-ubyte.gz", DATA_DIR))?;
        let test_images = Self::load_images(&format!("{}/t10k-images-idx3-ubyte.gz", DATA_DIR))?;
        let test_labels = Self::load_labels(&format!("{}/t10k-labels-idx1-ubyte.gz", DATA_DIR))?;

        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }

    /// Parse IDX image file
    /// Format: [magic(4)] [num_images(4)] [rows(4)] [cols(4)] [pixels...]
    fn load_images(path: &str) -> Result<Mat, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = GzDecoder::new(file);

        // Read header
        let magic = reader.read_u32::<BigEndian>()?;
        if magic != 2051 {
            return Err(format!("Invalid magic number for images: {}", magic).into());
        }

        let num_images = reader.read_u32::<BigEndian>()? as usize;
        let rows = reader.read_u32::<BigEndian>()? as usize;
        let cols = reader.read_u32::<BigEndian>()? as usize;

        // Read pixel data
        let pixels_per_image = rows * cols;
        let mut data = vec![0u8; num_images * pixels_per_image];
        reader.read_exact(&mut data)?;

        // Convert to f64 and normalize to [0, 1]
        let normalized: Vec<f64> = data.iter().map(|&p| p as f64 / 255.0).collect();

        Ok(Mat::new(num_images, pixels_per_image, normalized))
    }

    /// Parse IDX label file
    /// Format: [magic(4)] [num_labels(4)] [labels...]
    fn load_labels(path: &str) -> Result<Mat, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = GzDecoder::new(file);

        // Read header
        let magic = reader.read_u32::<BigEndian>()?;
        if magic != 2049 {
            return Err(format!("Invalid magic number for labels: {}", magic).into());
        }

        let num_labels = reader.read_u32::<BigEndian>()? as usize;

        // Read labels
        let mut labels = vec![0u8; num_labels];
        reader.read_exact(&mut labels)?;

        // One-hot encode (10 classes)
        Ok(Self::one_hot(&labels, 10))
    }

    /// One-hot encode labels
    fn one_hot(labels: &[u8], num_classes: usize) -> Mat {
        let num_samples = labels.len();
        let mut data = vec![0.0; num_samples * num_classes];

        for (i, &label) in labels.iter().enumerate() {
            data[i * num_classes + label as usize] = 1.0;
        }

        Mat::new(num_samples, num_classes, data)
    }

    /// Get a subset of training data (for mini-batch or PSO fitness evaluation)
    pub fn train_subset(&self, start: usize, count: usize) -> DataSplit {
        let end = (start + count).min(self.train_images.rows);
        DataSplit::new(
            self.train_images.slice_rows(start, end),
            self.train_labels.slice_rows(start, end),
        )
    }

    /// Number of training samples
    pub fn train_len(&self) -> usize {
        self.train_images.rows
    }

    /// Number of test samples
    pub fn test_len(&self) -> usize {
        self.test_images.rows
    }
}

impl Dataset for MnistDataset {
    fn train_data(&self) -> DataSplit {
        DataSplit::new(self.train_images.clone(), self.train_labels.clone())
    }

    fn val_data(&self) -> Option<DataSplit> {
        // Could split off last 10k training samples as validation
        // For simplicity, we return None
        None
    }

    fn test_data(&self) -> Option<DataSplit> {
        Some(DataSplit::new(
            self.test_images.clone(),
            self.test_labels.clone(),
        ))
    }

    fn name(&self) -> &str {
        "mnist"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_hot() {
        let labels: Vec<u8> = vec![0, 1, 9, 5];
        let one_hot = MnistDataset::one_hot(&labels, 10);

        assert_eq!(one_hot.rows, 4);
        assert_eq!(one_hot.cols, 10);

        // Check first sample (label 0)
        assert_eq!(one_hot.get(0, 0), 1.0);
        assert_eq!(one_hot.get(0, 1), 0.0);

        // Check second sample (label 1)
        assert_eq!(one_hot.get(1, 0), 0.0);
        assert_eq!(one_hot.get(1, 1), 1.0);

        // Check third sample (label 9)
        assert_eq!(one_hot.get(2, 9), 1.0);

        // Check fourth sample (label 5)
        assert_eq!(one_hot.get(3, 5), 1.0);
    }
}
