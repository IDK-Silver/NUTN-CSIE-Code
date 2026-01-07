use rand::Rng;

/// Row-major matrix implementation
#[derive(Debug, Clone)]
pub struct Mat {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl Mat {
    /// Create a matrix with given data
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { data, rows, cols }
    }

    /// Create a matrix filled with zeros
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Create a matrix from 2D slice (convenience method)
    pub fn from_slice(data: &[&[f64]]) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        let mut flat = Vec::with_capacity(rows * cols);
        for row in data {
            assert_eq!(row.len(), cols);
            flat.extend_from_slice(row);
        }
        Self {
            data: flat,
            rows,
            cols,
        }
    }

    /// Create a random matrix with values in [min, max]
    pub fn random(rows: usize, cols: usize, min: f64, max: f64) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..rows * cols)
            .map(|_| rng.gen_range(min..=max))
            .collect();
        Self { data, rows, cols }
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    /// Set element at (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.cols + col] = value;
    }

    /// Get a slice of a row
    pub fn row(&self, row: usize) -> &[f64] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Matrix multiplication: self(m×k) × other(k×n) → result(m×n)
    pub fn matmul(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.rows, "Matrix dimensions mismatch for matmul");
        let mut result = Mat::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    /// Transpose: self(m×n) → result(n×m)
    pub fn transpose(&self) -> Mat {
        let mut result = Mat::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    /// Element-wise addition (supports broadcasting for bias)
    /// If other is (1, cols), broadcast to all rows
    pub fn add(&self, other: &Mat) -> Mat {
        if self.rows == other.rows && self.cols == other.cols {
            // Same shape: element-wise
            let data: Vec<f64> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect();
            Mat::new(self.rows, self.cols, data)
        } else if other.rows == 1 && other.cols == self.cols {
            // Broadcasting (1, cols) to (rows, cols)
            let mut result = self.clone();
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let val = result.get(i, j) + other.get(0, j);
                    result.set(i, j, val);
                }
            }
            result
        } else {
            panic!(
                "Cannot add matrices with shapes ({}, {}) and ({}, {})",
                self.rows, self.cols, other.rows, other.cols
            );
        }
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Mat) -> Mat {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Mat::new(self.rows, self.cols, data)
    }

    /// Element-wise multiplication (Hadamard product)
    pub fn mul(&self, other: &Mat) -> Mat {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Mat::new(self.rows, self.cols, data)
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f64) -> Mat {
        let data: Vec<f64> = self.data.iter().map(|x| x * scalar).collect();
        Mat::new(self.rows, self.cols, data)
    }

    /// Sum along axis
    /// axis=0: sum over rows → (1, cols)
    /// axis=1: sum over cols → (rows, 1)
    pub fn sum_axis(&self, axis: usize) -> Mat {
        match axis {
            0 => {
                // Sum over rows: (rows, cols) → (1, cols)
                let mut result = Mat::zeros(1, self.cols);
                for j in 0..self.cols {
                    let mut sum = 0.0;
                    for i in 0..self.rows {
                        sum += self.get(i, j);
                    }
                    result.set(0, j, sum);
                }
                result
            }
            1 => {
                // Sum over cols: (rows, cols) → (rows, 1)
                let mut result = Mat::zeros(self.rows, 1);
                for i in 0..self.rows {
                    let mut sum = 0.0;
                    for j in 0..self.cols {
                        sum += self.get(i, j);
                    }
                    result.set(i, 0, sum);
                }
                result
            }
            _ => panic!("Invalid axis: {}", axis),
        }
    }

    /// Apply a function element-wise
    pub fn map<F>(&self, f: F) -> Mat
    where
        F: Fn(f64) -> f64,
    {
        let data: Vec<f64> = self.data.iter().map(|&x| f(x)).collect();
        Mat::new(self.rows, self.cols, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul() {
        let a = Mat::from_slice(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let b = Mat::from_slice(&[&[7.0, 8.0], &[9.0, 10.0], &[11.0, 12.0]]);
        let c = a.matmul(&b);
        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 2);
        assert_eq!(c.get(0, 0), 58.0);
        assert_eq!(c.get(0, 1), 64.0);
        assert_eq!(c.get(1, 0), 139.0);
        assert_eq!(c.get(1, 1), 154.0);
    }

    #[test]
    fn test_transpose() {
        let a = Mat::from_slice(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let b = a.transpose();
        assert_eq!(b.rows, 3);
        assert_eq!(b.cols, 2);
        assert_eq!(b.get(0, 0), 1.0);
        assert_eq!(b.get(2, 1), 6.0);
    }

    #[test]
    fn test_broadcast_add() {
        let a = Mat::from_slice(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0], &[7.0, 8.0]]);
        let b = Mat::from_slice(&[&[10.0, 20.0]]);
        let c = a.add(&b);
        assert_eq!(c.get(0, 0), 11.0);
        assert_eq!(c.get(0, 1), 22.0);
        assert_eq!(c.get(3, 0), 17.0);
        assert_eq!(c.get(3, 1), 28.0);
    }

    #[test]
    fn test_sum_axis() {
        let a = Mat::from_slice(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
        let sum0 = a.sum_axis(0);
        assert_eq!(sum0.rows, 1);
        assert_eq!(sum0.cols, 2);
        assert_eq!(sum0.get(0, 0), 9.0);
        assert_eq!(sum0.get(0, 1), 12.0);

        let sum1 = a.sum_axis(1);
        assert_eq!(sum1.rows, 3);
        assert_eq!(sum1.cols, 1);
        assert_eq!(sum1.get(0, 0), 3.0);
        assert_eq!(sum1.get(1, 0), 7.0);
        assert_eq!(sum1.get(2, 0), 11.0);
    }
}
