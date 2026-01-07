use serde::Deserialize;
use std::fs;
use std::path::Path;

/// PSO configuration loaded from YAML
#[derive(Debug, Deserialize)]
pub struct PsoYamlConfig {
    pub num_particles: usize,
    pub w: f64,
    pub c1: f64,
    pub c2: f64,
    pub pos_min: f64,
    pub pos_max: f64,
    pub vel_max: f64,
    pub max_iter: usize,
    pub target_loss: f64,
}

impl PsoYamlConfig {
    /// Load PSO config from YAML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: PsoYamlConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Load default PSO config from config/pso/default.yaml
    pub fn load_default() -> Result<Self, Box<dyn std::error::Error>> {
        Self::load("config/pso/default.yaml")
    }
}

/// SGD configuration loaded from YAML
#[derive(Debug, Deserialize)]
pub struct SgdYamlConfig {
    pub lr: f64,
    pub max_iter: usize,
    pub target_loss: f64,
}

impl SgdYamlConfig {
    /// Load SGD config from YAML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: SgdYamlConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Load default SGD config from config/gradient-descent/sgd/default.yaml
    pub fn load_default() -> Result<Self, Box<dyn std::error::Error>> {
        Self::load("config/gradient-descent/sgd/default.yaml")
    }
}

/// Model saved after training
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SavedModel {
    pub architecture: String,
    pub optimizer: String,
    pub final_loss: f64,
    pub iterations: usize,
    pub weights: ModelWeights,
}

/// Neural network weights
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ModelWeights {
    pub linear1_weight: Vec<f64>,
    pub linear1_bias: Vec<f64>,
    pub linear2_weight: Vec<f64>,
    pub linear2_bias: Vec<f64>,
}

impl SavedModel {
    /// Save model to JSON file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let parent = path.as_ref().parent();
        if let Some(p) = parent {
            if !p.as_os_str().is_empty() {
                fs::create_dir_all(p)?;
            }
        }
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Load model from JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let model: SavedModel = serde_json::from_str(&content)?;
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pso_config_load() {
        let config = PsoYamlConfig::load_default().unwrap();
        assert_eq!(config.num_particles, 100);
        assert!((config.w - 0.729).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_config_load() {
        let config = SgdYamlConfig::load_default().unwrap();
        assert!((config.lr - 0.5).abs() < 1e-6);
    }
}
