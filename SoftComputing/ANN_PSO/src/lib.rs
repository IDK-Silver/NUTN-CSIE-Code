pub mod mat;
pub mod layer;
pub mod loss;
pub mod optimizer;
pub mod utils;
pub mod config;
pub mod network;

pub use mat::Mat;
pub use layer::{Layer, Linear, Sigmoid};
pub use loss::{mse, mse_grad};
pub use optimizer::{Pso, PsoConfig, Sgd};
pub use utils::{save_loss_history, plot_loss_curve};
pub use config::{PsoYamlConfig, SgdYamlConfig, SavedModel, ModelWeights};
pub use network::{XorNetwork, XorCache, xor_data, test_cases};
