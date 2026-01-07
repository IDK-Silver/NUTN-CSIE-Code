pub mod mat;
pub mod layer;
pub mod loss;
pub mod optimizer;
pub mod utils;
pub mod config;
pub mod dataset;
pub mod model;

pub use mat::Mat;
pub use layer::{Layer, Linear, Sigmoid, ReLU, Softmax};
pub use loss::{mse, mse_grad, cross_entropy, cross_entropy_softmax_grad};
pub use optimizer::{Pso, PsoConfig, Sgd};
pub use utils::{save_loss_history, plot_loss_curve, plot_accuracy_curve, plot_confusion_matrix};
pub use config::{
    PsoYamlConfig, SgdYamlConfig, SavedModel, ModelWeights,
    MnistPsoYamlConfig, MnistSgdYamlConfig, MnistSavedModel, MnistModelWeights,
};
pub use dataset::{Dataset, DataSplit, XorDataset, MnistDataset};
pub use model::{Model, GradientModel, XorNetwork, XorCache, MnistNetwork, MnistCache};
