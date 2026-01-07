use ann_pso::{
    load_loss_history, load_accuracy_history, load_confusion_matrix,
    plot_loss_curve, plot_accuracy_curve, plot_confusion_matrix,
};
use std::env;
use std::path::Path;

fn get_train_dir(optimizer: &str) -> String {
    match optimizer {
        "pso" => "blob/mnist/train/pso".to_string(),
        _ => format!("blob/mnist/train/gradient-descent/{}", optimizer),
    }
}

fn get_predict_dir(optimizer: &str) -> String {
    match optimizer {
        "pso" => "blob/mnist/predict/pso".to_string(),
        _ => format!("blob/mnist/predict/gradient-descent/{}", optimizer),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let optimizer = args.get(1).map(|s| s.as_str()).unwrap_or("sgd");
    let plot_type = args.get(2).map(|s| s.as_str());

    println!("=== MNIST: Generate Plots for {} ===\n", optimizer.to_uppercase());

    let train_dir = get_train_dir(optimizer);
    let predict_dir = get_predict_dir(optimizer);

    match plot_type {
        Some("loss") => plot_loss_only(&train_dir, optimizer),
        Some("accuracy") => plot_accuracy_only(&train_dir, optimizer),
        Some("confusion") => plot_confusion_only(&predict_dir, optimizer),
        None => {
            // Generate all plots
            plot_loss_only(&train_dir, optimizer);
            plot_accuracy_only(&train_dir, optimizer);
            plot_confusion_only(&predict_dir, optimizer);
        }
        Some(t) => {
            eprintln!("Unknown plot type: {}", t);
            eprintln!("Available types: loss, accuracy, confusion");
            std::process::exit(1);
        }
    }

    println!("\nDone!");
}

fn plot_loss_only(train_dir: &str, optimizer: &str) {
    let csv_path = format!("{}/loss.csv", train_dir);
    let png_path = format!("{}/loss.png", train_dir);

    if !Path::new(&csv_path).exists() {
        eprintln!("Loss data not found: {}", csv_path);
        eprintln!("Run training first.");
        return;
    }

    let losses = load_loss_history(&csv_path)
        .expect("Failed to load loss history");

    let title = format!("MNIST {} Loss Curve", optimizer.to_uppercase());
    plot_loss_curve(&losses, &png_path, &title)
        .expect("Failed to plot loss curve");

    println!("Loss curve saved to: {}", png_path);
}

fn plot_accuracy_only(train_dir: &str, optimizer: &str) {
    let csv_path = format!("{}/accuracy.csv", train_dir);
    let png_path = format!("{}/accuracy.png", train_dir);

    if !Path::new(&csv_path).exists() {
        eprintln!("Accuracy data not found: {}", csv_path);
        eprintln!("Run training first.");
        return;
    }

    let (train_acc, test_acc) = load_accuracy_history(&csv_path)
        .expect("Failed to load accuracy history");

    let title = format!("MNIST {} Accuracy Curve", optimizer.to_uppercase());
    plot_accuracy_curve(&train_acc, &test_acc, &png_path, &title)
        .expect("Failed to plot accuracy curve");

    println!("Accuracy curve saved to: {}", png_path);
}

fn plot_confusion_only(predict_dir: &str, optimizer: &str) {
    let csv_path = format!("{}/confusion_matrix.csv", predict_dir);
    let png_path = format!("{}/confusion_matrix.png", predict_dir);

    if !Path::new(&csv_path).exists() {
        eprintln!("Confusion matrix data not found: {}", csv_path);
        eprintln!("Run prediction first.");
        return;
    }

    let matrix = load_confusion_matrix(&csv_path)
        .expect("Failed to load confusion matrix");

    let title = format!("MNIST {} Confusion Matrix", optimizer.to_uppercase());
    plot_confusion_matrix(&matrix, &png_path, &title)
        .expect("Failed to plot confusion matrix");

    println!("Confusion matrix saved to: {}", png_path);
}
