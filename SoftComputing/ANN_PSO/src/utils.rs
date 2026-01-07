use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use plotters::prelude::*;

/// Save loss history to a CSV file
///
/// # Arguments
/// * `losses` - Vector of loss values (one per iteration)
/// * `path` - Output file path (e.g., "output/pso_loss.csv")
///
/// # CSV Format
/// ```csv
/// iteration,loss
/// 0,1.234567
/// 1,1.123456
/// ...
/// ```
pub fn save_loss_history(losses: &[f64], path: &str) -> io::Result<()> {
    // Create parent directory if it doesn't exist
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut file = File::create(path)?;

    // Write header
    writeln!(file, "iteration,loss")?;

    // Write data (iteration starts from 1)
    for (i, loss) in losses.iter().enumerate() {
        writeln!(file, "{},{}", i + 1, loss)?;
    }

    Ok(())
}

/// Plot loss curve and save as PNG
///
/// # Arguments
/// * `losses` - Vector of loss values (one per iteration)
/// * `path` - Output file path (e.g., "output/pso_loss.png")
/// * `title` - Chart title (e.g., "PSO Loss Curve")
pub fn plot_loss_curve(
    losses: &[f64],
    path: &str,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create parent directory if it doesn't exist
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    // Create drawing area
    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find max loss for y-axis range
    let max_loss = losses.iter().cloned().fold(0.0_f64, f64::max);
    let num_iterations = losses.len();

    // Create chart (iteration starts from 1)
    // Use f64 range with 0.5 offset to avoid extra tick mark
    let num_iter_f64 = num_iterations as f64;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.5..(num_iter_f64 + 0.5), 0.0..max_loss * 1.1)?;

    // Configure mesh (grid)
    chart
        .configure_mesh()
        .x_labels(num_iterations.min(10))
        .x_label_formatter(&|x| format!("{}", *x as i32)) // Show as integers
        .x_desc("Iteration")
        .y_desc("Loss")
        .draw()?;

    // Draw line series (iteration starts from 1)
    chart.draw_series(LineSeries::new(
        losses.iter().enumerate().map(|(i, &loss)| ((i + 1) as f64, loss)),
        &BLUE,
    ))?;

    // Ensure the chart is saved
    root.present()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_save_loss_history() {
        let losses = vec![1.0, 0.5, 0.25, 0.1];
        let path = "target/test_output/test_loss.csv";

        // Save
        save_loss_history(&losses, path).unwrap();

        // Read and verify
        let content = fs::read_to_string(path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        assert_eq!(lines[0], "iteration,loss");
        assert_eq!(lines[1], "1,1");
        assert_eq!(lines[2], "2,0.5");
        assert_eq!(lines[3], "3,0.25");
        assert_eq!(lines[4], "4,0.1");

        // Cleanup
        fs::remove_file(path).ok();
    }
}
