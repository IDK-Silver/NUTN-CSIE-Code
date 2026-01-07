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

/// Plot accuracy curve (train and test accuracy over epochs)
pub fn plot_accuracy_curve(
    train_acc: &[f64],
    test_acc: &[f64],
    path: &str,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create parent directory if it doesn't exist
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let num_epochs = train_acc.len().max(test_acc.len());
    let num_epochs_f64 = num_epochs as f64;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.5..(num_epochs_f64 + 0.5), 0.0..105.0)?;

    chart
        .configure_mesh()
        .x_labels(num_epochs.min(10))
        .x_label_formatter(&|x| format!("{}", *x as i32))
        .x_desc("Epoch")
        .y_desc("Accuracy (%)")
        .draw()?;

    // Draw train accuracy (blue)
    chart.draw_series(LineSeries::new(
        train_acc.iter().enumerate().map(|(i, &acc)| ((i + 1) as f64, acc)),
        &BLUE,
    ))?.label("Train").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Draw test accuracy (red)
    chart.draw_series(LineSeries::new(
        test_acc.iter().enumerate().map(|(i, &acc)| ((i + 1) as f64, acc)),
        &RED,
    ))?.label("Test").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Draw legend
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::LowerRight)
        .draw()?;

    root.present()?;
    Ok(())
}

/// Plot confusion matrix as heatmap (10x10 for MNIST digits)
pub fn plot_confusion_matrix(
    matrix: &[[usize; 10]; 10],
    path: &str,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create parent directory if it doesn't exist
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let root = BitMapBackend::new(path, (700, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find max value for color scaling
    let max_val = matrix.iter()
        .flat_map(|row| row.iter())
        .cloned()
        .max()
        .unwrap_or(1) as f64;

    // Create chart area with margins for labels
    let chart_area = root.margin(60, 40, 60, 40);

    // Draw title
    root.draw(&Text::new(
        title,
        (350, 20),
        ("sans-serif", 24).into_font().color(&BLACK),
    ))?;

    // Draw axis labels
    root.draw(&Text::new(
        "Predicted",
        (350, 660),
        ("sans-serif", 18).into_font().color(&BLACK),
    ))?;

    // Draw "True" label vertically (approximate with horizontal text)
    root.draw(&Text::new(
        "True",
        (15, 350),
        ("sans-serif", 18).into_font().color(&BLACK),
    ))?;

    let cell_size = 55;
    let offset_x = 50;
    let offset_y = 50;

    // Draw cells
    for (i, row) in matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            let x = offset_x + j * cell_size;
            let y = offset_y + i * cell_size;

            // Color intensity based on value
            let intensity = if max_val > 0.0 { val as f64 / max_val } else { 0.0 };
            let color = RGBColor(
                (255.0 * (1.0 - intensity * 0.7)) as u8,
                (255.0 * (1.0 - intensity * 0.7)) as u8,
                255,
            );

            // Draw cell background
            chart_area.draw(&Rectangle::new(
                [(x as i32, y as i32), ((x + cell_size) as i32, (y + cell_size) as i32)],
                color.filled(),
            ))?;

            // Draw cell border
            chart_area.draw(&Rectangle::new(
                [(x as i32, y as i32), ((x + cell_size) as i32, (y + cell_size) as i32)],
                BLACK.stroke_width(1),
            ))?;

            // Draw value text
            let text_color = if intensity > 0.5 { &WHITE } else { &BLACK };
            chart_area.draw(&Text::new(
                format!("{}", val),
                ((x + cell_size / 2 - 10) as i32, (y + cell_size / 2 - 8) as i32),
                ("sans-serif", 14).into_font().color(text_color),
            ))?;
        }
    }

    // Draw column labels (0-9) at top
    for j in 0..10 {
        let x = offset_x + j * cell_size + cell_size / 2 - 5;
        chart_area.draw(&Text::new(
            format!("{}", j),
            (x as i32, (offset_y - 15) as i32),
            ("sans-serif", 16).into_font().color(&BLACK),
        ))?;
    }

    // Draw row labels (0-9) at left
    for i in 0..10 {
        let y = offset_y + i * cell_size + cell_size / 2 - 8;
        chart_area.draw(&Text::new(
            format!("{}", i),
            ((offset_x - 25) as i32, y as i32),
            ("sans-serif", 16).into_font().color(&BLACK),
        ))?;
    }

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
