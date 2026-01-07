use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
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

/// Save accuracy history to a CSV file
///
/// # CSV Format
/// ```csv
/// epoch,train_acc,test_acc
/// 1,45.23,42.15
/// 2,78.56,75.32
/// ```
pub fn save_accuracy_history(train_acc: &[f64], test_acc: &[f64], path: &str) -> io::Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut file = File::create(path)?;
    writeln!(file, "epoch,train_acc,test_acc")?;

    for (i, (train, test)) in train_acc.iter().zip(test_acc.iter()).enumerate() {
        writeln!(file, "{},{:.4},{:.4}", i + 1, train, test)?;
    }

    Ok(())
}

/// Save confusion matrix to a CSV file
///
/// # CSV Format
/// First row is header (0-9), followed by 10 rows of data
pub fn save_confusion_matrix(matrix: &[[usize; 10]; 10], path: &str) -> io::Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut file = File::create(path)?;

    // Write header
    writeln!(file, "0,1,2,3,4,5,6,7,8,9")?;

    // Write data rows
    for row in matrix.iter() {
        let row_str: Vec<String> = row.iter().map(|v| v.to_string()).collect();
        writeln!(file, "{}", row_str.join(","))?;
    }

    Ok(())
}

/// Load loss history from a CSV file
pub fn load_loss_history(path: &str) -> io::Result<Vec<f64>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut losses = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        if i == 0 { continue; } // Skip header
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            if let Ok(loss) = parts[1].parse::<f64>() {
                losses.push(loss);
            }
        }
    }

    Ok(losses)
}

/// Load accuracy history from a CSV file
pub fn load_accuracy_history(path: &str) -> io::Result<(Vec<f64>, Vec<f64>)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut train_acc = Vec::new();
    let mut test_acc = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        if i == 0 { continue; } // Skip header
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 3 {
            if let (Ok(train), Ok(test)) = (parts[1].parse::<f64>(), parts[2].parse::<f64>()) {
                train_acc.push(train);
                test_acc.push(test);
            }
        }
    }

    Ok((train_acc, test_acc))
}

/// Load confusion matrix from a CSV file
pub fn load_confusion_matrix(path: &str) -> io::Result<[[usize; 10]; 10]> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut matrix = [[0usize; 10]; 10];

    for (i, line) in reader.lines().enumerate() {
        if i == 0 { continue; } // Skip header
        let row_idx = i - 1;
        if row_idx >= 10 { break; }

        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();
        for (j, part) in parts.iter().enumerate().take(10) {
            if let Ok(val) = part.parse::<usize>() {
                matrix[row_idx][j] = val;
            }
        }
    }

    Ok(matrix)
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

    // Create drawing area (high resolution)
    let root = BitMapBackend::new(path, (1600, 1200)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find max loss for y-axis range
    let max_loss = losses.iter().cloned().fold(0.0_f64, f64::max);
    let num_iterations = losses.len();

    // Create chart (iteration starts from 1)
    // Use f64 range with 0.5 offset to avoid extra tick mark
    let num_iter_f64 = num_iterations as f64;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 50))
        .margin(40)
        .x_label_area_size(80)
        .y_label_area_size(100)
        .build_cartesian_2d(0.5..(num_iter_f64 + 0.5), 0.0..max_loss * 1.1)?;

    // Configure mesh (grid)
    chart
        .configure_mesh()
        .x_labels(num_iterations.min(10))
        .x_label_formatter(&|x| format!("{}", *x as i32)) // Show as integers
        .x_desc("Iteration")
        .y_desc("Loss")
        .label_style(("sans-serif", 28))
        .axis_desc_style(("sans-serif", 32))
        .draw()?;

    // Draw line series (iteration starts from 1)
    chart.draw_series(LineSeries::new(
        losses.iter().enumerate().map(|(i, &loss)| ((i + 1) as f64, loss)),
        BLUE.stroke_width(3),
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

    // High resolution
    let root = BitMapBackend::new(path, (1600, 1200)).into_drawing_area();
    root.fill(&WHITE)?;

    let num_epochs = train_acc.len().max(test_acc.len());
    let num_epochs_f64 = num_epochs as f64;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 50))
        .margin(40)
        .x_label_area_size(80)
        .y_label_area_size(100)
        .build_cartesian_2d(0.5..(num_epochs_f64 + 0.5), 0.0..105.0)?;

    chart
        .configure_mesh()
        .x_labels(num_epochs.min(10))
        .x_label_formatter(&|x| format!("{}", *x as i32))
        .x_desc("Epoch")
        .y_desc("Accuracy (%)")
        .label_style(("sans-serif", 28))
        .axis_desc_style(("sans-serif", 32))
        .draw()?;

    // Draw train accuracy (blue)
    chart.draw_series(LineSeries::new(
        train_acc.iter().enumerate().map(|(i, &acc)| ((i + 1) as f64, acc)),
        BLUE.stroke_width(3),
    ))?.label("Train").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 60, y)], BLUE.stroke_width(3)));

    // Draw test accuracy (red)
    chart.draw_series(LineSeries::new(
        test_acc.iter().enumerate().map(|(i, &acc)| ((i + 1) as f64, acc)),
        RED.stroke_width(3),
    ))?.label("Test").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 60, y)], RED.stroke_width(3)));

    // Draw legend
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::LowerRight)
        .legend_area_size(80)
        .label_font(("sans-serif", 28))
        .draw()?;

    root.present()?;
    Ok(())
}

/// Plot confusion matrix as heatmap (10x10 for MNIST digits)
/// Shows row-normalized percentages (Recall distribution)
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

    // High resolution
    let width: i32 = 1400;
    let height: i32 = 1400;
    let root = BitMapBackend::new(path, (width as u32, height as u32)).into_drawing_area();
    root.fill(&WHITE)?;

    // Calculate row sums for percentage calculation
    let row_sums: Vec<usize> = matrix.iter()
        .map(|row| row.iter().sum())
        .collect();

    // Create chart area with margins for labels
    let chart_area = root.margin(120, 80, 120, 80);

    // Draw title (centered)
    let title_x = width / 2;
    root.draw(&Text::new(
        title,
        (title_x, 40),
        ("sans-serif", 48).into_font().color(&BLACK).pos(plotters::style::text_anchor::Pos::new(
            plotters::style::text_anchor::HPos::Center,
            plotters::style::text_anchor::VPos::Top,
        )),
    ))?;

    // Draw "Predicted" label (centered, with more space from grid)
    root.draw(&Text::new(
        "Predicted",
        (width / 2, height - 30),
        ("sans-serif", 36).into_font().color(&BLACK).pos(plotters::style::text_anchor::Pos::new(
            plotters::style::text_anchor::HPos::Center,
            plotters::style::text_anchor::VPos::Bottom,
        )),
    ))?;

    // Draw "True" label
    root.draw(&Text::new(
        "True",
        (30, height / 2),
        ("sans-serif", 36).into_font().color(&BLACK),
    ))?;

    let cell_size: usize = 100;
    let offset_x: usize = 100;
    let offset_y: usize = 100;

    // Draw cells with row-normalized percentages
    for (i, row) in matrix.iter().enumerate() {
        let row_sum = row_sums[i];
        for (j, &val) in row.iter().enumerate() {
            let x = offset_x + j * cell_size;
            let y = offset_y + i * cell_size;

            // Calculate percentage (row-normalized)
            let percent = if row_sum > 0 {
                100.0 * val as f64 / row_sum as f64
            } else {
                0.0
            };

            // Color intensity based on percentage (100% = darkest)
            let intensity = percent / 100.0;
            let color = RGBColor(
                (255.0 * (1.0 - intensity * 0.8)) as u8,
                (255.0 * (1.0 - intensity * 0.8)) as u8,
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
                BLACK.stroke_width(2),
            ))?;

            // Draw percentage text
            let text_color = if intensity > 0.5 { &WHITE } else { &BLACK };
            let text = if percent >= 10.0 {
                format!("{:.1}%", percent)
            } else if percent >= 1.0 {
                format!("{:.1}%", percent)
            } else if percent > 0.0 {
                format!("{:.2}%", percent)
            } else {
                "0%".to_string()
            };

            // Center text in cell
            let text_x = (x + cell_size / 2) as i32;
            let text_y = (y + cell_size / 2) as i32;
            chart_area.draw(&Text::new(
                text,
                (text_x, text_y),
                ("sans-serif", 22).into_font().color(text_color).pos(plotters::style::text_anchor::Pos::new(
                    plotters::style::text_anchor::HPos::Center,
                    plotters::style::text_anchor::VPos::Center,
                )),
            ))?;
        }
    }

    // Draw column labels (0-9) at top
    for j in 0..10 {
        let x = (offset_x + j * cell_size + cell_size / 2) as i32;
        chart_area.draw(&Text::new(
            format!("{}", j),
            (x, (offset_y - 25) as i32),
            ("sans-serif", 32).into_font().color(&BLACK).pos(plotters::style::text_anchor::Pos::new(
                plotters::style::text_anchor::HPos::Center,
                plotters::style::text_anchor::VPos::Bottom,
            )),
        ))?;
    }

    // Draw row labels (0-9) at left
    for i in 0..10 {
        let y = (offset_y + i * cell_size + cell_size / 2) as i32;
        chart_area.draw(&Text::new(
            format!("{}", i),
            ((offset_x - 35) as i32, y),
            ("sans-serif", 32).into_font().color(&BLACK).pos(plotters::style::text_anchor::Pos::new(
                plotters::style::text_anchor::HPos::Right,
                plotters::style::text_anchor::VPos::Center,
            )),
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
