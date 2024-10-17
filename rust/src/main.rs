extern crate ndarray;
extern crate rand;

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use std::fs::File;
use std::io::{self, BufRead, BufReader};

struct NeuralNetwork {
    weights_input_hidden: Array2<f64>,
    biases_hidden: Array1<f64>,
    weights_hidden_output: Array2<f64>,
    biases_output: Array1<f64>,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        
        let weights_input_hidden = Array2::from_shape_fn((hidden_size, input_size), 
            |_| rng.gen_range(-1.0..1.0));
        let biases_hidden = Array1::from_shape_fn(hidden_size,
            |_| rng.gen_range(-1.0..1.0));
        let weights_hidden_output = Array2::from_shape_fn((output_size, hidden_size),
            |_| rng.gen_range(-1.0..1.0));
        let biases_output = Array1::from_shape_fn(output_size,
            |_| rng.gen_range(-1.0..1.0));

        NeuralNetwork {
            weights_input_hidden,
            biases_hidden,
            weights_hidden_output,
            biases_output,
        }
    }

    fn sigmoid(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|val| 1.0 / (1.0 + (-val).exp()))
    }

    fn sigmoid_derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        x * (1.0 - x)
    }

    fn forward(&self, input: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        // Hidden layer
        let hidden_input = input.dot(&self.weights_input_hidden.t()) + &self.biases_hidden;
        let hidden_activated = self.sigmoid(&hidden_input);

        // Output layer
        let output_input = hidden_activated.dot(&self.weights_hidden_output.t()) + &self.biases_output;
        let output_activated = self.sigmoid(&output_input);

        (hidden_activated, output_activated)
    }

    fn train(&mut self, input: &Array1<f64>, target: &Array1<f64>, learning_rate: f64) {
        // Forward pass
        let (hidden, output) = self.forward(input);

        // Output layer error
        let output_error = output.clone() - target;
        let d_output = &output_error * self.sigmoid_derivative(&output);

        // Hidden layer error
        let hidden_error = d_output.dot(&self.weights_hidden_output);
        let d_hidden = &hidden_error * self.sigmoid_derivative(&hidden);

        // Create the adjustment matrices
        let d_output_reshaped = d_output.clone().insert_axis(Axis(1));
        let hidden_reshaped = hidden.clone().insert_axis(Axis(0));
        let hidden_output_adjustment = &d_output_reshaped.dot(&hidden_reshaped);

        let d_hidden_reshaped = d_hidden.clone().insert_axis(Axis(1));
        let input_reshaped = input.clone().insert_axis(Axis(0));
        let input_hidden_adjustment = &d_hidden_reshaped.dot(&input_reshaped);

        // Update weights and biases
        self.weights_hidden_output = &self.weights_hidden_output - 
            &(learning_rate * hidden_output_adjustment);
        self.biases_output = &self.biases_output - 
            &(learning_rate * &d_output);
        
        self.weights_input_hidden = &self.weights_input_hidden - 
            &(learning_rate * input_hidden_adjustment);
        self.biases_hidden = &self.biases_hidden - 
            &(learning_rate * &d_hidden);
    }

    fn predict(&self, input: &Array1<f64>) -> Array1<f64> {
        let (_, output) = self.forward(input);
        output
    }
}

fn get_learning_rate(epoch: usize) -> f64 {
    if epoch < 200 {
        0.01
    } else if epoch < 500 {
        0.005
    } else {
        0.001
    }
}

fn normalize_minmax(data: &[f64]) -> (Vec<f64>, f64, f64) {
    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let normalized = data.iter()
        .map(|&x| 0.8 * ((x - min_val) / (max_val - min_val)) + 0.1)
        .collect();
    (normalized, min_val, max_val)
}

fn denormalize_minmax(value: f64, min_val: f64, max_val: f64) -> f64 {
    ((value - 0.1) / 0.8) * (max_val - min_val) + min_val
}

fn read_data_from_file(filename: &str) -> io::Result<Vec<f64>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut data = Vec::new();

    for line in reader.lines() {
        let line = line?;
        for value in line.split(',') {
            if let Ok(num) = value.trim().parse::<f64>() {
                data.push(num);
            }
        }
    }
    Ok(data)
}

fn main() -> io::Result<()> {
    // Read data from file
    let sales_data = match read_data_from_file("inputfile.txt") {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error reading input file: {}", e);
            return Err(e);
        }
    };

    if sales_data.len() < 12 {
        eprintln!("Insufficient data for training");
        return Ok(());
    }

    let input_size = 12;
    let hidden_size = (1.0 / 3.0 * input_size as f64 + 1.0) as usize;
    let output_size = 1;

    let mut nn = NeuralNetwork::new(input_size, hidden_size, output_size);

    // Normalize data using min-max normalization
    let (normalized_data, min_val, max_val) = normalize_minmax(&sales_data);

    const NUM_EPOCHS: usize = 1000;

    // Training
    for i in 0..=normalized_data.len() - input_size - 1 {
        let input_slice = &normalized_data[i..i + input_size];
        let input = Array1::from_vec(input_slice.to_vec());
        let target = Array1::from_vec(vec![normalized_data[i + input_size]]);

        for epoch in 0..NUM_EPOCHS {
            let learning_rate = get_learning_rate(epoch);
            nn.train(&input, &target, learning_rate);
        }
    }

    // Prediction
    let months_to_predict = 6;
    let mut current_input = Array1::from_vec(
        normalized_data[normalized_data.len() - input_size..].to_vec()
    );

    println!("\nPredictions for the next {} months:", months_to_predict);
    for m in 0..months_to_predict {
        let output = nn.predict(&current_input);
        let denormalized_prediction = denormalize_minmax(output[0], min_val, max_val);
        println!("Month {}: {:.2}", m + 1, denormalized_prediction);

        // Update input window for next prediction
        for i in 0..input_size - 1 {
            current_input[i] = current_input[i + 1];
        }
        current_input[input_size - 1] = output[0];
    }

    Ok(())
}
