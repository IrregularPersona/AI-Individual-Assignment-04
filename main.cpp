#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include <algorithm>

using namespace Eigen;
using namespace std;

class NeuralNetwork {
private:
    MatrixXd weights_input_hidden;
    VectorXd biases_hidden;
    MatrixXd weights_hidden_output;
    VectorXd biases_output;
    
public:
    NeuralNetwork(int inputsize, int hiddensize, int outputsize) {
        weights_input_hidden = MatrixXd::Random(hiddensize, inputsize);
        biases_hidden = VectorXd::Random(hiddensize);
        weights_hidden_output = MatrixXd::Random(outputsize, hiddensize);
        biases_output = VectorXd::Random(outputsize);
    }

    VectorXd forward(const VectorXd &input) {
        // Hidden layer
        VectorXd hidden = weights_input_hidden * input + biases_hidden;
        hidden = (1.0 / (1.0 + (-hidden.array()).exp())).matrix(); 
        
        VectorXd output = weights_hidden_output * hidden + biases_output;
        output = (1.0 / (1.0 + (-output.array()).exp())).matrix(); 
        return output;
    }

    void train(const VectorXd &input, const VectorXd &target, double learningRate) {
        // forward
        VectorXd hidden = weights_input_hidden * input + biases_hidden;
        hidden = (1.0 / (1.0 + (-hidden.array()).exp())).matrix();
        
        VectorXd output = weights_hidden_output * hidden + biases_output;
        output = (1.0 / (1.0 + (-output.array()).exp())).matrix();

        // backprop
        VectorXd output_error = output - target;
        VectorXd d_output = output_error.array() * (output.array() * (1 - output.array()));
        
        VectorXd hidden_error = weights_hidden_output.transpose() * d_output;
        VectorXd d_hidden = hidden_error.array() * (hidden.array() * (1 - hidden.array()));

        // update w & b
        weights_hidden_output -= learningRate * d_output * hidden.transpose();
        biases_output -= learningRate * d_output;
        weights_input_hidden -= learningRate * d_hidden * input.transpose();
        biases_hidden -= learningRate * d_hidden;
    }
};

double getLearningRate(int epoch) {
    // testing different rates & combinations
    if (epoch < 200) return 0.01; 
    else if (epoch < 500) return 0.005; 
    else return 0.001; 
}

int main() {
    ifstream inputfile("inputfile.txt");
    if (!inputfile.is_open()) {
        cerr << "Error opening input file\n";
        return 1;
    }

    vector<double> sales_data;
    string line;

    while (getline(inputfile, line)) {
        istringstream iss(line);
        double value;
        while (iss >> value) {
            if (iss.peek() == ',') iss.ignore();
            sales_data.push_back(value);
        }
    }
    inputfile.close();

    if (sales_data.size() < 12) {
        cerr << "Insufficient data for training.\n";
        return 1;
    }

    int inputsize = 12;
    int hiddensize = static_cast<int>((1.0 / 3.0) * inputsize + 1);
    int outputsize = 1;

    NeuralNetwork net(inputsize, hiddensize, outputsize);

    double minval = *min_element(sales_data.begin(), sales_data.end());
    double maxval = *max_element(sales_data.begin(), sales_data.end());

    vector<double> normalized_data;
    for (double value : sales_data) {
        double normalized_value = 0.8 * ((value - minval) / (maxval - minval)) + 0.1;
        normalized_data.push_back(normalized_value);
    }

    const int numEpochs = 1000;

    for (int i = 0; i <= normalized_data.size() - inputsize - 1; i++) {
        VectorXd input(inputsize);
        for (int j = 0; j < inputsize; j++) {
            input(j) = normalized_data[i + j];
        }

        VectorXd target(outputsize); // Ensure single-element vector
        target(0) = normalized_data[i + inputsize];

        for (int epoch = 0; epoch < numEpochs; ++epoch) {
            double learningRate = getLearningRate(epoch);
            net.train(input, target, learningRate);

            if (epoch % 100 == 0) {
                VectorXd output = net.forward(input);
                double denormalized_prediction = ((output(0) - 0.1) / 0.8) * (maxval - minval) + minval;

                // formatting things
                if (epoch == 0) {
                    cout << "Epoch " << epoch << "   | Learning Rate: " << learningRate
                         << " | Predicted (denormalized): " << denormalized_prediction << endl;
                } else {
                    cout << "Epoch " << epoch << " | Learning Rate: " << learningRate
                         << " | Predicted (denormalized): " << denormalized_prediction << endl;
                }
            }
        }
    }

    return 0;
}

