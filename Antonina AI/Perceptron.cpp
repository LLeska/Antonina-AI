#include "Perceptron.h"
#include "Layer.h"
#include <random>
#include <chrono>
#include <fstream>
#include <string>
#include <iostream>


double Perceptron::rand_double(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);
    return dis(gen);
}

double Perceptron::activation(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double Perceptron::dactivation(double y) {
    return y * (1.0 - y);
}

double Perceptron::solve_z(int L, int j) {
    Layer& prev_layer = layers[L - 1];
    Layer& curr_layer = layers[L];
    double z = 0.0;
    if (prev_layer.weights) {
        for (int k = 0; k < prev_layer.size; k++) {
            z += prev_layer.neurons[k] * prev_layer.weights[j][k];
        }
    }
    if (prev_layer.biases) {
        z += prev_layer.biases[j];
    }
    return z;
}


template <typename T>
void Perceptron::copy_array(int n, T* array_source, T* array_destination) {
    for (int i = 0; i < n; i++) {
        array_destination[i] = array_source[i];
    }
}

Perceptron::Perceptron() {
    learningRate = 0;
    layers_length = 0;
    layers = nullptr;
}


Perceptron::Perceptron(double learningRate_, int layers_length_, int* sizes) {
    learningRate = learningRate_;
    layers_length = layers_length_;
    layers = new Layer[layers_length_];
    for (int l = 0; l < layers_length_; l++) {
        int nextSize = 0;
        if (l < layers_length_ - 1) {
            nextSize = sizes[l + 1];
        }
        layers[l] = Layer(sizes[l], nextSize);
        if (nextSize > 0 && layers[l].biases && layers[l].weights) {
            for (int j = 0; j < nextSize; j++) {
                layers[l].biases[j] = rand_double(-1.0, 1.0);
                for (int k = 0; k < sizes[l]; k++) {
                    layers[l].weights[j][k] = rand_double(-1.0, 1.0);
                }
            }
        }
    }
}


double* Perceptron::feedForward(double* inputs, int n) {
    copy_array(n, inputs, layers[0].neurons);

    for (int L = 1; L < layers_length; L++) {
        for (int j = 0; j < layers[L].size; j++) {
            double z = solve_z(L, j);
            layers[L].neurons[j] = activation(z);
        }
    }
    return layers[layers_length - 1].neurons;
}

void Perceptron::backpropagation(double* targets) {
    int last = layers_length - 1;
    double** deltas = new double* [layers_length];
    for (int i = 0; i < layers_length; i++) {
        deltas[i] = new double[layers[i].size]();
    }
    for (int j = 0; j < layers[last].size; j++) {
        double output = layers[last].neurons[j];
        double error = targets[j] - output;
        deltas[last][j] = error * dactivation(output);
    }
    for (int L = last - 1; L >= 1; L--) {
        Layer& curr = layers[L];
        Layer& next = layers[L + 1];

        for (int k = 0; k < curr.size; k++) {
            double error = 0;
            for (int j = 0; j < next.size; j++) {
                error += layers[L].weights[j][k] * deltas[L + 1][j];
            }
            deltas[L][k] = error * dactivation(curr.neurons[k]);
        }
    }
    for (int L = 0; L < layers_length - 1; L++) {
        Layer& curr = layers[L];
        Layer& next = layers[L + 1];
        for (int j = 0; j < next.size; j++) {
            for (int k = 0; k < curr.size; k++) {
                curr.weights[j][k] += learningRate * deltas[L + 1][j] * curr.neurons[k];
            }
            curr.biases[j] += learningRate * deltas[L + 1][j];
        }
    }
    for (int i = 0; i < layers_length; i++) {
        delete[] deltas[i];
    }
    delete[] deltas;
}

void Perceptron::deinit() {
    if (layers) {
        for (int i = 0; i < layers_length; i++) {
            layers[i].deinit();
        }
        delete[] layers;
        layers = nullptr;
    }
}

void Perceptron::readFromFile(std::string file) {
    std::ifstream fin(file, std::ios::in);
    if (!fin.is_open()) {
        std::cerr << "Ошибка открытия файла" << std::endl;
        return;
    }
    this->deinit();
    fin >>learningRate >> layers_length;
    layers = new Layer[layers_length];
    for (int i = 0; i < layers_length; i++) {
        layers[i].readFromFile(&fin);
    }

    fin.close();
}

void Perceptron::writeInFile(std::string file) {
    std::ofstream fout(file, std::ios::out);
    fout << learningRate << ' ' << layers_length << '\n';
    for (int i = 0; i < layers_length; i++) {
        layers[i].writeInFile(&fout);
    }
    fout.close();
}