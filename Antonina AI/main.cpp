#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <random>
#include <chrono>
using namespace std;



double rand_double(double min, double max) {
    int64_t seed = std::chrono::steady_clock::now().time_since_epoch().count();
    static std::default_random_engine e(seed);
    std::uniform_int_distribution<int> d((int)min * 1000000, (int)max * 1000000);
    double ans = (double)d(e) / 1000000.0;
    return ans;
}


template <typename T>
void copy_array(int n, T* array_source, T* array_destination) {
    for (int i = 0; i < n; i++) {
        array_destination[i] = array_source[i];
    }
}


class Layer {
public:
    int size = 0;
    int nextSize = 0;
    double* neurons;
    double* biases;
    double** weights;


    Layer() {
    }


    Layer(int _size, int _nextSize) {
        size = _size;
        nextSize = _nextSize;
        neurons = new double[size]();

        if (_nextSize > 0) {
            biases = new double[_nextSize];
            weights = new double* [_nextSize];
            for (int i = 0; i < _nextSize; i++) {
                weights[i] = new double[size];
            }
        }
    }


    void deinit() {
        delete[] neurons;
        neurons = nullptr;
        if (nextSize > 0)
        {
            if (biases) {
                delete[] biases;
                biases = nullptr;
            }
            if (weights) {
                for (int i = 0; i < nextSize; i++) {
                    delete[] weights[i];
                }
                delete[] weights;
                weights = nullptr;
            }
        }
    }
};


class Perceptron {
private:
    double learningRate;
    Layer* layers = nullptr;
    int layers_length = 0;

    double rand_double(double min, double max) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(min, max);
        return dis(gen);
    }

    double activation(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    double dactivation(double y) {
        return y * (1.0 - y);
    }

    double solve_z(int L, int j) {
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

public:
    Perceptron(double learningRate_, int size, int* sizes) {
        learningRate = learningRate_;
        layers_length = size;
        layers = new Layer[size];
        for (int l = 0; l < size; l++) {
            int nextSize = 0;
            if (l < size - 1) {
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


    double* feedForward(double* inputs, int n) {
        copy_array(n, inputs, layers[0].neurons);

        for (int L = 1; L < layers_length; L++) {
            for (int j = 0; j < layers[L].size; j++) {
                double z = solve_z(L, j);
                layers[L].neurons[j] = activation(z);
            }
        }
        return layers[layers_length - 1].neurons;
    }

    void backpropagation(double* targets) {
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

    void deinit() {
        if (layers) {
            for (int i = 0; i < layers_length; i++) {
                layers[i].deinit();
            }
            delete[] layers;
            layers = nullptr;
        }
    }
};

int main() {
    double learning_rate = 0.01;
    int size = 4;
    int inputs_size = 2;
    int sizes[4] = { inputs_size, 8, 4, 2 };
    Perceptron p(learning_rate, size, sizes);
    int inputs_length = 4;
    double inputs[4][2] = { {0, 0},{0,1},{1,0},{1,1} };

    int output_size = 2;
    double targets[4][2] = { {1, 0}, {0, 1}, {0, 1}, {0, 1} };



    unsigned int epochs = 1000000;
    double* outputs;
    for (unsigned int num = 0; num < epochs; num++) {
        if (num == 0 || num == epochs-1) {
            cout << num << '\n';
        }
        //cout << num << '\n';
        for (int i = 0; i < inputs_length; i++) {
            outputs = p.feedForward(inputs[i], inputs_size);
            if (num == 0 || num == epochs - 1) {
                for (int k = 0; k < output_size; k++) {
                    double tar = targets[i][k];
                    double out = outputs[k];
                    cout << out << " : " << tar << ';';
                }
                cout << '\n';
            }
            p.backpropagation(targets[i]);
        }

    }
    p.deinit();

}
