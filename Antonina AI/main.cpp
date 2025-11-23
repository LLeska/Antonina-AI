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
    int size;
    double* neurons;
    double* biases;
    double** weights;
    int weight_size;
    int weight_len;

    Layer(int _size, int _nextSize) {
        size = _size;
        neurons = new double[size];
        biases = new double[size];
        weights = new double*[_nextSize];
        for (int i = 0; i < _size; i++) {
            weights[i] = new double[_nextSize];
        }
        weight_size = _nextSize;
    }

    Layer() {

    }
    
    void deinit() {
        delete[] neurons;
        delete[] biases;
        for (int i = 0; i < size; i++) {
            delete[] weights[i];
        }
        delete[] weights;
    }
};


class Perceptron {
private:
    double learningRate;
    Layer* layers;
    int layers_length;


public:
    Perceptron(double learningRate_, int size, int* sizes) {
        learningRate = learningRate_;
        layers_length = size;
        layers = new Layer[size];
        for (int l = 0; l < size; l++) {
            int nextSize = 0;
            if (l < size - 1) nextSize = sizes[l + 1];
            layers[l] = Layer(sizes[l], nextSize);
            for (int i = 0; i < sizes[l]; i++) { 
                layers[l].biases[i] = rand_double(0.0, 1.0)*2.0 - 1.0;
                for (int j = 0; j < nextSize; j++) {
                    layers[l].weights[i][j] = rand_double(0.0, 1.0) * 2.0 - 1.0;
                }
            }
        }
    }


    double activation(double x) {
        return 1 / (1 + exp(-x));
    }

    double dactivation(double y) {
        return y * (1 - y);
    }



    double* feedForward(double* inputs, int n) {
        copy_array(n, inputs, layers[0].neurons);

        for (int i = 1; i < layers_length; i++) {
            Layer l = layers[i - 1];
            Layer l1 = layers[i];
            for (int j = 0; j < l1.size; j++) {
                l1.neurons[j] = 0;
                for (int k = 0; k < l.size; k++) {
                    l1.neurons[j] += l.neurons[k] * l.weights[k][j];
                }
                l1.neurons[j] += l1.biases[j];
                for (int ind = 0; ind < l1.size; ind++) {
                    l1.neurons[j] = activation(l1.neurons[j]);
                }
            }
        }
        return layers[layers_length - 1].neurons;
    }

    void backpropagation(double* targets) {
        double* errors = new double[layers[layers_length - 1].size];
        for (int i = 0; i < layers[layers_length - 1].size; i++) {
            errors[i] = targets[i] - layers[layers_length - 1].neurons[i];
        }
        for (int k = layers_length - 2; k >= 0; k--) {
            Layer l = layers[k];
            Layer l1 = layers[k + 1];
            double* errorsNext = new double[l.size];
            double* gradients = new double[l1.size];
            for (int i = 0; i < l1.size; i++) {
                gradients[i] = errors[i] * dactivation(layers[k + 1].neurons[i]);
                gradients[i] *= learningRate;
            }
            double** deltas = new double*[l.size];
            for (int ind = 0; ind < l1.size; ind++) {
                deltas[ind] = new double[l.size];
            }
            for (int i = 0; i < l1.size; i++) {
                for (int j = 0; j < l.size; j++) {
                    deltas[i][j] = gradients[i] * l.neurons[j];
                }
            }
            for (int i = 0; i < l.size; i++) {
                errorsNext[i] = 0;
                for (int j = 0; j < l1.size; j++) {
                    errorsNext[i] += l.weights[i][j] * errors[j];
                }
            }
            errors = new double[l.size];
            copy_array(l.size, errorsNext, errors);
            double** weightsNew = new double*[l.weight_size];
            for (int ind = 0; ind < l.size; ind++) {
                weightsNew[ind] = new double[l.weight_size];
            }

            for (int i = 0; i < l1.size; i++) {
                for (int j = 0; j < l.size; j++) {
                    weightsNew[j][i] = l.weights[j][i] + deltas[i][j];
                }
            }
            l.weights = weightsNew;
            for (int i = 0; i < l1.size; i++) {
                l1.biases[i] += gradients[i];
            }
            for (int ind = 0; ind < l.size; ind++) {
                 delete[] weightsNew[ind];
            }
            for (int ind = 0; ind < l1.size; ind++) {
                delete[] deltas[ind];
            }
            //delete[] weightsNew;
            delete[] errorsNext;
            delete[] gradients;
            //delete[] deltas;
        }
        delete[] errors;
    }

    void deinit() {
        delete[] layers;
    }
};

int main() {
    double learning_rate = 0.001;
    int size = 2;
    int inputs_size = 2;
    int sizes[2] = {inputs_size, 1 };
    Perceptron p(learning_rate, size, sizes);
    int inputs_length = 4;
    double inputs[4][2] = { {0, 0},{0,1},{1,0},{1,1} };

    int output_size = 1;
    double targets[4][1] = { {0}, {0}, {0}, {1} };

    

    int epochs = 10000;
    double* outputs;
    for (int _ = 0; _ < epochs; _++) {
        cout << _ << '\n';
        for (int i = 0; i < 4; i++) {
            outputs = p.feedForward(inputs[i], inputs_size);            
            p.backpropagation(targets[i]);
            for (int k = 0; k < output_size; k++) {
                cout << outputs[k] << " : " << targets[i][k] << '\n';
            } 
        }
        cout << '\n';
    }
    delete[] outputs;
    p.deinit();
    
}