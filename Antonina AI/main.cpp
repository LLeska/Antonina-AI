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
   

    Layer(int _size, int _nextSize) {
        size = _size;
        neurons = new double[size];
        biases = new double[size];
        weights = new double*[_size];
        for (int i = 0; i < _nextSize; i++) {
            weights[i] = new double[_size];
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
            for (int k = 0; k < sizes[l]; k++) { 
                layers[l].biases[k] = rand_double(0.0, 1.0)*2.0 - 1.0;
                for (int j = 0; j < nextSize; j++) {
                    layers[l].weights[j][k] = rand_double(0.0, 1.0) * 2.0 - 1.0;
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
        for (int L = 1; L < layers_length; L++) {
            for (int j = 0; j < layers[L].size; j++) {
                double z = solve_z(L, j);
                layers[L].neurons[j] = activation(z);
            }
        }
        return layers[layers_length - 1].neurons;
    }


    double solve_z(int L, int j) {
        Layer l = layers[L - 1];
        Layer l1 = layers[L];
        l1.neurons[j] = 0;
        for (int k = 0; k < l.size; k++) {
            l1.neurons[j] += l.neurons[k] * l.weights[j][k];
        }
        l1.neurons[j] += l1.biases[j];
        return l1.neurons[j];
    }


    void backpropagation(double* targets) {
        double** costs = new double* [layers_length];
        double** dcosts = new double* [layers_length];
        for (int i = 0; i < layers_length; i++) {
            costs[i] = new double[layers[i].size];
            dcosts[i] = new double[layers[i].size];
        }
        copy_array(layers[layers_length - 1].size, layers[layers_length - 1].neurons, costs[layers_length - 1]);
        for (int i = 0; i < layers[layers_length - 1].size; i++) {
            costs[layers_length - 1][i] -= targets[i];
            dcosts[layers_length - 1][i] = 2 * costs[layers_length - 1][i];
            costs[layers_length - 1][i] *= costs[layers_length - 1][i];
        }


        for (int L = layers_length - 2; L >= 0; L--) {
            for (int k = 0; k < layers[L].size; k++) {
                dcosts[L][k] = 0;
                for (int j = 0; j < layers[L+1].size; j++) {
                    double z = solve_z(L+1, j);
                    dcosts[L][k] += layers[L].weights[j][k] * dactivation(z) * dcosts[L + 1][j];
                }
            }
        }
    }

    void deinit() {
        delete[] layers;
    }
};

int main() {
    double learning_rate = 0.001;
    int size = 2;
    int inputs_size = 2;
    int sizes[2] = {inputs_size, 2 };
    Perceptron p(learning_rate, size, sizes);
    int inputs_length = 4;
    double inputs[4][2] = { {0, 0},{0,1},{1,0},{1,1} };

    int output_size = 2;
    double targets[4][2] = { {1, 0}, {1, 0}, {1, 0}, {0, 1} };

    

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