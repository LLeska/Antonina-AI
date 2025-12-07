#include "Perceptron.h"
#include <random>
#include <iostream>
#include <chrono>

double EPSILON_ = 0.1;
double NOT_MUTAHION_ = 0.2;

template<typename T>
T Perceptron::random_in_range(T a, T b) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> dist(a, b);
        return dist(gen);
    }
    else if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(a, b);
        return dist(gen);
    }
}

double Perceptron::activation(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double Perceptron::dactivation(double y) {
    return y * (1.0 - y);
}

double Perceptron::solveZ(int L, int j) {
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
void Perceptron::copyArray(int n, T* array_source, T* array_destination) {
    for (int i = 0; i < n; i++) {
        array_destination[i] = array_source[i];
    }
}

Perceptron::Perceptron() {
    EPSILON = EPSILON_;
    NOT_MUTAHION = NOT_MUTAHION_;
    learningRate = 0;
    length = 0;
    layers = nullptr;
}

Perceptron::Perceptron(Perceptron& p) {
    EPSILON = p.EPSILON;
    NOT_MUTAHION = p.NOT_MUTAHION;
    learningRate = p.learningRate;
    length = p.length;
    if (p.layers != nullptr && length > 0) {
        layers = new Layer[length];
        for (int i = 0; i < length; i++) {
            layers[i] = p.layers[i];
        }
    }
    else {
        layers = nullptr;
    }
}

Perceptron::Perceptron(double learningRate_, int length_, int* sizes) {
    EPSILON = EPSILON_;
    NOT_MUTAHION = NOT_MUTAHION_;
    learningRate = learningRate_;
    length = length_;
    layers = new Layer[length];
    for (int l = 0; l < length; l++) {
        int nextSize = 0;
        if (l < length - 1) {
            nextSize = sizes[l + 1];
        }
        layers[l] = Layer(sizes[l], nextSize);
        if (nextSize > 0 && layers[l].biases && layers[l].weights) {
            for (int j = 0; j < nextSize; j++) {
                layers[l].biases[j] = random_in_range(-1.0, 1.0);
                for (int k = 0; k < sizes[l]; k++) {
                    layers[l].weights[j][k] = random_in_range(-1.0, 1.0);
                }
            }  
        }
    }
}

Perceptron& Perceptron::operator=(const Perceptron& p) {
    if (this == &p) {
        return *this;
    }
    deInit();
    EPSILON = p.EPSILON;
    NOT_MUTAHION = p.NOT_MUTAHION;
    learningRate = p.learningRate;
    length = p.length;
    if (p.layers && length > 0) {
        layers = new Layer[length];
        for (int i = 0; i < length; i++) {
            layers[i] = p.layers[i];
        }
    }
    else {
        layers = nullptr;
    }
    return *this;

}

void Perceptron::backpropagation(double* targets) {
    int last = length - 1;
    double** deltas = new double* [length];
    for (int i = 0; i < length; i++) {
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
    for (int L = 0; L < length - 1; L++) {
        Layer& curr = layers[L];
        Layer& next = layers[L + 1];
        for (int j = 0; j < next.size; j++) {
            for (int k = 0; k < curr.size; k++) {
                curr.weights[j][k] += learningRate * deltas[L + 1][j] * curr.neurons[k];
            }
            curr.biases[j] += learningRate * deltas[L + 1][j];
        }
    }
    for (int i = 0; i < length; i++) {
        delete[] deltas[i];
    }
    delete[] deltas;
}

Perceptron::Perceptron(Perceptron* p1, Perceptron* p2) {
    learningRate = p1->learningRate;
    length = p1->length;
    layers = new Layer[length];
    for (int i = 0; i < length; i++) {
        if (random_in_range(0.0, 1.0) > 0.5) {
            layers[i] = p1->layers[i];
        }
        else {
            layers[i] = p2->layers[i];
        }
    }
    if (random_in_range(0.0, 1.0) > NOT_MUTAHION) {
        double* tar = new double[layers[length - 1].size];
        for (int i = 0; i < layers[length - 1].size; i++) {
            tar[i] = random_in_range(0.0, 1.0)*EPSILON;
        }
        backpropagation(tar);
        delete[] tar;
    }
}

int Perceptron::getOut() {
    int maxi = 0;
    for (int i = 1; i < layers[length - 1].size; i++) {
        if (layers[length - 1].neurons[maxi] < layers[length - 1].neurons[0]) {
            maxi = i;
        }
    }
    return maxi;
}

void Perceptron::feedForward(double* inputs) {
    copyArray(layers[0].size, inputs, layers[0].neurons);

    for (int L = 1; L < length; L++) {
        for (int j = 0; j < layers[L].size; j++) {
            double z = solveZ(L, j);
            layers[L].neurons[j] = activation(z);
        }
    }
}

bool Perceptron::isInitialized() {
    if (layers == nullptr) {
        return false;
    }
    return true;
}


Perceptron::~Perceptron() {
    deInit();
}

void Perceptron::deInit() {
    if (layers) {
        for (int i = 0; i < length; i++) {
            layers[i].deInit();
        }
        delete[] layers;
        layers = nullptr;
    }
}

void Perceptron::readFromFile(std::ifstream* fin) {
    if (!fin->is_open()) {
        std::cerr << "Ошибка открытия файла" << std::endl;
        return;
    }
    this->deInit();
    *fin >> learningRate >> length;
    layers = new Layer[length];
    for (int i = 0; i < length; i++) {
        layers[i].readFromFile(fin);
    }
}

void Perceptron::writeInFile(std::ofstream* fout) {
    *fout << learningRate << ' ' << length << '\n';
    for (int i = 0; i < length; i++) {
        layers[i].writeInFile(fout);
    }
}

void Perceptron::readFromFile(std::string file) {
    std::ifstream fin(file, std::ios::in);
    readFromFile(&fin);
    fin.close();
}

void Perceptron::writeInFile(std::string file) {
    std::ofstream fout(file, std::ios::out);
    writeInFile(&fout);
    fout.close();
}
