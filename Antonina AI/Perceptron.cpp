#include "Perceptron.h"
#include <random>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>


Perceptron::Perceptron() {
    EPSILON = 0.1;
    NOT_MUTAHION = 0.04;
    learningRate = 0;
    length = 0;
    layers = nullptr;
}

Perceptron::Perceptron(double learningRate_, int length_, int* sizes) {
    EPSILON = 0.1;
    NOT_MUTAHION = 0.04;
    learningRate = learningRate_;
    length = length_;
    layers = new Layer[length];

    static thread_local std::mt19937 gen(
        (unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int l = 0; l < length; l++) {
        int nextSize = (l < length - 1) ? sizes[l + 1] : 0;
        layers[l] = Layer(sizes[l], nextSize);

        if (nextSize > 0 && layers[l].biases && layers[l].weights) {
            for (int j = 0; j < nextSize; j++) {
                layers[l].biases[j] = dist(gen);
                double range = sqrt(6.0 / (sizes[l] + nextSize));
                for (int k = 0; k < sizes[l]; k++)
                    layers[l].weights[j * sizes[l] + k] = dist(gen) * range;
            }
        }
    }
}

Perceptron::Perceptron(const Perceptron& p) {
    EPSILON = p.EPSILON;
    NOT_MUTAHION = p.NOT_MUTAHION;
    learningRate = p.learningRate;
    length = p.length;
    if (p.layers && length > 0) {
        layers = new Layer[length];
        for (int i = 0; i < length; i++) layers[i] = p.layers[i];
    }
    else {
        layers = nullptr;
    }
}

Perceptron::Perceptron(Perceptron&& p) {
    EPSILON = p.EPSILON;
    NOT_MUTAHION = p.NOT_MUTAHION;
    learningRate = p.learningRate;
    length = p.length;
    layers = p.layers;
    p.layers = nullptr;
    p.length = 0;
}

Perceptron::Perceptron(Perceptron* p1, Perceptron* p2) {
    EPSILON = 0.1;
    NOT_MUTAHION = 0.04;
    learningRate = p1->learningRate;
    length = p1->length;
    layers = new Layer[length];

    static thread_local std::mt19937 gen(
        (unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    for (int i = 0; i < length; i++) {
        int size = p1->layers[i].size;
        int nextSize = p1->layers[i].nextSize;
        layers[i] = Layer(size, nextSize);
        if (nextSize <= 0) continue;

        for (int j = 0; j < nextSize; j++) {
            double alpha = uni01(gen);
            layers[i].biases[j] = alpha * p1->layers[i].biases[j]
                + (1.0 - alpha) * p2->layers[i].biases[j];
            for (int k = 0; k < size; k++) {
                double aw = uni01(gen);
                layers[i].weights[j * size + k] =
                    aw * p1->layers[i].weights[j * size + k] +
                    (1.0 - aw) * p2->layers[i].weights[j * size + k];
            }
        }
    }
}


Perceptron& Perceptron::operator=(const Perceptron& p) {
    if (this == &p) return *this;
    deInit();
    EPSILON = p.EPSILON;
    NOT_MUTAHION = p.NOT_MUTAHION;
    learningRate = p.learningRate;
    length = p.length;
    if (p.layers && length > 0) {
        layers = new Layer[length];
        for (int i = 0; i < length; i++) layers[i] = p.layers[i];
    }
    else {
        layers = nullptr;
    }
    return *this;
}

Perceptron& Perceptron::operator=(Perceptron&& p) {
    if (this != &p) {
        deInit();
        EPSILON = p.EPSILON;
        NOT_MUTAHION = p.NOT_MUTAHION;
        learningRate = p.learningRate;
        length = p.length;
        layers = p.layers;
        p.layers = nullptr;
        p.length = 0;
    }
    return *this;
}


Perceptron::~Perceptron() { deInit(); }

void Perceptron::deInit() {
    if (layers) { delete[] layers; layers = nullptr; }
    length = 0;
}


void Perceptron::feedForward(double* inputs) {
    for (int i = 0; i < layers[0].size; i++)
        layers[0].neurons[i] = inputs[i];

    for (int L = 1; L < length; L++) {
        Layer& prev = layers[L - 1];
        Layer& curr = layers[L];
        for (int j = 0; j < curr.size; j++) {
            double z = prev.biases[j];
            for (int k = 0; k < prev.size; k++)
                z += prev.neurons[k] * prev.weights[j * prev.size + k];
            curr.neurons[j] = 1.0 / (1.0 + exp(-z));
        }
    }
}

int Perceptron::getOut() {
    int maxi = 0;
    for (int i = 1; i < layers[length - 1].size; i++)
        if (layers[length - 1].neurons[maxi] < layers[length - 1].neurons[i])
            maxi = i;
    return maxi;
}

void Perceptron::mutate(double sigma, double prob) {
    if (sigma > 0.0) EPSILON = sigma;
    if (prob >= 0.0) NOT_MUTAHION = prob;

    static thread_local std::mt19937 gen(
        (unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::normal_distribution<double> gauss(0.0, EPSILON);
    const double WEIGHT_MIN = -10.0;
    const double WEIGHT_MAX = 10.0;

    for (int l = 0; l < length; l++) {
        int size = layers[l].size;
        int nextSize = layers[l].nextSize;
        if (nextSize <= 0) continue;

        for (int j = 0; j < nextSize; j++) {
            if (uni01(gen) < NOT_MUTAHION) {
                layers[l].biases[j] += gauss(gen);
                layers[l].biases[j] = std::clamp(layers[l].biases[j], WEIGHT_MIN, WEIGHT_MAX);
            }
            for (int k = 0; k < size; k++) {
                if (uni01(gen) < NOT_MUTAHION) {
                    layers[l].weights[j * size + k] += gauss(gen);
                    layers[l].weights[j * size + k] = std::clamp(layers[l].weights[j * size + k], WEIGHT_MIN, WEIGHT_MAX);
                }
            }
        }
    }
}

bool Perceptron::isInitialized() { return layers != nullptr; }


void Perceptron::readFromFile(std::ifstream* fin) {
    if (!fin->is_open() || fin->fail()) {
        std::cerr << "Stream error before reading Perceptron" << std::endl;
        return;
    }
    deInit();
    *fin >> learningRate >> length;
    if (fin->fail() || length <= 0 || length > 100) {
        std::cerr << "Invalid length=" << length << std::endl;
        length = 0;
        return;
    }
    layers = new Layer[length];
    for (int i = 0; i < length; i++) {
        layers[i].readFromFile(fin);
        if (fin->fail()) {
            std::cerr << "Error reading Layer " << i << std::endl;
            break;
        }
    }
}

void Perceptron::writeInFile(std::ofstream* fout) {
    *fout << learningRate << ' ' << length << '\n';
    for (int i = 0; i < length; i++) layers[i].writeInFile(fout);
}

void Perceptron::readFromFile(std::string file) {
    std::ifstream fin(file);
    readFromFile(&fin);
}

void Perceptron::writeInFile(std::string file) {
    std::ofstream fout(file);
    writeInFile(&fout);
}


template<typename T>
T Perceptron::random_in_range(T a, T b) {
    static thread_local std::mt19937 gen(std::random_device{}());
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> dist(a, b);
        return dist(gen);
    }
    else {
        std::uniform_real_distribution<T> dist(a, b);
        return dist(gen);
    }
}