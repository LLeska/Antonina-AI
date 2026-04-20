#include "Layer.h"
#include <fstream>    

Layer::Layer() {
    size = 0;
    nextSize = 0;
    neurons = nullptr;
    biases = nullptr;
    weights = nullptr;
}

Layer::Layer(int _size, int _nextSize) {
    size = _size;
    nextSize = _nextSize;
    neurons = new double[size]();

    if (nextSize > 0) {
        biases = new double[nextSize];
        weights = new double* [nextSize];
        for (int i = 0; i < nextSize; i++) {
            weights[i] = new double[size];
        }
    }
    else {
        biases = nullptr;
        weights = nullptr;
    }
}

Layer::Layer(const Layer& l) {
    size = l.size;
    nextSize = l.nextSize;
    neurons = new double[size]();
    for (int i = 0; i < size; i++) {
        neurons[i] = l.neurons[i];
    }
    if (nextSize > 0) {
        biases = new double[nextSize];
        weights = new double* [nextSize];
        for (int i = 0; i < nextSize; i++) {
            weights[i] = new double[size];
            biases[i] = l.biases[i];
            for (int j = 0; j < size; j++) {
                weights[i][j] = l.weights[i][j];
            }
        }
    }
    else {
        biases = nullptr;
        weights = nullptr;
    }
}

Layer& Layer::operator=(const Layer& l) {
    if (this == &l) return *this;

    int new_size = l.size;
    int new_next = l.nextSize;

    double* new_neurons = new double[new_size]();
    for (int i = 0; i < new_size; i++) new_neurons[i] = l.neurons[i];

    double* new_biases = nullptr;
    double** new_weights = nullptr;

    if (new_next > 0) {
        new_biases = new double[new_next];
        new_weights = new double* [new_next];
        for (int i = 0; i < new_next; i++) new_weights[i] = nullptr;
        for (int i = 0; i < new_next; i++) {
            new_weights[i] = new double[new_size];
            new_biases[i] = l.biases[i];
            for (int j = 0; j < new_size; j++)
                new_weights[i][j] = l.weights[i][j];
        }
    }

    deInit();
    size = new_size;
    nextSize = new_next;
    neurons = new_neurons;
    biases = new_biases;
    weights = new_weights;
    return *this;
}

Layer::Layer(Layer&& l) {
    size = l.size;
    nextSize = l.nextSize;
    neurons = l.neurons;
    biases = l.biases;
    weights = l.weights;
    l.size = 0;
    l.nextSize = 0;
    l.neurons = nullptr;
    l.biases = nullptr;
    l.weights = nullptr;
}

Layer& Layer::operator=(Layer&& l) {
    if (this != &l) {
        deInit();
        size = l.size;
        nextSize = l.nextSize;
        neurons = l.neurons;
        biases = l.biases;
        weights = l.weights;
        l.size = 0;
        l.nextSize = 0;
        l.neurons = nullptr;
        l.biases = nullptr;
        l.weights = nullptr;
    }
    return *this;
}

void Layer::readFromFile(std::ifstream* fin) {
    this->deInit();
    *fin >> size >> nextSize;
    neurons = new double[size]();
    *fin >> neurons[0];
    for (int i = 1; i < size; i++) {
        *fin >> neurons[i];
    }
    if (nextSize > 0) {
        biases = new double[nextSize];
        weights = new double* [nextSize];
        for (int i = 0; i < nextSize; i++) {
            weights[i] = new double[size];
        }
        *fin >> biases[0];
        for (int i = 1; i < nextSize; i++) {
            *fin >> biases[i];
        }
        for (int i = 0; i < nextSize; i++) {
            for (int j = 0; j < size; j++) {
                *fin >> weights[i][j];
            }
        }
    }
}

void Layer::writeInFile(std::ofstream* fout) {
    *fout << size << ' ' << nextSize << '\n';
    *fout << neurons[0];
    for (int i = 1; i < size; i++) {
        *fout << ' ' << neurons[i];
    }
    *fout << '\n';
    if (nextSize > 0) {
        *fout << biases[0];
        for (int i = 1; i < nextSize; i++) {
            *fout << ' ' << biases[i];
        }
        *fout << '\n';
        for (int i = 0; i < nextSize; i++) {
            *fout << weights[i][0];
            for (int j = 1; j < size; j++) {
                *fout << ' ' << weights[i][j];
            }
            *fout << '\n';
        }
    }
}

void Layer::deInit() {
    if (neurons != nullptr) {
        delete[] neurons;
        neurons = nullptr;
    }
    if (biases != nullptr) {
        delete[] biases;
        biases = nullptr;
    }
    if (weights != nullptr) {
        for (int i = 0; i < nextSize; i++) {
            delete[] weights[i];
        }
        delete[] weights;
        weights = nullptr;
    }
    size = 0;
    nextSize = 0;
}

Layer::~Layer() {
    deInit();
}