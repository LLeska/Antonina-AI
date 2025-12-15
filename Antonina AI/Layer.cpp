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
    if (this != &l) {
        deInit();
        size = l.size;
        nextSize = l.nextSize;
        if (size > 0) {
            neurons = new double[size];
            for (int i = 0; i < size; i++) {
                neurons[i] = l.neurons[i];
            }
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
    if (nextSize > 0)
    {
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
    }
}

Layer::~Layer() {
    deInit();
}