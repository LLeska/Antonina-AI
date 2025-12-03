#include "Layer.h"
#include <fstream>    
Layer::Layer() {
    nextSize = 0;
    neurons = nullptr;
}

Layer::Layer(int _size, int _nextSize) {
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

void Layer::readFromFile(std::ifstream* fin) {
    this->deinit();
    *fin >> size >> nextSize;
    neurons = new double[size]();
    if (nextSize > 0) {
        biases = new double[nextSize];
        weights = new double* [nextSize];
        for (int i = 0; i < nextSize; i++) {
            weights[i] = new double[size];
        }
    
        *fin >> neurons[0];
        for (int i = 1; i < size; i++) {
            *fin >> neurons[i];
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

void Layer::deinit() {
    if (neurons && neurons != nullptr) {
        delete[] neurons;
        neurons = nullptr;
    }
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