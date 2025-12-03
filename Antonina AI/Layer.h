#pragma once
#include <fstream>
class Layer
{
public:
    int size;
    int nextSize;
    double* neurons;
    double* biases;
    double** weights;

    Layer();

    Layer(int _size, int _nextSize);

    void readFromFile(std::ifstream* fin);

    void writeInFile(std::ofstream* fout);

    void deinit();
};

