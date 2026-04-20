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

    Layer(const Layer& l);

    Layer& operator=(const Layer& other);

    Layer(Layer&& l) noexcept;

    Layer& operator=(Layer&& other) noexcept;

    void readFromFile(std::ifstream* fin);

    void writeInFile(std::ofstream* fout);
    
    void deInit();

    ~Layer();
};

