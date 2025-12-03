#pragma once
#include "Layer.h"
#include <string>


class Perceptron
{
private:
    double learningRate;
    Layer* layers;
    int layers_length;

    double rand_double(double min, double max);

    double activation(double x);

    double dactivation(double y);

    double solve_z(int L, int j);

    template <typename T>
    void copy_array(int n, T* array_source, T* array_destination);
public:    
    Perceptron(double learningRate_, int layers_length_, int* sizes);

    Perceptron();

    double* feedForward(double* inputs, int n);

    void backpropagation(double* targets);

    void readFromFile(std::string file);

    void writeInFile(std::string file);

    void deinit();
};

