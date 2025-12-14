#pragma once
#include "Perceptron.h"
#include <fstream>
#include <string>

class NeuroEvolution
{
private:
	int population;
	double learningRate;
	int length;
	int parents_size;
	Perceptron* neuros;
	int* sizes;
	int* fitness;
	
	template<typename T>
	T random_in_range(T a, T b);

	void clearFitness();

	int partition(int low, int high);

	void quickSort(int low, int high);
	

public:
	NeuroEvolution();

	NeuroEvolution(double learningRate_, int length_, int* sizes_, int parents_size_, int population_);

	~NeuroEvolution();

	void feedForward(double* state);

	int getPopulation();

	Perceptron** getNeuros();

	int* getFitness();

	void setFitness(int* fitness_);

	void evolution();

	void readFromFile(std::string file);

	void writeInFile(std::string file);

	void readFromFile(std::ifstream* fin);

	void writeInFile(std::ofstream* fout);

	bool allEqual();

	Perceptron* demonstrate();

	void deinit();

};

