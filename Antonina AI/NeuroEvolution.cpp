#include "NeuroEvolution.h"
#include <random>
#include <iostream>
#include <chrono>


template<typename T>
T NeuroEvolution::random_in_range(T a, T b) {
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

int NeuroEvolution::partition( int low, int high) {
	int pivot = fitness[high];
	int i = low - 1; 

	for (int j = low; j < high; j++) {
		if (fitness[j] <= pivot) {
			i++;
			Perceptron p = neuros[i];
			neuros[i] = neuros[j];
			neuros[j] = p;
			int temp = fitness[i];
			fitness[i] = fitness[j];
			fitness[j] = temp;
		}
	}
	int temp = fitness[i + 1];
	fitness[i + 1] = fitness[high];
	fitness[high] = temp;
	Perceptron p = neuros[i];
	neuros[i] = neuros[high];
	neuros[high] = p;

	return i + 1;
}

void NeuroEvolution::quickSort(int low, int high) {
	if (low < high) {
		int pi = partition(low, high);
		quickSort(low, pi - 1);
		quickSort(pi + 1, high);
	}
}

NeuroEvolution::NeuroEvolution() {
	learningRate = 0;
	population = 0;
	length = 0;
	sizes = nullptr;
	neuros = nullptr;
	fitness = nullptr;
}

NeuroEvolution::NeuroEvolution(double learningRate_, int length_, int* sizes_, int parents_size_ = 10, int population_ = 100) {
	if (length_ < 2) {
		std::cerr << "Ошибка инициализации, некоректные входные данные";
		return;
	}
	learningRate = learningRate_;
	population = population_;
	length = length_;
	parents_size = parents_size_;
	fitness = nullptr;
	sizes = new int[length];
	for (int i = 0; i < length; i++) {
		sizes[i] = sizes_[i];
	}
	neuros = new Perceptron[population];
	for (int i = 0; i < population; i++) {
		neuros[i] = Perceptron(learningRate, length, sizes);
	}
}

NeuroEvolution::NeuroEvolution(NeuroEvolution& ne) {
	population = ne.population;
	length = ne.length;
	sizes = nullptr;
	fitness = nullptr;
}

NeuroEvolution::~NeuroEvolution() {
	deinit();
}

void NeuroEvolution::feedForward(double *state) {
	for (int i = 0; i < population; i++) {
		neuros[i].feedForward(state);
	}
}

int NeuroEvolution::getPopulation() { return population; }

Perceptron** NeuroEvolution::getNeuros() { return &neuros; }

int* NeuroEvolution::getFitness() { return fitness; }

void NeuroEvolution::clearFitness() {
	if (fitness != nullptr) {
		delete[] fitness;
		fitness = nullptr;
	}
	fitness = new int[population];
}

void NeuroEvolution::setFitness(int* fitness_) {
	clearFitness();
	for (int i = 0; i < population; i++) {
		fitness[i] = fitness_[i];
	}
}


void NeuroEvolution::evolution() {
	quickSort(0, population);
	Perceptron* best = new Perceptron[parents_size];
	for (int i = 0; i < parents_size; i++) {
		best[i] = neuros[i];
	}
	delete[] neuros;
	neuros = new Perceptron[population];
	for (int i = 0; i < parents_size/5; i++) {
		neuros[i] = best[i];
	}
	for (int i = parents_size / 5; i < population - parents_size / 5; i++) {
		int ind1 = random_in_range(0, parents_size - 1);
		int ind2 = random_in_range(0, parents_size - 1);
		neuros[i] = Perceptron(best+ind1, best+ind2);
	}
}

void NeuroEvolution::readFromFile(std::string file) {
	std::ifstream fin(file, std::ios::in);
	readFromFile(&fin);
	fin.close();
}

void NeuroEvolution::writeInFile(std::string file) {
	std::ofstream fout(file, std::ios::out);
	writeInFile(&fout);
	fout.close();
}

void NeuroEvolution::readFromFile(std::ifstream* fin) {
	if (!fin->is_open()) {
        std::cerr << "Ошибка открытия файла" << std::endl;
        return;
    }
    this->deinit();
    *fin >> learningRate >> length >> population;
	sizes = new int[length];
	for (int i = 0; i < length; i++) {
		*fin >> sizes[i];
	}
	neuros = new Perceptron[population];
	for (int i = 0; i < population; i++) {
		neuros[i].readFromFile(fin);
	}
	bool flag;
	*fin >> flag;
	if (flag) {
		fitness = new int[population];
		for (int i = 0; i < population; i++) {
			*fin >> population;
		}
	}
}

void NeuroEvolution::writeInFile(std::ofstream* fout) {
	*fout << learningRate << ' ' << length << ' ' << population << '\n';
	*fout << sizes[0];
	for (int i = 0; i < length; i++) {
		*fout << ' ' << sizes[i];
	}
	*fout << '\n';
	for (int i = 0; i < population; i++) {
		neuros[i].writeInFile(fout);
	}
	if (fitness == nullptr) *fout << false << '\n';
	else {
		*fout << true << '\n';
		*fout << fitness[0];
		for (int i = 0; i < population; i++) {
			*fout << ' ' << fitness[i];
		}
		*fout << '\n';
	}
}

Perceptron* NeuroEvolution::demonstrate() {
	return &neuros[0];
}

void NeuroEvolution::deinit() {
	population = 0;
	length = 0;
	if (sizes != nullptr) {
		delete[] sizes;
	}
	if (neuros != nullptr) {
		delete[] neuros;
	}
}
