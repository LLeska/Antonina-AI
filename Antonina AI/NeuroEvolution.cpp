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

int NeuroEvolution::partition(int low, int high) {
	int pivot = fitness[high];
	int i = low - 1;

	for (int j = low; j < high; j++) {
		if (fitness[j] >= pivot) {
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
	Perceptron p = neuros[i + 1];
	neuros[i + 1] = neuros[high];
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
	parents_size = 0;
	sizes = nullptr;
	neuros = nullptr;
	fitness = nullptr;
	clearFitness();
}

NeuroEvolution::NeuroEvolution(double learningRate_, int length_, int* sizes_, int parents_size_ = 10, int population_ = 100) {
	learningRate = learningRate_;
	population = population_;
	length = length_;
	parents_size = parents_size_;
	best_fitness_ever = INT_MIN;
	generations_without_improvement = 0;
	current_epsilon = 0.5;  
	current_mutation_prob = 0.2;
	fitness = nullptr;
	sizes = new int[length];
	for (int i = 0; i < length; i++) {
		sizes[i] = sizes_[i];
	}
	neuros = new Perceptron[population];
	for (int i = 0; i < population; i++) {
		Perceptron p(learningRate, length, sizes);
		neuros[i] = p;
	}
	clearFitness();
	//evolution();
}

NeuroEvolution::~NeuroEvolution() {
	deinit();
}

void NeuroEvolution::feedForward(double* state) {
	for (int i = 0; i < population; i++) {
		neuros[i].feedForward(state);
	}
}

int NeuroEvolution::getPopulation() { return population; }

Perceptron* NeuroEvolution::getNeuros(int i) { return &(neuros[i]); }

int* NeuroEvolution::getFitness() { return fitness; }

void NeuroEvolution::clearFitness() {
	if (fitness != nullptr) {
		delete[] fitness;
		fitness = nullptr;
	}
	fitness = new int[population];
	for (int i = 0; i < population; i++) {
		fitness[i] = 0;
	}
}

void NeuroEvolution::setFitness(int* fitness_) {
	if (fitness == nullptr) {
		fitness = new int[population];
	}
	for (int i = 0; i < population; i++) {
		fitness[i] = fitness_[i];
	}
}



void NeuroEvolution::evolution() {
	if (population <= 0 || neuros == nullptr) return;

	if (!allEqual()) {
		quickSort(0, population - 1);
	}

	int current_best = fitness[0];
	bool improved = false;

	if (current_best > best_fitness_ever) {
		best_fitness_ever = current_best;
		generations_without_improvement = 0;
		improved = true;
		std::cout << "NEW RECORD: " << best_fitness_ever << std::endl;
	}
	else {
		++generations_without_improvement;
		improved = false;
	}

	if (improved) {
		current_epsilon = std::max(0.01, current_epsilon * 0.90);
		current_mutation_prob = std::max(0.01, current_mutation_prob * 0.90);
	}
	else {
		double grow_factor = 1.0 + 0.1 * std::min(generations_without_improvement, 15);
		current_epsilon = std::min(3.0, current_epsilon * grow_factor);
		current_mutation_prob = std::min(0.95, current_mutation_prob * grow_factor);
	}

	static int log_counter = 0;
	if (++log_counter % 10 == 0) {
		std::cout << "Stagnation: " << generations_without_improvement
			<< " | Eps: " << current_epsilon
			<< " | MutProb: " << current_mutation_prob
			<< " | Best: " << best_fitness_ever << std::endl;
	}

	if (generations_without_improvement > 200) {
		std::cout << "MAJOR RESET after " << generations_without_improvement
			<< " gens! Best was: " << best_fitness_ever << std::endl;

		int survivors = population / 20;  
		Perceptron* saved = new Perceptron[survivors];
		for (int i = 0; i < survivors; ++i) saved[i] = neuros[i];

		for (int i = survivors; i < population; ++i) {
			Perceptron random_one(learningRate, length, sizes);
			neuros[i] = random_one;
		}

		
		for (int i = 0; i < survivors; ++i) neuros[i] = saved[i];
		delete[] saved;

		generations_without_improvement = 0;
		current_epsilon = 2.0; 
		current_mutation_prob = 0.6;
		clearFitness();
		return;
	}

	int elite_count = std::max(20, population / 8);
	int immigrants = std::max(50, population / 15);

	int use_parents = std::min(parents_size, population);
	Perceptron* best = new Perceptron[use_parents];
	for (int i = 0; i < use_parents; ++i) best[i] = neuros[i];

	Perceptron* nextGen = new Perceptron[population];

	for (int i = 0; i < elite_count; ++i) {
		nextGen[i] = neuros[i];
	}

	static thread_local std::mt19937 gen((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
	std::uniform_int_distribution<int> parentDist(0, use_parents - 1);

	long long avg_fitness = 0;
	for (int i = 0; i < population; ++i) {
		avg_fitness += fitness[i];
	}
	avg_fitness /= population;

	double diversity = (double)avg_fitness / (fitness[0] + 1.0);

	
	int tournament_size;
	if (generations_without_improvement > 100) {
		tournament_size = 2;  
	}
	else if (generations_without_improvement > 50) {
		tournament_size = diversity > 0.75 ? 3 : 2;
	}
	else {
		tournament_size = diversity > 0.85 ? 4 : 3;
	}

	auto tournament = [&]()->int {
		int winner = parentDist(gen);
		for (int t = 1; t < tournament_size; ++t) {
			int cand = parentDist(gen);
			if (fitness[cand] > fitness[winner]) winner = cand;
		}
		return winner;
		};

	int idx = elite_count;
	while (idx < population - immigrants) {
		int i1 = tournament();

		Perceptron child;

		if (generations_without_improvement > 80 || random_in_range(0.0, 1.0) < 0.5) {
			child = best[i1];
		}
		else {
			int i2 = tournament();
			child = Perceptron(&best[i1], &best[i2]);
		}

		double eps = current_epsilon;
		double prob = current_mutation_prob;

		if (generations_without_improvement > 30) {
			eps *= random_in_range(0.7, 1.6);
			prob *= random_in_range(0.8, 1.4);
		}

		child.mutate(eps, prob);
		nextGen[idx++] = child;
	}

	for (int i = population - immigrants; i < population; ++i) {
		Perceptron random_one(learningRate, length, sizes);
		nextGen[i] = random_one;
	}

	delete[] best;
	delete[] neuros;
	neuros = nextGen;
	clearFitness();
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
	int population_;
	int parents_size_;
	*fin >> learningRate >> length
		>> population_ >> parents_size_
		>> best_fitness_ever >> generations_without_improvement
		>> current_epsilon >> current_mutation_prob;

	sizes = new int[length];
	for (int i = 0; i < length; i++) {
		*fin >> sizes[i];
	}

	neuros = new Perceptron[std::max(population, population_)];
	for (int i = 0; i < std::min(population, population_); i++) {
		neuros[i].readFromFile(fin);
	}
	for (int i = 0; i < population - population_; i++) {
		Perceptron p(learningRate, length, sizes);
		neuros[population_ + i] = p;
	}

	std::cout << "Loaded: best=" << best_fitness_ever
		<< " stagnation=" << generations_without_improvement
		<< " eps=" << current_epsilon
		<< " prob=" << current_mutation_prob << std::endl;

	if (generations_without_improvement > 150) {
		std::cout << " WARNING: Long stagnation detected! Resetting mutation params..." << std::endl;
		generations_without_improvement = 0; 
		current_epsilon = 1.5;  
		current_mutation_prob = 0.5;
		std::cout << "Reset to: eps=" << current_epsilon
			<< " prob=" << current_mutation_prob << std::endl;
	}
}


void NeuroEvolution::writeInFile(std::ofstream* fout) {
	*fout << learningRate << ' ' << length \
		<< ' ' << population<<' ' << \
		parents_size << ' ' << best_fitness_ever \
		<< ' ' << generations_without_improvement\
		<< ' ' << current_epsilon << ' ' \
		<< current_mutation_prob << '\n';
	*fout << sizes[0];
	for (int i = 1; i < length; i++) {
		*fout << ' ' << sizes[i];
	}
	*fout << '\n';
	for (int i = 0; i < population; i++) {
		neuros[i].writeInFile(fout);
	}
}

bool NeuroEvolution::allEqual() {
	for (int i = 1; i < population; i++) {
		if (fitness[0] != fitness[i]) return false;
	}
	return true;
}

Perceptron* NeuroEvolution::demonstrate() {
	return &neuros[0];
}

void NeuroEvolution::deinit() {
	if (sizes != nullptr) {
		delete[] sizes;
	}
	if (neuros != nullptr) {
		delete[] neuros;
	}
}
