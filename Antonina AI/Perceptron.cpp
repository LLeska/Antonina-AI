#include "Perceptron.h"
#include <random>
#include <iostream>
#include <chrono>

double EPSILON_ = 0.1;
double NOT_MUTAHION_ = 0.04;

template<typename T>
T Perceptron::random_in_range(T a, T b) {
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

double Perceptron::activation(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double Perceptron::dactivation(double y) {
    return y * (1.0 - y);
}

double Perceptron::solveZ(int L, int j) {
    Layer& prev_layer = layers[L - 1];
    Layer& curr_layer = layers[L];
    double z = 0.0;
    if (prev_layer.weights) {
        for (int k = 0; k < prev_layer.size; k++) {
            z += prev_layer.neurons[k] * prev_layer.weights[j][k];
        }
    }
    if (prev_layer.biases) {
        z += prev_layer.biases[j];
    }
    return z;
}


template <typename T>
void Perceptron::copyArray(int n, T* array_source, T* array_destination) {
    for (int i = 0; i < n; i++) {
        array_destination[i] = array_source[i];
    }
}

Perceptron::Perceptron() {
    EPSILON = EPSILON_;
    NOT_MUTAHION = NOT_MUTAHION_;
    learningRate = 0;
    length = 0;
    layers = nullptr;
}

Perceptron::Perceptron(Perceptron& p) {
    EPSILON = p.EPSILON;
    NOT_MUTAHION = p.NOT_MUTAHION;
    learningRate = p.learningRate;
    length = p.length;
    if (p.layers != nullptr && length > 0) {
        layers = new Layer[length];
        for (int i = 0; i < length; i++) {
            layers[i] = p.layers[i];
        }
    }
    else {
        layers = nullptr;
    }
}

Perceptron::Perceptron(double learningRate_, int length_, int* sizes) {
    EPSILON = EPSILON_;
    NOT_MUTAHION = NOT_MUTAHION_;
    learningRate = learningRate_;
    length = length_;
    layers = new Layer[length];

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    static std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    for (int l = 0; l < length; l++) {
        int nextSize = 0;
        if (l < length - 1) {
            nextSize = sizes[l + 1];
        }
        layers[l] = Layer(sizes[l], nextSize);

        if (nextSize > 0 && layers[l].biases && layers[l].weights) {
            for (int j = 0; j < nextSize; j++) {
                layers[l].biases[j] = distribution(generator);

                for (int k = 0; k < sizes[l]; k++) {
                    double range = sqrt(6.0 / (sizes[l] + nextSize));
                    layers[l].weights[j][k] = distribution(generator) * range;
                }
            }
        }
    }
}

Perceptron& Perceptron::operator=(const Perceptron& p) {
    if (this == &p) {
        return *this;
    }
    deInit();
    EPSILON = p.EPSILON;
    NOT_MUTAHION = p.NOT_MUTAHION;
    learningRate = p.learningRate;
    length = p.length;
    if (p.layers && length > 0) {
        layers = new Layer[length];
        for (int i = 0; i < length; i++) {
            layers[i] = p.layers[i];
        }
    }
    else {
        layers = nullptr;
    }
    return *this;

}

void Perceptron::backpropagation(double* targets) {
    int last = length - 1;
    double** deltas = new double* [length];
    for (int i = 0; i < length; i++) {
        deltas[i] = new double[layers[i].size]();
    }
    for (int j = 0; j < layers[last].size; j++) {
        double output = layers[last].neurons[j];
        double error = targets[j] - output;
        deltas[last][j] = error * dactivation(output);
    }
    for (int L = last - 1; L >= 1; L--) {
        Layer& curr = layers[L];
        Layer& next = layers[L + 1];

        for (int k = 0; k < curr.size; k++) {
            double error = 0;
            for (int j = 0; j < next.size; j++) {
                error += layers[L].weights[j][k] * deltas[L + 1][j];
            }
            deltas[L][k] = error * dactivation(curr.neurons[k]);
        }
    }
    for (int L = 0; L < length - 1; L++) {
        Layer& curr = layers[L];
        Layer& next = layers[L + 1];
        for (int j = 0; j < next.size; j++) {
            for (int k = 0; k < curr.size; k++) {
                curr.weights[j][k] += learningRate * deltas[L + 1][j] * curr.neurons[k];
            }
            curr.biases[j] += learningRate * deltas[L + 1][j];
        }
    }
    for (int i = 0; i < length; i++) {
        delete[] deltas[i];
    }
    delete[] deltas;
}

Perceptron::Perceptron(Perceptron* p1, Perceptron* p2) {
    learningRate = p1->learningRate;
    length = p1->length;
    EPSILON = EPSILON_;
    NOT_MUTAHION = NOT_MUTAHION_;

    layers = new Layer[length];

    static thread_local std::mt19937 gen((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    for (int i = 0; i < length; i++) {
        int size = p1->layers[i].size;
        int nextSize = p1->layers[i].nextSize;
        layers[i] = Layer(size, nextSize);

        if (nextSize <= 0) continue;

        for (int j = 0; j < nextSize; j++) {
            double alpha = uni01(gen);
            layers[i].biases[j] = alpha * p1->layers[i].biases[j] + (1.0 - alpha) * p2->layers[i].biases[j];
            for (int k = 0; k < size; k++) {
                double alpha_w = uni01(gen);
                layers[i].weights[j][k] = alpha_w * p1->layers[i].weights[j][k] + (1.0 - alpha_w) * p2->layers[i].weights[j][k];
            }
        }
    }
}

void Perceptron::mutate(double sigma, double prob) {
    if (sigma > 0.0) EPSILON = sigma;
    if (prob >= 0.0) NOT_MUTAHION = prob;

    static thread_local std::mt19937 gen((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
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
                    layers[l].weights[j][k] += gauss(gen);
                    layers[l].weights[j][k] = std::clamp(layers[l].weights[j][k], WEIGHT_MIN, WEIGHT_MAX);
                }
            }
        }
    }
}

void Perceptron::adaptMutationGlobals(bool improved, int stagnation, double min_eps, double max_eps, double min_prob, double max_prob) {
    const double DECAY = 0.92;   
    const double GROW = 1.08;    

    if (improved) {
        EPSILON_ = std::max(min_eps, EPSILON_ * DECAY);
        NOT_MUTAHION_ = std::max(min_prob, NOT_MUTAHION_ * DECAY);
    }
    else {

        double factor = std::pow(GROW, std::min(stagnation, 20));
        EPSILON_ = std::min(max_eps, EPSILON_ * factor);
        NOT_MUTAHION_ = std::min(max_prob, NOT_MUTAHION_ * factor);
    }

    if (!std::isfinite(EPSILON_)) EPSILON_ = min_eps;
    if (!std::isfinite(NOT_MUTAHION_)) NOT_MUTAHION_ = min_prob;
}

double Perceptron::getEpsilon() {
    return EPSILON;
}

double Perceptron::getNotMutation() {
    return NOT_MUTAHION;
}

int Perceptron::getOut() {
    int maxi = 0;
    for (int i = 1; i < layers[length - 1].size; i++) {
        if (layers[length - 1].neurons[maxi] < layers[length - 1].neurons[i]) {
            maxi = i;
        }
    }
    return maxi;
}

void Perceptron::feedForward(double* inputs) {
    copyArray(layers[0].size, inputs, layers[0].neurons);

    for (int L = 1; L < length; L++) {
        for (int j = 0; j < layers[L].size; j++) {
            double z = solveZ(L, j);
            layers[L].neurons[j] = activation(z);
        }
    }
}

bool Perceptron::isInitialized() {
    if (layers == nullptr) {
        return false;
    }
    return true;
}


Perceptron::~Perceptron() {
    deInit();
}

void Perceptron::deInit() {
    if (layers) {
        for (int i = 0; i < length; i++) {
            layers[i].deInit();
        }
        delete[] layers;
        layers = nullptr;
    }
}

void Perceptron::readFromFile(std::ifstream* fin) {
    if (!fin->is_open()) {
        std::cerr << "Ошибка открытия файла" << std::endl;
        return;
    }
    this->deInit();
    *fin >> learningRate >> length;
    layers = new Layer[length];
    for (int i = 0; i < length; i++) {
        layers[i].readFromFile(fin);
    }
}

void Perceptron::writeInFile(std::ofstream* fout) {
    *fout << learningRate << ' ' << length << '\n';
    for (int i = 0; i < length; i++) {
        layers[i].writeInFile(fout);
    }
}

void Perceptron::readFromFile(std::string file) {
    std::ifstream fin(file, std::ios::in);
    readFromFile(&fin);
    fin.close();
}

void Perceptron::writeInFile(std::string file) {
    std::ofstream fout(file, std::ios::out);
    writeInFile(&fout);
    fout.close();
}
