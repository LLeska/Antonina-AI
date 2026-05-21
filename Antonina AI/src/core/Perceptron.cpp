#include "Perceptron.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

Perceptron::Perceptron() {
  EPSILON = 0.1;
  NOT_MUTAHION = 0.04;
  learningRate = 0;
  length = 0;
  layers = nullptr;
}

Perceptron::Perceptron(double learningRate_, int length_, int *sizes) {
  EPSILON = 0.1;
  NOT_MUTAHION = 0.04;
  learningRate = learningRate_;
  length = length_;
  layers = new Layer[length];

  static thread_local std::mt19937 gen(
      (unsigned)std::chrono::high_resolution_clock::now()
          .time_since_epoch()
          .count());
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  for (int l = 0; l < length; l++) {
    int nextSize = (l < length - 1) ? sizes[l + 1] : 0;
    layers[l] = Layer(sizes[l], nextSize);

    if (nextSize > 0 && layers[l].biases && layers[l].weights) {
      double range = sqrt(6.0 / (sizes[l] + nextSize));
      for (int j = 0; j < nextSize; j++) {
        layers[l].biases[j] = 0.0;
        for (int k = 0; k < sizes[l]; k++)
          layers[l].weights[j * sizes[l] + k] = dist(gen) * range;
      }
    }
  }
}

Perceptron::Perceptron(const Perceptron &p) {
  EPSILON = p.EPSILON;
  NOT_MUTAHION = p.NOT_MUTAHION;
  learningRate = p.learningRate;
  length = p.length;
  if (p.layers && length > 0) {
    layers = new Layer[length];
    for (int i = 0; i < length; i++)
      layers[i] = p.layers[i];
  } else {
    layers = nullptr;
  }
}

Perceptron::Perceptron(Perceptron &&p) {
  EPSILON = p.EPSILON;
  NOT_MUTAHION = p.NOT_MUTAHION;
  learningRate = p.learningRate;
  length = p.length;
  layers = p.layers;
  p.layers = nullptr;
  p.length = 0;
}

Perceptron::Perceptron(Perceptron *p1, Perceptron *p2) {
  EPSILON = 0.1;
  NOT_MUTAHION = 0.04;
  learningRate = p1->learningRate;
  length = p1->length;
  layers = new Layer[length];

  static thread_local std::mt19937 gen(
      (unsigned)std::chrono::high_resolution_clock::now()
          .time_since_epoch()
          .count());
  std::uniform_real_distribution<double> uni01(0.0, 1.0);

  for (int i = 0; i < length; i++) {
    int size = p1->layers[i].size;
    int nextSize = p1->layers[i].nextSize;
    layers[i] = Layer(size, nextSize);
    if (nextSize <= 0)
      continue;

    for (int j = 0; j < nextSize; j++) {
      double alpha = uni01(gen);
      layers[i].biases[j] = alpha * p1->layers[i].biases[j] +
                            (1.0 - alpha) * p2->layers[i].biases[j];
      for (int k = 0; k < size; k++) {
        double aw = uni01(gen);
        layers[i].weights[j * size + k] =
            aw * p1->layers[i].weights[j * size + k] +
            (1.0 - aw) * p2->layers[i].weights[j * size + k];
      }
    }
  }
}

Perceptron &Perceptron::operator=(const Perceptron &p) {
  if (this == &p)
    return *this;
  deInit();
  EPSILON = p.EPSILON;
  NOT_MUTAHION = p.NOT_MUTAHION;
  learningRate = p.learningRate;
  length = p.length;
  if (p.layers && length > 0) {
    layers = new Layer[length];
    for (int i = 0; i < length; i++)
      layers[i] = p.layers[i];
  } else {
    layers = nullptr;
  }
  return *this;
}

Perceptron &Perceptron::operator=(Perceptron &&p) {
  if (this != &p) {
    deInit();
    EPSILON = p.EPSILON;
    NOT_MUTAHION = p.NOT_MUTAHION;
    learningRate = p.learningRate;
    length = p.length;
    layers = p.layers;
    p.layers = nullptr;
    p.length = 0;
  }
  return *this;
}

Perceptron::~Perceptron() { deInit(); }

void Perceptron::deInit() {
  if (layers) {
    delete[] layers;
    layers = nullptr;
  }
  length = 0;
}

void Perceptron::feedForward(double *inputs) {
  for (int i = 0; i < layers[0].size; i++)
    layers[0].neurons[i] = inputs[i];

  for (int L = 1; L < length; L++) {
    Layer &prev = layers[L - 1];
    Layer &curr = layers[L];
    for (int j = 0; j < curr.size; j++) {
      double z = prev.biases[j];
      for (int k = 0; k < prev.size; k++)
        z += prev.neurons[k] * prev.weights[j * prev.size + k];
      curr.neurons[j] = 1.0 / (1.0 + exp(-z));
    }
  }
}

void Perceptron::feedForwardBatch(const double *batch_inputs, double *batch_outputs, int B) const {
  if (B <= 0 || layers == nullptr || length <= 0)
    return;

  int max_layer = 0;
  for (int l = 0; l < length; l++)
    if (layers[l].size > max_layer)
      max_layer = layers[l].size;

  static thread_local std::vector<double> buf_a;
  static thread_local std::vector<double> buf_b;
  const size_t plane = (size_t)max_layer * B;
  if (buf_a.size() < plane)
    buf_a.resize(plane);
  if (buf_b.size() < plane)
    buf_b.resize(plane);

  double *x_plane = buf_a.data();
  double *y_plane = buf_b.data();

  const int in_size = layers[0].size;

  for (int b = 0; b < B; b++) {
    const double *src = batch_inputs + (size_t)b * in_size;
    for (int k = 0; k < in_size; k++)
      x_plane[(size_t)k * B + b] = src[k];
  }

  for (int L = 1; L < length; L++) {
    const Layer &prev = layers[L - 1];
    const int prev_sz = prev.size;
    const int curr_sz = layers[L].size;
    const double *W = prev.weights;
    const double *bias = prev.biases;

    for (int j = 0; j < curr_sz; j++) {
      double *y_row = y_plane + (size_t)j * B;
      const double bj = bias[j];

      for (int b = 0; b < B; b++)
        y_row[b] = bj;
      const double *w_j = W + j * prev_sz;

      for (int k = 0; k < prev_sz; k++) {
        const double w_jk = w_j[k];
        const double *x_row = x_plane + (size_t)k * B;
        for (int b = 0; b < B; b++)
          y_row[b] += w_jk * x_row[b];
      }

      for (int b = 0; b < B; b++)
        y_row[b] = 1.0 / (1.0 + exp(-y_row[b]));
    }

    std::swap(x_plane, y_plane);
  }

  const int out_sz = layers[length - 1].size;
  for (int b = 0; b < B; b++) {
    double *dst = batch_outputs + (size_t)b * out_sz;
    for (int j = 0; j < out_sz; j++)
      dst[j] = x_plane[(size_t)j * B + b];
  }
}

int Perceptron::getOut() {
  int maxi = 0;
  for (int i = 1; i < layers[length - 1].size; i++)
    if (layers[length - 1].neurons[maxi] < layers[length - 1].neurons[i])
      maxi = i;
  return maxi;
}

double Perceptron::outputValue(int index) const {
  if (!layers || length <= 0 || index < 0 || index >= layers[length - 1].size)
    return 0.0;
  return layers[length - 1].neurons[index];
}

std::unique_ptr<Brain> Perceptron::cloneBrain() const {
  return std::make_unique<Perceptron>(*this);
}

void Perceptron::getOutBatch(const double *batch_outputs, int *out_args, int B) const {
  const int out_sz = layers[length - 1].size;
  for (int b = 0; b < B; b++) {
    const double *o = batch_outputs + (size_t)b * out_sz;
    int maxi = 0;
    for (int j = 1; j < out_sz; j++)
      if (o[maxi] < o[j])
        maxi = j;
    out_args[b] = maxi;
  }
}

void Perceptron::mutate(double sigma, double prob) {
  if (sigma > 0.0)
    EPSILON = sigma;
  if (prob >= 0.0)
    NOT_MUTAHION = prob;

  static thread_local std::mt19937 gen(
      (unsigned)std::chrono::high_resolution_clock::now()
          .time_since_epoch()
          .count());
  std::uniform_real_distribution<double> uni01(0.0, 1.0);
  const double WEIGHT_MIN = -5.0;
  const double WEIGHT_MAX = 5.0;

  for (int l = 0; l < length; l++) {
    int size = layers[l].size;
    int nextSize = layers[l].nextSize;
    if (nextSize <= 0)
      continue;

    const double layer_scale = sqrt(2.0 / (size + nextSize));
    std::normal_distribution<double> gauss(0.0, EPSILON * layer_scale);

    for (int j = 0; j < nextSize; j++) {
      if (uni01(gen) < NOT_MUTAHION) {
        layers[l].biases[j] += gauss(gen);
        layers[l].biases[j] =
            std::clamp(layers[l].biases[j], WEIGHT_MIN, WEIGHT_MAX);
      }
      for (int k = 0; k < size; k++) {
        if (uni01(gen) < NOT_MUTAHION) {
          layers[l].weights[j * size + k] += gauss(gen);
          layers[l].weights[j * size + k] = std::clamp(
              layers[l].weights[j * size + k], WEIGHT_MIN, WEIGHT_MAX);
        }
      }
    }
  }
}

bool Perceptron::isInitialized() { return layers != nullptr; }

int Perceptron::layerSize(int index) const {
  if (!layers || index < 0 || index >= length)
    return 0;
  return layers[index].size;
}

double Perceptron::neuronValue(int layer, int index) const {
  if (!layers || layer < 0 || layer >= length)
    return 0.0;
  const Layer &l = layers[layer];
  if (!l.neurons || index < 0 || index >= l.size)
    return 0.0;
  return l.neurons[index];
}

double Perceptron::connectionWeight(int from_layer, int from, int to) const {
  if (!layers || from_layer < 0 || from_layer + 1 >= length)
    return 0.0;
  const Layer &layer = layers[from_layer];
  if (!layer.weights || from < 0 || from >= layer.size || to < 0 ||
      to >= layer.nextSize)
    return 0.0;
  return layer.weights[to * layer.size + from];
}

void Perceptron::readFromFile(std::ifstream *fin) {
  if (!fin->is_open() || fin->fail()) {
    std::cerr << "Stream error before reading Perceptron" << std::endl;
    return;
  }
  deInit();
  *fin >> learningRate >> length;
  if (fin->fail() || length <= 0 || length > 10000) {
    std::cerr << "Invalid length=" << length << std::endl;
    length = 0;
    return;
  }
  layers = new Layer[length];
  for (int i = 0; i < length; i++) {
    layers[i].readFromFile(fin);
    if (fin->fail()) {
      std::cerr << "Error reading Layer " << i << std::endl;
      break;
    }
  }
}

void Perceptron::writeInFile(std::ofstream *fout) {
  *fout << learningRate << ' ' << length << '\n';
  for (int i = 0; i < length; i++)
    layers[i].writeInFile(fout);
}

void Perceptron::readFromFile(std::string file) {
  std::ifstream fin(file);
  readFromFile(&fin);
}

void Perceptron::writeInFile(std::string file) {
  std::ofstream fout(file);
  writeInFile(&fout);
}
