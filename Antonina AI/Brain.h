#pragma once

#include <memory>

class Brain {
public:
  virtual ~Brain() = default;
  virtual void feedForward(double *inputs) = 0;
  virtual int getOut() = 0;
  virtual std::unique_ptr<Brain> cloneBrain() const = 0;
};
