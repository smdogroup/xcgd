#pragma once

#include <exception>

class StencilConstructionFailed : public std::exception {
 public:
  StencilConstructionFailed(int elem) : elem(elem) {}

  const char *what() const noexcept {
    return "Stencil construction failed: not enough DOF nodes found.";
  }

  inline int get_elem_index() const { return elem; }

 private:
  int elem;
};
