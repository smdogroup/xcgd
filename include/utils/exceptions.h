#pragma once

#include <exception>
#include <string>

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

class NotImplemented : public std::exception {
 public:
  NotImplemented(const std::string &msg) : msg(msg) {}

  const char *what() const noexcept { return msg.c_str(); }

 private:
  std::string msg;
};
