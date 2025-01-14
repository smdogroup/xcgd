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

class LapackFailed : public std::exception {
 public:
  LapackFailed(std::string rountine, int exit_code) {
    std::snprintf(msg, 256, "Lapack rountine %s failed with exit code %d",
                  rountine.c_str(), exit_code);
  }
  const char *what() const noexcept { return msg; }

 private:
  char msg[256];
};

class NotImplemented : public std::exception {
 public:
  NotImplemented(const std::string &msg) : msg(msg) {}

  const char *what() const noexcept { return msg.c_str(); }

 private:
  std::string msg;
};
