#pragma once

#include <map>
#include <stdexcept>
#include <vector>

class DegenerateStencilLogger {
 public:
  static void enable() { active = true; }
  static void disable() { active = false; }

  static void add(int elem, int nnodes, int* nodes) {
    if (!active) return;
    stencils[elem] = std::vector<int>(nodes, nodes + nnodes);
  }

  static void clear() {
    if (!active) return;
    stencils.clear();
  }

  const static std::map<int, std::vector<int>>& get_stencils() {
    return stencils;
  }

 private:
  inline static bool active = false;

  // elem -> stencil node indices
  inline static std::map<int, std::vector<int>> stencils = {};

  // stencil polynomial order along xyz dimension
  inline static std::map<int, std::tuple<int, int, int>> stencil_orders = {};
};

class VandermondeCondLogger {
 public:
  static void enable() { active = true; }
  static void disable() { active = false; }

  static void add(int elem, double cond) {
    if (!active) return;
    conds[elem] = cond;
  }

  static void clear() {
    if (!active) return;
    conds.clear();
  }

  // Get condition number of elem, if elem does not exist, quitely return NaN
  static double get(int elem) {
    try {
      return conds.at(elem);
    } catch (const std::out_of_range& e) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }
  const static std::map<int, double>& get_conds() { return conds; }

 private:
  inline static bool active = false;

  // elem -> stencil node indices
  inline static std::map<int, double> conds = {};
};
