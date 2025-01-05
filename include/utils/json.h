#pragma once

#include <fstream>
#include <string>

#include "../../external/json.hpp"

using json = nlohmann::json;

inline void write_json(std::string json_path, const json& json_obj) {
  std::ofstream o(json_path);
  o << std::setw(4) << json_obj << std::endl;
};

inline json read_json(std::string json_path) {
  std::ifstream i(json_path);
  json j;
  i >> j;
  return j;
};
