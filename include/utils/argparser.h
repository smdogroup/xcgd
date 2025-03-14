#pragma once

#include <algorithm>
#include <any>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>

#include "utils/json.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

inline std::map<std::string, std::string> get_cmd_args(int argc, char* argv[]) {
  std::map<std::string, std::string> keyvals;
  for (int i = 0; i < argc - 1; i++) {
    std::string entry(argv[i + 1]);
    auto e = entry.find('=');
    std::string key, val;
    key = entry.substr(0, e);
    if (e != std::string::npos) {
      keyvals[key] = entry.substr(e + 1, entry.size() - e - 1);
    } else {
      keyvals[key] = {};
    }
  }
  return keyvals;
}

/**
 * @brief A very simple command-line argument parser
 *
 * Supported types of arguments: int, double, string
 */
class ArgParser {
 private:
  using arg_t = std::variant<int, double, std::string>;

  // Usage: isValidArgType<T>::value == true if T is a valid argument type
  template <typename T, typename VARIANT_T = arg_t>
  struct isValidArgType;
  template <typename T, typename... ALL_T>
  struct isValidArgType<T, std::variant<ALL_T...>>
      : public std::disjunction<std::is_same<T, ALL_T>...> {};

 public:
  template <typename Type>
  void add_argument(std::string name, Type default_val,
                    std::set<Type> choices = {}) {
    static_assert(isValidArgType<Type>::value,
                  "attempted argument type is not supported");

    name = strip_leading_hyphens(name);
    if (name.empty()) return;

    args[name] = default_val;
    default_strings[name] = option_to_string(default_val);
    std::set<std::string> choice_strs;
    for (auto c : choices) {
      choice_strs.insert(option_to_string(c));
    }
    choice_strings[name] = choice_strs;
    keys.push_back(name);
  }

  void write_args_to_file(std::string path) {
    auto folder = std::filesystem::path(path).parent_path();
    if (!std::filesystem::exists(folder)) {
      std::filesystem::create_directories(folder);
    }

    FILE* fp = std::fopen(path.c_str(), "w");

    for (std::string k : keys) {
      arg_t v = args[k];

      if (std::holds_alternative<int>(v)) {
        std::fprintf(fp, "--%s=%d ", k.c_str(), std::get<int>(v));
      } else if (std::holds_alternative<double>(v)) {
        std::fprintf(fp, "--%s=%.10f ", k.c_str(), std::get<double>(v));
      } else if (std::holds_alternative<std::string>(v)) {
        std::fprintf(fp, "--%s=%s ", k.c_str(),
                     std::get<std::string>(v).c_str());
      }
    }
    std::fprintf(fp, "\n");
    std::fclose(fp);
  }

  template <typename Type>
  Type get(std::string key) {
    static_assert(isValidArgType<Type>::value,
                  "attempted argument type is not supported");
    if (!args.count(key)) {
      std::fprintf(stderr, "[Error] unknown argument %s\n", key.c_str());
      return {};
    }
    return std::get<Type>(args[key]);
  }

  void parse_args(int argc, char* argv[]) {
    std::map<std::string, std::string> cmd_args = get_cmd_args(argc, argv);

    if (cmd_args.count("--help") or cmd_args.count("-h") or
        cmd_args.count("--h")) {
      show_help_info_and_exit();
    }

    for (auto kv : cmd_args) {
      std::string k = strip_leading_hyphens(kv.first);
      std::string v = kv.second;

      // Check if the option is restricted to certain choices
      if (choice_strings[k].size() > 0) {
        if (!choice_strings[k].count(v)) {
          std::string msg =
              "argument " + k + " has invalid choice: " + v + " (choose from";
          for (auto c : choice_strings[k]) {
            msg += " " + c + ",";
          }
          msg.pop_back();  // Remove the last comma
          msg += ")";
          throw std::runtime_error(msg.c_str());
        }
      }

      if (!args.count(k)) {
        std::printf("[Warning] unknown cmd argument %s\n", k.c_str());
        continue;
      }

      if (std::holds_alternative<int>(args[k])) {
        args[k] = std::stoi(v);
      } else if (std::holds_alternative<double>(args[k])) {
        args[k] = std::stod(v);
      } else if (std::holds_alternative<std::string>(args[k])) {
        args[k] = v;
      }
    }
  }

 private:
  void show_help_info_and_exit() {
    std::printf("Usage: ./prog_name [-h,--help]");
    for (std::string k : keys) {
      if (choice_strings[k].size() > 0) {
        std::string msg = " --" + k + "=[";
        for (auto c : choice_strings[k]) {
          msg += c + ",";
        }
        msg.pop_back();
        msg += "]";
        std::cout << msg;
      } else {
        std::cout << " --" << k << "=(default:" << default_strings[k] << ")";
      }
    }
    std::printf("\n");
    exit(0);
  }

  std::string strip_leading_hyphens(std::string name) {
    name.erase(0, std::min(name.find_first_not_of('-'), name.size() - 1));
    return name;
  }

  template <class Type>
  std::string option_to_string(Type val) {
    return std::to_string(val);
  }

  std::string option_to_string(std::string val) { return val; }

  std::map<std::string, arg_t> args;
  std::map<std::string, std::string> default_strings;
  std::map<std::string, std::set<std::string>> choice_strings;
  std::vector<std::string> keys;
};

class ConfigParser {
 public:
  ConfigParser() = default;  // Construct an empty parser, options can be set
                             // via set_options() later

  ConfigParser(std::string cfg_path) {
    realpath(cfg_path.c_str(), abs_path);
    bool fail = false;
    std::string line;
    std::ifstream infile(cfg_path);
    int lno = 0;
    while (std::getline(infile, line)) {
      lno++;
      // skip empty lines and comments
      if (line.length() == 0 or line[0] == '#') continue;

      std::string key, eq, val;
      std::istringstream(line) >> key >> eq >> val;

      if (eq != "=") {
        fail = true;
        std::fprintf(stderr, "%s:%d: unknown token %s, = expected\n", abs_path,
                     lno, eq.c_str());
      }

      else if (val.empty()) {
        fail = true;
        std::fprintf(stderr, "%s:%d: key %s missing value\n", abs_path, lno,
                     key.c_str());
      }

      else {
        cfg[key] = val;
      }
    }

    if (fail) {
      exit(-1);
    }
  }

  void set_options(const json& kvpairs_json) {
    for (const auto& item : kvpairs_json.items()) {
      cfg[item.key()] = item.value();
    }
  }

  int get_int_option(std::string key) const {
    if (cfg.count(key) == 0) {
      key_not_found(key);
    }
    return std::stoi(cfg.at(key));
  }
  double get_double_option(std::string key) const {
    if (cfg.count(key) == 0) {
      key_not_found(key);
    }
    return std::stod(cfg.at(key));
  }
  std::string get_str_option(std::string key) const {
    if (cfg.count(key) == 0) {
      key_not_found(key);
    }
    std::string str = cfg.at(key);
    // Strip single and double quotes, if any
    if (str.size() >= 2 && str.front() == '"' && str.back() == '"') {
      str = str.substr(1, str.size() - 2);  // Remove the quotes
    }
    return str;
  }
  bool get_bool_option(std::string key) const {
    if (cfg.count(key) == 0) {
      key_not_found(key);
    }
    std::string val = cfg.at(key);
    std::transform(val.begin(), val.end(), val.begin(), ::tolower);
    if (val == "true") {
      return true;
    } else if (val == "false") {
      return false;
    } else {
      std::fprintf(stderr, "invalid value %s for a boolean option %s\n",
                   cfg.at(key).c_str(), key.c_str());
      exit(-1);
    }
  }

  int get_num_options() const { return cfg.size(); }
  const auto& get_options() const { return cfg; }

 private:
  void key_not_found(std::string key) const {
    std::fprintf(stderr, "key %s not found in config file %s\n", key.c_str(),
                 abs_path);
    exit(-1);
  }

  std::map<std::string, std::string> cfg;
  char abs_path[PATH_MAX];
};
