#ifndef XCGD_PARSER_H
#define XCGD_PARSER_H

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

class ConfigParser {
 public:
  ConfigParser(std::string cfg_path) {
    char abs_path[PATH_MAX];
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
    return cfg.at(key);
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

 private:
  void key_not_found(std::string key) const {
    std::fprintf(stderr, "key %s not found in config file %s\n", key.c_str(),
                 abs_path);
    exit(-1);
  }

  std::map<std::string, std::string> cfg;
  char abs_path[PATH_MAX];
};

#endif  // XCGD_PARSER_H