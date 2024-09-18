#include <fstream>
#include <vector>

#include "test_commons.h"
#include "utils/json.h"

TEST(utils, JSONSerialDeserial) {
  json j = {{"pi", 3.141},
            {"happy", true},
            {"name", "Niels"},
            {"nothing", nullptr},
            {"answer", {{"everything", 42}}},
            {"list", {1, 0, 2}},
            {"object", {{"currency", "USD"}, {"value", 42.99}}},
            {"vec", std::vector<double>{42.1, 42.2, 42.3}}};

  write_json("test.json", j);
  json j2 = read_json("test.json");
  EXPECT_TRUE(j == j2);
}
