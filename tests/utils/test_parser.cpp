#include "test_commons.h"
#include "utils/parser.h"

TEST(utils, CfgParser) {
  ConfigParser parser("test_parser.cfg");

  EXPECT_EQ(parser.get_str_option("string_key"), "hello/xcgd");
  EXPECT_EQ(parser.get_str_option("foo"), "bar");
  EXPECT_EQ(parser.get_int_option("int_key"), 5);
  EXPECT_DOUBLE_EQ(parser.get_double_option("num_key"), 4.2);
  EXPECT_EQ(parser.get_num_options(), 4);
}
