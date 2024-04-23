#include "test_commons.h"
#include "utils/parser.h"

TEST(utils, CfgParser) {
  ConfigParser parser("test_parser.cfg");

  EXPECT_EQ(parser.get_str_option("string_key"), "hello/xcgd");
  EXPECT_EQ(parser.get_str_option("foo"), "bar");
  EXPECT_EQ(parser.get_int_option("int_key"), 5);
  EXPECT_DOUBLE_EQ(parser.get_double_option("num_key"), 4.2);
  EXPECT_EQ(parser.get_num_options(), 9);
  EXPECT_EQ(parser.get_bool_option("is_true"), true);
  EXPECT_EQ(parser.get_bool_option("is_true_too"), true);
  EXPECT_EQ(parser.get_bool_option("is_false"), false);
  EXPECT_EQ(parser.get_bool_option("is_false_too"), false);
  EXPECT_DEATH({ parser.get_bool_option("wrong_bool"); },
               "invalid value .* for a boolean option .*");
}
