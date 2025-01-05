#include "test_commons.h"
#include "utils/argparser.h"
#include "utils/json.h"

TEST(utils, ArgParserPass) {
  ArgParser p;
  p.add_argument<int>("--Np_1d", 2, {2, 3, 4});
  p.add_argument<int>("--nxy", 64);
  p.add_argument<double>("--nitsche_eta", 1e6);
  p.add_argument<std::string>("--prefix", {});
  p.add_argument<std::string>("--instance", "circle", {"circle", "wedge"});

  int argc = 4;
  char arg1[] = "./whatever";
  char arg2[] = "--nxy=128";
  char arg3[] = "--instance=wedge";
  char arg4[] = "--prefix=results";
  char* argv[] = {arg1, arg2, arg3, arg4};

  p.parse_args(argc, argv);

  int Np_1d = p.get<int>("Np_1d");
  int nxy = p.get<int>("nxy");
  double nitsche_eta = p.get<double>("nitsche_eta");
  std::string prefix = p.get<std::string>("prefix");

  EXPECT_EQ(Np_1d, 2);
  EXPECT_EQ(nxy, 128);
  EXPECT_DOUBLE_EQ(nitsche_eta, 1e6);
  EXPECT_EQ(prefix, "results");
}

TEST(utils, ArgParserFail) {
  ArgParser p;
  p.add_argument<int>("--Np_1d", 2, {2, 3, 4});
  p.add_argument<int>("--nxy", 64);
  p.add_argument<double>("--nitsche_eta", 1e6);
  p.add_argument<std::string>("--prefix", {});
  p.add_argument<std::string>("--instance", "circle", {"circle", "wedge"});

  int argc = 4;
  char arg1[] = "./whatever";
  char arg2[] = "--nxy=128";
  char arg3[] = "--instance=ball";
  char arg4[] = "--prefix=results";
  char* argv[] = {arg1, arg2, arg3, arg4};

  EXPECT_THROW(
      {
        try {
          p.parse_args(argc, argv);
        } catch (const std::runtime_error& e) {
          EXPECT_STREQ(e.what(),
                       "argument instance has invalid choice: ball (choose "
                       "from circle, wedge)");
          throw;
        }
      },
      std::runtime_error);
}

TEST(utils, CfgParser) {
  ConfigParser parser("test_parser.cfg");

  EXPECT_EQ(parser.get_str_option("string_key"), "hello/xcgd");
  EXPECT_EQ(parser.get_str_option("foo"), "bar");
  EXPECT_EQ(parser.get_int_option("int_key"), 5);
  EXPECT_DOUBLE_EQ(parser.get_double_option("num_key"), 4.2);
  EXPECT_DOUBLE_EQ(parser.get_double_option("num_key_2"), 100000.0);
  EXPECT_EQ(parser.get_num_options(), 10);
  EXPECT_EQ(parser.get_bool_option("is_true"), true);
  EXPECT_EQ(parser.get_bool_option("is_true_too"), true);
  EXPECT_EQ(parser.get_bool_option("is_false"), false);
  EXPECT_EQ(parser.get_bool_option("is_false_too"), false);
  EXPECT_DEATH({ parser.get_bool_option("wrong_bool"); },
               "invalid value .* for a boolean option .*");

  // Save cfg to json
  json j;
  j["cfg"] = parser.get_options();
  write_json("test_parser_out.json", j);
  EXPECT_EQ(read_json("test_parser_out.json"),
            read_json("test_parser.cfg.json"));
}

TEST(utils, CfgParserFromMap) {
  json j = read_json("test_parser.cfg.json");
  ConfigParser parser;
  parser.set_options(j["cfg"]);

  EXPECT_EQ(parser.get_str_option("string_key"), "hello/xcgd");
  EXPECT_EQ(parser.get_str_option("foo"), "bar");
  EXPECT_EQ(parser.get_int_option("int_key"), 5);
  EXPECT_DOUBLE_EQ(parser.get_double_option("num_key"), 4.2);
  EXPECT_DOUBLE_EQ(parser.get_double_option("num_key_2"), 100000.0);
  EXPECT_EQ(parser.get_num_options(), 10);
  EXPECT_EQ(parser.get_bool_option("is_true"), true);
  EXPECT_EQ(parser.get_bool_option("is_true_too"), true);
  EXPECT_EQ(parser.get_bool_option("is_false"), false);
  EXPECT_EQ(parser.get_bool_option("is_false_too"), false);
  EXPECT_DEATH({ parser.get_bool_option("wrong_bool"); },
               "invalid value .* for a boolean option .*");
}
