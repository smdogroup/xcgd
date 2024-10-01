#include <gtest/gtest.h>

#include <fstream>
#include <vector>

#include "test_commons.h"
#include "utils/misc.h"

template <int N>
int foo() {
  return N * 3;
}

TEST(utils, MiscSwitcher) {
  constexpr int Max = 3;
  int ret;
  auto f = [&ret]<int N>() { ret = foo<N>(); };

  switcher<Max>::run(f, 0);
  EXPECT_EQ(ret, 0);

  switcher<Max>::run(f, 1);
  EXPECT_EQ(ret, 3);

  switcher<Max>::run(f, 2);
  EXPECT_EQ(ret, 6);

  EXPECT_THROW(
      {
        try {
          switcher<Max>::run(f, -1);
        } catch (const std::runtime_error& e) {
          EXPECT_STREQ("only positive runtime values are supported, got -1",
                       e.what());
          throw;
        }
      },
      std::runtime_error);

  EXPECT_THROW(
      {
        try {
          switcher<Max>::run(f, 4);
        } catch (const std::runtime_error& e) {
          EXPECT_STREQ(
              "runtime value 4 exceeds the maximum pre-compiled value 3, if "
              "this "
              "is intended, change the Max template argument for the switch in "
              "source code",
              e.what());
          throw;
        }
      },
      std::runtime_error);
}
