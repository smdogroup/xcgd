#pragma once

// Put this in class definition to test private methods using GTest
#define FRIEND_TEST(test_case_name, test_name) \
  friend class test_case_name##_##test_name##_Test
