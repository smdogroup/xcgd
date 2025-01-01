#include <utility>
#include <vector>

#include "elements/gd_mesh.h"
#include "test_commons.h"

TEST(elements, pstencil_Np_1d_2) {
  constexpr int Np_1d = 2;
  std::vector<std::vector<bool>> pstencil = {{1, 0}, {1, 1}};
  auto pterms = pstencil_to_pterms<Np_1d>(pstencil);

  EXPECT_EQ(pterms.size(), 3);

  EXPECT_EQ(pterms[0].first, 0);
  EXPECT_EQ(pterms[0].second, 0);

  EXPECT_EQ(pterms[1].first, 0);
  EXPECT_EQ(pterms[1].second, 1);

  EXPECT_EQ(pterms[2].first, 1);
  EXPECT_EQ(pterms[2].second, 0);
}

TEST(elements, pstencil_Np_1d_4) {
  constexpr int Np_1d = 4;
  std::vector<std::vector<bool>> pstencil = {
      {0, 0, 1, 0}, {0, 1, 1, 1}, {1, 0, 1, 1}, {1, 1, 0, 0}};
  auto pterms = pstencil_to_pterms<Np_1d>(pstencil);

  EXPECT_EQ(pterms.size(), 9);

  EXPECT_EQ(pterms[0].first, 0);
  EXPECT_EQ(pterms[0].second, 0);

  EXPECT_EQ(pterms[1].first, 0);
  EXPECT_EQ(pterms[1].second, 1);

  EXPECT_EQ(pterms[2].first, 0);
  EXPECT_EQ(pterms[2].second, 2);

  EXPECT_EQ(pterms[3].first, 1);
  EXPECT_EQ(pterms[3].second, 0);

  EXPECT_EQ(pterms[4].first, 1);
  EXPECT_EQ(pterms[4].second, 1);

  EXPECT_EQ(pterms[5].first, 1);
  EXPECT_EQ(pterms[5].second, 2);

  EXPECT_EQ(pterms[6].first, 2);
  EXPECT_EQ(pterms[6].second, 0);

  EXPECT_EQ(pterms[7].first, 2);
  EXPECT_EQ(pterms[7].second, 1);

  EXPECT_EQ(pterms[8].first, 3);
  EXPECT_EQ(pterms[8].second, 0);
}
