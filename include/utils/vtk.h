#ifndef XCGD_VTK_H
#define XCGD_VTK_H

#include <complex>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

// Linear and nonlinear cell types in VTK
struct VTKID {
  static constexpr int VERTEX = 1;
  static constexpr int POLY_VERTEX = 2;
  static constexpr int LINE = 3;
  static constexpr int POLY_LINE = 4;
  static constexpr int TRIANGLE = 5;
  static constexpr int TRIANGLE_STRIP = 6;
  static constexpr int POLYGON = 7;
  static constexpr int PIXEL = 8;
  static constexpr int QUAD = 9;
  static constexpr int TETRA = 10;
  static constexpr int VOXEL = 11;
  static constexpr int HEXAHEDRON = 12;
  static constexpr int WEDGE = 13;
  static constexpr int PYRAMID = 14;
  static constexpr int QUADRATIC_EDGE = 21;
  static constexpr int QUADRATIC_TRIANGLE = 22;
  static constexpr int QUADRATIC_QUAD = 23;
  static constexpr int QUADRATIC_TETRA = 24;
  static constexpr int QUADRATIC_HEXAHEDRON = 25;
};

struct VTK_NVERTS {
  static constexpr int VERTEX = 1;
  static constexpr int LINE = 2;
  static constexpr int TRIANGLE = 3;
  static constexpr int PIXEL = 4;
  static constexpr int QUAD = 4;
  static constexpr int TETRA = 4;
  static constexpr int VOXEL = 8;
  static constexpr int HEXAHEDRON = 8;
  static constexpr int WEDGE = 6;
  static constexpr int PYRAMID = 5;
  static constexpr int QUADRATIC_EDGE = 3;
  static constexpr int QUADRATIC_TRIANGLE = 6;
  static constexpr int QUADRATIC_QUAD = 8;
  static constexpr int QUADRATIC_TETRA = 10;
  static constexpr int QUADRATIC_HEXAHEDRON = 20;
};

void write_real_val(std::FILE* fp_, double val) {
  std::fprintf(fp_, "%-20.15f", val);
}

void write_real_val(std::FILE* fp_, std::complex<double> val) {
  std::fprintf(fp_, "%-20.15f", val.real());
}

// VTK writer for 2d and 3d mesh
template <typename T>
class ToVTK {
 public:
  ToVTK(const int spatial_dim, const int num_nodes, const int num_elements,
        const int nnodes_per_elem, const int* element_nodes, const T* xloc,
        const int vtk_elem_type = -1, const std::string vtk_name = "result.vtk")
      : spatial_dim_(spatial_dim),
        num_nodes_(num_nodes),
        num_elements_(num_elements),
        nnodes_per_elem_(nnodes_per_elem),
        element_nodes_(element_nodes),
        xloc_(xloc),
        vtk_elem_type_(vtk_elem_type) {
    // Open file and destroy old contents
    fp_ = std::fopen(vtk_name.c_str(), "w+");

    if (spatial_dim_ != 2 and spatial_dim_ != 3) {
      char msg[256];
      std::snprintf(msg, sizeof(msg),
                    "Invalid spatial_dim, got %d, expect 2 or 3", spatial_dim_);
      throw std::runtime_error(msg);
    }

    // If not provided, infer vtk type id from nnodes_per_elem
    if (vtk_elem_type_ == -1) {
      if (spatial_dim_ == 2) {
        switch (nnodes_per_elem_) {
          case 3:
            vtk_elem_type_ = VTKID::TRIANGLE;
            break;
          case 4:
            vtk_elem_type_ = VTKID::QUAD;
            break;
          case 6:
            vtk_elem_type_ = VTKID::QUADRATIC_TRIANGLE;
            break;
          case 8:
            vtk_elem_type_ = VTKID::QUADRATIC_QUAD;
            break;
        }
      } else {  // spatial_dim == 3
        switch (nnodes_per_elem_) {
          case 4:
            vtk_elem_type_ = VTKID::TETRA;
            break;
          case 5:
            vtk_elem_type_ = VTKID::PYRAMID;
            break;
          case 6:
            vtk_elem_type_ = VTKID::WEDGE;
            break;
          case 8:
            vtk_elem_type_ = VTKID::HEXAHEDRON;
            break;
          case 10:
            vtk_elem_type_ = VTKID::QUADRATIC_TETRA;
            break;
          case 20:
            vtk_elem_type_ = VTKID::QUADRATIC_HEXAHEDRON;
            break;
        }
      }
    }

    if (vtk_elem_type_ == -1) {
      char msg[256];
      snprintf(msg, 256,
               "Cannot infer element type from nnodes_per_elem(%d) for a "
               "%d-dimensional mesh",
               nnodes_per_elem_, spatial_dim_);
      throw std::runtime_error(msg);
    }
  }

  ~ToVTK() {
    // Close file
    std::fclose(fp_);
  }

  void write_mesh() {
    // Write header
    std::fprintf(fp_, "# vtk DataFile Version 3.0\n");
    std::fprintf(fp_, "my example\n");
    std::fprintf(fp_, "ASCII\n");
    std::fprintf(fp_, "DATASET UNSTRUCTURED_GRID\n");

    // Write nodes
    std::fprintf(fp_, "POINTS %d double\n", num_nodes_);
    for (int i = 0; i < num_nodes_; i++) {
      if (spatial_dim_ == 2) {
        write_real_val(fp_, xloc_[2 * i]);
        write_real_val(fp_, xloc_[2 * i + 1]);
        write_real_val(fp_, 0.0);
        std::fprintf(fp_, "\n");
      } else {
        write_real_val(fp_, xloc_[2 * i]);
        write_real_val(fp_, xloc_[2 * i + 1]);
        write_real_val(fp_, xloc_[2 * i + 2]);
        std::fprintf(fp_, "\n");
      }
    }

    // Write connectivity
    std::fprintf(fp_, "CELLS %d %d\n", num_elements_,
                 num_elements_ * (1 + nnodes_per_elem_));
    for (int i = 0; i < num_elements_; i++) {
      std::fprintf(fp_, "%d ", nnodes_per_elem_);
      for (int j = 0; j < nnodes_per_elem_; j++) {
        std::fprintf(fp_, "%d ", element_nodes_[nnodes_per_elem_ * i + j]);
      }
      std::fprintf(fp_, "\n");
    }

    // Write cell type
    std::fprintf(fp_, "CELL_TYPES %d\n", num_elements_);
    for (int i = 0; i < num_elements_; i++) {
      std::fprintf(fp_, "%d\n", vtk_elem_type_);
    }
  }

  void write_sol(const std::string sol_name, const T* sol_vec) {
    // Write header
    if (!vtk_has_sol_header_) {
      std::fprintf(fp_, "POINT_DATA %d\n", num_nodes_);
      vtk_has_sol_header_ = true;
    }
    std::fprintf(fp_, "SCALARS %s double 1\n", sol_name.c_str());
    std::fprintf(fp_, "LOOKUP_TABLE default\n");

    // Write data
    for (int i = 0; i < num_nodes_; i++) {
      write_real_val(fp_, sol_vec[i]);
      std::fprintf(fp_, "\n");
    }
  }

 private:
  int spatial_dim_;
  int num_nodes_;
  int num_elements_;
  int nnodes_per_elem_;
  const int* element_nodes_;
  const T* xloc_;
  int vtk_elem_type_;
  std::FILE* fp_ = nullptr;
  bool vtk_has_sol_header_ = false;
};

#endif  // XCGD_VTK_H