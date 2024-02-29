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

void write_real_val(std::FILE* fp, double val) {
  std::fprintf(fp, "%-20.15f", val);
}

void write_real_val(std::FILE* fp, std::complex<double> val) {
  std::fprintf(fp, "%-20.15f", val.real());
}

// VTK writer for 2d and 3d mesh
template <typename T, class Mesh>
class ToVTK {
 private:
  static constexpr int spatial_dim = Mesh::spatial_dim;

 public:
  ToVTK(const Mesh& mesh, const std::string vtk_name = "result.vtk",
        int vtk_elem_type_ = -1)
      : mesh(mesh), vtk_elem_type(vtk_elem_type_) {
    // Open file and destroy old contents
    fp = std::fopen(vtk_name.c_str(), "w+");

    if (spatial_dim != 2 and spatial_dim != 3) {
      char msg[256];
      std::snprintf(msg, sizeof(msg),
                    "Invalid spatial_dim, got %d, expect 2 or 3", spatial_dim);
      throw std::runtime_error(msg);
    }

    // If not provided, infer vtk type id from mesh.verts_per_element
    if (vtk_elem_type == -1) {
      if constexpr (spatial_dim == 2) {
        switch (mesh.verts_per_element) {
          case 3:
            vtk_elem_type = VTKID::TRIANGLE;
            break;
          case 4:
            vtk_elem_type = VTKID::QUAD;
            break;
          case 6:
            vtk_elem_type = VTKID::QUADRATIC_TRIANGLE;
            break;
          case 8:
            vtk_elem_type = VTKID::QUADRATIC_QUAD;
            break;
        }
      } else {  // spatial_dim == 3
        switch (mesh.verts_per_element) {
          case 4:
            vtk_elem_type = VTKID::TETRA;
            break;
          case 5:
            vtk_elem_type = VTKID::PYRAMID;
            break;
          case 6:
            vtk_elem_type = VTKID::WEDGE;
            break;
          case 8:
            vtk_elem_type = VTKID::HEXAHEDRON;
            break;
          case 10:
            vtk_elem_type = VTKID::QUADRATIC_TETRA;
            break;
          case 20:
            vtk_elem_type = VTKID::QUADRATIC_HEXAHEDRON;
            break;
        }
      }
    }

    if (vtk_elem_type == -1) {
      char msg[256];
      snprintf(msg, 256,
               "Cannot infer element type from verts_per_element(%d) for a "
               "%d-dimensional mesh",
               mesh.verts_per_element, spatial_dim);
      throw std::runtime_error(msg);
    }
  }

  ~ToVTK() {
    // Close file
    std::fclose(fp);
  }

  void write_mesh() const {
    // Write header
    std::fprintf(fp, "# vtk DataFile Version 3.0\n");
    std::fprintf(fp, "my example\n");
    std::fprintf(fp, "ASCII\n");
    std::fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");

    // Write nodes
    std::fprintf(fp, "POINTS %d double\n", mesh.get_num_nodes());
    for (int i = 0; i < mesh.get_num_nodes(); i++) {
      T xloc[spatial_dim];
      mesh.get_node_xloc(i, xloc);
      if constexpr (spatial_dim == 2) {
        write_real_val(fp, xloc[0]);
        write_real_val(fp, xloc[1]);
        write_real_val(fp, 0.0);
        std::fprintf(fp, "\n");
      } else {
        write_real_val(fp, xloc[0]);
        write_real_val(fp, xloc[1]);
        write_real_val(fp, xloc[2]);
        std::fprintf(fp, "\n");
      }
    }

    // Write connectivity
    std::fprintf(fp, "CELLS %d %d\n", mesh.get_num_elements(),
                 mesh.get_num_elements() * (1 + mesh.verts_per_element));
    for (int i = 0; i < mesh.get_num_elements(); i++) {
      std::fprintf(fp, "%d ", mesh.verts_per_element);
      int verts[mesh.verts_per_element];
      mesh.get_elem_dof_verts(i, verts);
      for (int j = 0; j < mesh.verts_per_element; j++) {
        std::fprintf(fp, "%d ", verts[j]);
      }
      std::fprintf(fp, "\n");
    }

    // Write cell type
    std::fprintf(fp, "CELL_TYPES %d\n", mesh.get_num_elements());
    for (int i = 0; i < mesh.get_num_elements(); i++) {
      std::fprintf(fp, "%d\n", vtk_elem_type);
    }
  }

  void write_sol(const std::string sol_name, const T* sol_vec) {
    // Write header
    if (!vtk_has_sol_header) {
      std::fprintf(fp, "POINT_DATA %d\n", mesh.get_num_nodes());
      vtk_has_sol_header = true;
    }
    std::fprintf(fp, "SCALARS %s double 1\n", sol_name.c_str());
    std::fprintf(fp, "LOOKUP_TABLE default\n");

    // Write data
    for (int i = 0; i < mesh.get_num_nodes(); i++) {
      write_real_val(fp, sol_vec[i]);
      std::fprintf(fp, "\n");
    }
  }

 private:
  const Mesh& mesh;
  int vtk_elem_type;
  std::FILE* fp = nullptr;
  bool vtk_has_sol_header = false;
};

#endif  // XCGD_VTK_H