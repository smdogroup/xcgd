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
  std::fprintf(fp, "%-25.15e", val);
}

void write_real_val(std::FILE* fp, std::complex<double> val) {
  std::fprintf(fp, "%-25.15e", val.real());
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

    // If not provided, infer vtk type id from mesh.corner_nodes_per_element
    if (vtk_elem_type == -1) {
      if constexpr (spatial_dim == 2) {
        switch (mesh.corner_nodes_per_element) {
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
        switch (mesh.corner_nodes_per_element) {
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
      snprintf(
          msg, 256,
          "Cannot infer element type from corner_nodes_per_element(%d) for a "
          "%d-dimensional mesh",
          mesh.corner_nodes_per_element, spatial_dim);
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
                 mesh.get_num_elements() * (1 + mesh.corner_nodes_per_element));
    for (int i = 0; i < mesh.get_num_elements(); i++) {
      std::fprintf(fp, "%d ", mesh.corner_nodes_per_element);
      int nodes[mesh.corner_nodes_per_element];
      mesh.get_elem_corner_nodes(i, nodes);
      for (int j = 0; j < mesh.corner_nodes_per_element; j++) {
        std::fprintf(fp, "%d ", nodes[j]);
      }
      std::fprintf(fp, "\n");
    }

    // Write cell type
    std::fprintf(fp, "CELL_TYPES %d\n", mesh.get_num_elements());
    for (int i = 0; i < mesh.get_num_elements(); i++) {
      std::fprintf(fp, "%d\n", vtk_elem_type);
    }
  }

  // Write nodal scalars
  void write_sol(const std::string sol_name, const T* sol_vec) {
    // Write header
    if (!vtk_has_nodal_sol_header) {
      std::fprintf(fp, "POINT_DATA %d\n", mesh.get_num_nodes());
      vtk_has_nodal_sol_header = true;
    }
    std::fprintf(fp, "SCALARS %s double 1\n", sol_name.c_str());
    std::fprintf(fp, "LOOKUP_TABLE default\n");

    // Write data
    for (int i = 0; i < mesh.get_num_nodes(); i++) {
      write_real_val(fp, sol_vec[i]);
      std::fprintf(fp, "\n");
    }
  }

  // Write nodal vectors
  void write_vec(const std::string sol_name, const T* vec) {
    // Write header
    if (!vtk_has_nodal_sol_header) {
      std::fprintf(fp, "POINT_DATA %d\n", mesh.get_num_nodes());
      vtk_has_nodal_sol_header = true;
    }
    std::fprintf(fp, "VECTORS %s double\n", sol_name.c_str());

    // Write data
    for (int i = 0; i < mesh.get_num_nodes(); i++) {
      if constexpr (spatial_dim == 2) {
        write_real_val(fp, vec[spatial_dim * i]);
        write_real_val(fp, vec[spatial_dim * i + 1]);
        write_real_val(fp, 0.0);
      } else {
        write_real_val(fp, vec[spatial_dim * i]);
        write_real_val(fp, vec[spatial_dim * i + 1]);
        write_real_val(fp, vec[spatial_dim * i + 2]);
      }
      std::fprintf(fp, "\n");
    }
  }

  // Write cell scalars
  void write_cell_sol(const std::string sol_name, const T* sol_vec) {
    // Write header
    if (!vtk_has_cell_sol_header) {
      std::fprintf(fp, "CELL_DATA %d\n", mesh.get_num_elements());
      vtk_has_cell_sol_header = true;
    }
    std::fprintf(fp, "SCALARS %s double 1\n", sol_name.c_str());
    std::fprintf(fp, "LOOKUP_TABLE default\n");

    // Write data
    for (int e = 0; e < mesh.get_num_elements(); e++) {
      write_real_val(fp, sol_vec[e]);
      std::fprintf(fp, "\n");
    }
  }

  // Write cell vectors
  void write_cell_vec(const std::string sol_name, const T* sol_vec) {
    // Write header
    if (!vtk_has_cell_sol_header) {
      std::fprintf(fp, "CELL_DATA %d\n", mesh.get_num_elements());
      vtk_has_cell_sol_header = true;
    }
    std::fprintf(fp, "VECTORS %s double\n", sol_name.c_str());

    // Write data
    for (int e = 0; e < mesh.get_num_elements(); e++) {
      if constexpr (spatial_dim == 2) {
        write_real_val(fp, sol_vec[spatial_dim * e]);
        write_real_val(fp, sol_vec[spatial_dim * e + 1]);
        write_real_val(fp, 0.0);
      } else {
        write_real_val(fp, sol_vec[spatial_dim * e]);
        write_real_val(fp, sol_vec[spatial_dim * e + 1]);
        write_real_val(fp, sol_vec[spatial_dim * e + 2]);
      }
      std::fprintf(fp, "\n");
    }
  }

 private:
  const Mesh& mesh;
  int vtk_elem_type;
  std::FILE* fp = nullptr;
  bool vtk_has_nodal_sol_header = false;
  bool vtk_has_cell_sol_header = false;
};

/**
 * @brief Scattered data to vtk
 *
 * Example usage:
 *   FieldToVTK<T, spatial_dim> vtk("field.vtk");
 *   vtk.add_scalar_field(xloc, vals);
 *   vtk.write_vtk();
 */
template <typename T, int spatial_dim>
class FieldToVTK {
 public:
  FieldToVTK(const std::string vtk_name = "field.vtk") {
    fp = std::fopen(vtk_name.c_str(), "w+");

    // Write header
    std::fprintf(fp, "# vtk DataFile Version 3.0\n");
    std::fprintf(fp, "my example\n");
    std::fprintf(fp, "ASCII\n");
    std::fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
  }

  // void add_vector_field(std::vector<T> xloc, std::vector<T> vecs) {}

  void add_scalar_field(std::vector<T> xloc, std::vector<T> vals) {
    xloc_scalars.insert(xloc_scalars.end(), xloc.begin(), xloc.end());
    scalars.insert(scalars.end(), vals.begin(), vals.end());
  }

  void write_vtk() {
    int nverts = scalars.size();
    if (nverts * spatial_dim != xloc_scalars.size()) {
      char msg[256];
      std::snprintf(msg, 256,
                    "incompatible scalars (size %d) and xloc_scalars (size "
                    "%d) for the %d dimentional problem",
                    (int)scalars.size(), (int)xloc_scalars.size(), spatial_dim);
      throw std::runtime_error(msg);
    }

    // Write vertices
    std::fprintf(fp, "POINTS %d double\n", nverts);
    for (int i = 0; i < xloc_scalars.size(); i += spatial_dim) {
      write_real_val(fp, xloc_scalars[i]);
      write_real_val(fp, xloc_scalars[i + 1]);
      if (spatial_dim == 2) {
        write_real_val(fp, 0.0);
      } else {
        write_real_val(fp, xloc_scalars[i + 2]);
      }
      std::fprintf(fp, "\n");
    }

    // Write field data
    std::fprintf(fp, "POINT_DATA %d \n", nverts);
    std::fprintf(fp, "SCALARS scalar double 1\n");
    std::fprintf(fp, "LOOKUP_TABLE default\n");
    for (T s : scalars) {
      write_real_val(fp, s);
      std::fprintf(fp, "\n");
    }
  }

  ~FieldToVTK() { std::fclose(fp); }

 private:
  std::FILE* fp;
  // std::vector<T> xloc_vectors;
  // std::vector<T> vectors;
  std::vector<T> xloc_scalars;
  std::vector<T> scalars;
};

/**
 * @brief Scattered data to vtk
 *
 * Example usage:
 *   FieldToVTK<T, spatial_dim> vtk("field.vtk");
 *   vtk.add_scalar_field(xloc, vals);
 *   vtk.write_vtk();
 */
template <typename T, int spatial_dim>
class FieldToVTKNew {
 public:
  FieldToVTKNew(const std::string vtk_name = "field.vtk") {
    fp = std::fopen(vtk_name.c_str(), "w+");

    // Write header
    std::fprintf(fp, "# vtk DataFile Version 3.0\n");
    std::fprintf(fp, "my example\n");
    std::fprintf(fp, "ASCII\n");
    std::fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
  }

  void add_mesh(const std::vector<T>& xloc) {
    xloc_scalars.insert(xloc_scalars.end(), xloc.begin(), xloc.end());
  }

  void add_sol(const std::string name, const std::vector<T>& vals) {
    scalars[name].insert(scalars[name].end(), vals.begin(), vals.end());
  }

  void reset_sol(const std::string name) { scalars[name].clear(); }

  void add_vec(const std::string name, const std::vector<T>& vec) {
    vectors[name].insert(vectors[name].end(), vec.begin(), vec.end());
  }

  void reset_vec(const std::string name) { vectors[name].clear(); }

  void write_mesh() {
    int nverts = xloc_scalars.size() / spatial_dim;
    std::fprintf(fp, "POINTS %d double\n", nverts);
    for (int i = 0; i < xloc_scalars.size(); i += spatial_dim) {
      write_real_val(fp, xloc_scalars[i]);
      write_real_val(fp, xloc_scalars[i + 1]);
      if (spatial_dim == 2) {
        write_real_val(fp, 0.0);
      } else {
        write_real_val(fp, xloc_scalars[i + 2]);
      }
      std::fprintf(fp, "\n");
    }
  }

  void write_sol(const std::string sol_name) {
    int nverts = scalars[sol_name].size();
    if (nverts * spatial_dim != xloc_scalars.size()) {
      char msg[256];
      std::snprintf(msg, 256,
                    "incompatible scalars (size %d) and xloc_scalars (size "
                    "%d) for the %d dimentional problem",
                    (int)scalars[sol_name].size(), (int)xloc_scalars.size(),
                    spatial_dim);
      throw std::runtime_error(msg);
    }

    if (!vtk_has_nodal_sol_header) {
      vtk_has_nodal_sol_header = true;
      std::fprintf(fp, "POINT_DATA %d \n", nverts);
    }

    std::fprintf(fp, "SCALARS %s double 1\n", sol_name.c_str());
    std::fprintf(fp, "LOOKUP_TABLE default\n");
    for (T s : scalars[sol_name]) {
      write_real_val(fp, s);
      std::fprintf(fp, "\n");
    }
  }

  void write_vec(const std::string sol_name) {
    if (vectors[sol_name].size() != xloc_scalars.size()) {
      char msg[256];
      std::snprintf(msg, 256,
                    "incompatible vectors (size %d) and xloc_scalars (size "
                    "%d) for the %d dimentional problem",
                    (int)vectors[sol_name].size(), (int)xloc_scalars.size(),
                    spatial_dim);
      throw std::runtime_error(msg);
    }

    int nverts = xloc_scalars.size() / spatial_dim;
    if (!vtk_has_nodal_sol_header) {
      vtk_has_nodal_sol_header = true;
      std::fprintf(fp, "POINT_DATA %d \n", nverts);
    }

    std::fprintf(fp, "VECTORS %s double\n", sol_name.c_str());

    for (int i = 0; i < nverts; i++) {
      if constexpr (spatial_dim == 2) {
        write_real_val(fp, vectors[sol_name][spatial_dim * i]);
        write_real_val(fp, vectors[sol_name][spatial_dim * i + 1]);
        write_real_val(fp, 0.0);
      } else {
        write_real_val(fp, vectors[sol_name][spatial_dim * i]);
        write_real_val(fp, vectors[sol_name][spatial_dim * i + 1]);
        write_real_val(fp, vectors[sol_name][spatial_dim * i + 2]);
      }
      std::fprintf(fp, "\n");
    }
  }

  ~FieldToVTKNew() { std::fclose(fp); }

 private:
  std::FILE* fp;
  std::vector<T> xloc_scalars;
  std::map<std::string, std::vector<T>> scalars;
  std::map<std::string, std::vector<T>> vectors;

  bool vtk_has_nodal_sol_header = false;
};

// Save stencils for selected elements to vtk
template <typename T, class Mesh>
class StencilToVTK {
 private:
  static constexpr int spatial_dim = Mesh::spatial_dim;

 public:
  StencilToVTK(const Mesh& mesh, const std::string vtk_name = "stencils.vtk")
      : mesh(mesh) {
    // Open file and destroy old contents
    fp = std::fopen(vtk_name.c_str(), "w+");

    if (spatial_dim != 2 and spatial_dim != 3) {
      char msg[256];
      std::snprintf(msg, sizeof(msg),
                    "Invalid spatial_dim, got %d, expect 2 or 3", spatial_dim);
      throw std::runtime_error(msg);
    }

    // Write xloc of nodes
    write_nodes();
  }

  ~StencilToVTK() {
    // Close file
    std::fclose(fp);
  }

  void write_stencils(std::map<int, std::vector<int>>& stencils) {
    int nelems = mesh.get_num_elements();
    int poly_data_size = nelems;
    for (const auto& [elem, nodes] : stencils) {
      poly_data_size += nodes.size();
    }

    // Connectivity
    std::fprintf(fp, "CELLS %d %d\n", nelems, poly_data_size);
    for (int elem = 0; elem < nelems; elem++) {
      if (stencils.count(elem)) {
        std::fprintf(fp, "%zu ", stencils[elem].size());
        for (int n : stencils[elem]) {
          std::fprintf(fp, "%d ", n);
        }
        std::fprintf(fp, "\n");
      } else {
        std::fprintf(fp, "0\n");
      }
    }

    // Cell types are all polygons
    std::fprintf(fp, "CELL_TYPES %d\n", nelems);
    for (int elem = 0; elem < nelems; elem++) {
      std::fprintf(fp, "%d\n", VTKID::POLYGON);
    }
  }

 private:
  void write_nodes() const {
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
  }

  const Mesh& mesh;
  std::FILE* fp = nullptr;
};

#endif  // XCGD_VTK_H
