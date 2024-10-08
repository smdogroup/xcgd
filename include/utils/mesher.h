#ifndef XCGD_MESHER_H
#define XCGD_MESHER_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Data Structures
struct Node {
  int index;
  double x, y, z;
};

struct Element {
  int index;
  std::vector<int> nodeIndices;
};

struct NodeSet {
  std::string name;
  std::vector<int> nodeIndices;
};

// Utility Functions
std::string trim(const std::string &str) {
  size_t first = str.find_first_not_of(' ');
  if (std::string::npos == first) {
    return str;
  }
  size_t last = str.find_last_not_of(' ');
  return str.substr(first, (last - first + 1));
}

// Main Parsing Logic
template <typename T>
void load_mesh(std::string filename, int *num_elements, int *num_nodes,
               int **element_nodes, T **xloc) {
  std::ifstream meshFile(filename);
  if (!meshFile.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
  }

  std::string line;

  std::vector<Node> nodes;
  std::map<int, Element> elements;
  std::map<std::string, NodeSet> nodeSets;

  bool inNodesSection = false, inElementsSection = false,
       inNodeSetsSection = false;
  std::string currentSetName;

  while (getline(meshFile, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '*') {
      inNodesSection = line.find("*Node") != std::string::npos;
      inElementsSection = line.find("*Element") != std::string::npos;
      inNodeSetsSection = line.find("*Nset") != std::string::npos;

      if (inNodeSetsSection) {
        size_t namePos = line.find("Nset=");
        if (namePos != std::string::npos) {
          currentSetName = line.substr(namePos + 5);
          nodeSets[currentSetName] = NodeSet{currentSetName, {}};
        }
      }

      continue;
    }

    if (inNodesSection) {
      std::istringstream iss(line);
      std::string indexStr;
      std::getline(iss, indexStr,
                   ',');  // Read up to the first comma to get the node index.
      int nodeIndex = std::stoi(indexStr);  // Convert index string to int.

      Node node;
      node.index = nodeIndex;

      std::string coordinateStr;
      std::getline(iss, coordinateStr,
                   ',');  // Read up to the next comma for the x coordinate.
      node.x = std::stod(coordinateStr);  // Convert to double.

      std::getline(iss, coordinateStr,
                   ',');  // Read up to the next comma for the y coordinate.
      node.y = std::stod(coordinateStr);  // Convert to double.

      std::getline(iss,
                   coordinateStr);  // Read the rest of the line for the z
                                    // coordinate (assuming no more commas).
      node.z = std::stod(coordinateStr);  // Convert to double.

      nodes.push_back(node);
    } else if (inElementsSection) {
      std::istringstream iss(line);
      Element element;
      if (!(iss >> element.index)) {  // Read and check the element's index.
        std::cerr << "Failed to read element index from line: " << line
                  << std::endl;
        continue;  // Skip to the next line if the element index can't be read.
      }

      // Read the rest of the line as a single string.
      std::string restOfLine;
      std::getline(iss, restOfLine);

      // Use another stringstream to parse the node indices from restOfLine.
      std::istringstream nodeStream(restOfLine);
      std::string
          nodeIndexStr;  // Use a string to temporarily hold each node index.

      while (std::getline(nodeStream, nodeIndexStr,
                          ',')) {     // Read up to the next comma.
        if (!nodeIndexStr.empty()) {  // Check if the string is not empty.
          std::istringstream indexStream(
              nodeIndexStr);  // Use another stringstream to convert string to
                              // int.
          int nodeIndex;
          if (indexStream >> nodeIndex) {  // Convert the string to an int.
            element.nodeIndices.push_back(nodeIndex);
          }
        }
      }
      elements[element.index] = element;
    } else if (inNodeSetsSection && !currentSetName.empty()) {
      std::istringstream iss(line);
      int nodeIndex;
      while (iss >> nodeIndex) {
        nodeSets[currentSetName].nodeIndices.push_back(nodeIndex);
      }
    }
  }

  meshFile.close();

  // Convert elements and nodeSets to flat structures
  int num_elems = elements.size();
  int *elem_nodes = new int[10 * num_elems];

  for (const auto &elem : elements) {
    for (int j = 0; j < 10; j++) {
      elem_nodes[10 * (elem.second.index - 1) + j] =
          elem.second.nodeIndices[j] - 1;
    }
  }

  int num_ns = nodes.size();
  T *x = new T[3 * num_ns];

  for (const auto &node : nodes) {
    x[3 * (node.index - 1)] = node.x;
    x[3 * (node.index - 1) + 1] = node.y;
    x[3 * (node.index - 1) + 2] = node.z;
  }

  *num_elements = num_elems;
  *num_nodes = num_ns;
  *element_nodes = elem_nodes;
  *xloc = x;
}

template <typename T>
void create_single_element_mesh(int *num_elements, int *num_nodes,
                                int **element_nodes, T **xloc,
                                int *ndof_bcs = nullptr,
                                int **dof_bcs = nullptr) {
  int num_elem = 1;
  int num_ns = 10;

  int ndof_bcs_ = 6;
  int *dof_bcs_ = new int[3 * ndof_bcs_];

  int *elem_nodes = new int[10];
  T *x = new T[3 * num_ns];

  for (int i = 0; i < num_ns; i++) {
    elem_nodes[i] = i;
  }

  x[0] = 1.0;
  x[1] = 0.0;
  x[2] = 0.0;

  x[3] = 0.0;
  x[4] = 1.0;
  x[5] = 0.0;

  x[6] = 0.0;
  x[7] = 0.0;
  x[8] = 0.0;

  x[9] = 0.0;
  x[10] = 0.0;
  x[11] = 1.0;

  x[12] = 0.6;
  x[13] = 0.6;
  x[14] = 0.0;

  x[15] = 0.0;
  x[16] = 0.5;
  x[17] = 0.0;

  x[18] = 0.5;
  x[19] = 0.0;
  x[20] = 0.0;

  x[21] = 0.6;
  x[22] = 0.0;
  x[23] = 0.6;

  x[24] = 0.0;
  x[25] = 0.6;
  x[26] = 0.6;

  x[27] = 0.0;
  x[28] = 0.0;
  x[29] = 0.5;

  std::vector<int> bc_nodes = {0, 1, 2, 4, 5, 6};

  int index = 0;
  for (int node : bc_nodes) {
    for (int i = 0; i < 3; i++, index++) {
      dof_bcs_[index] = 3 * node + i;
    }
  }

  *element_nodes = elem_nodes;
  *num_nodes = num_ns;
  *num_elements = num_elem;
  *xloc = x;
  if (ndof_bcs) {
    *ndof_bcs = ndof_bcs_;
  }
  if (dof_bcs) {
    *dof_bcs = dof_bcs_;
  }
}

template <typename T>
void create_2d_rect_quad_mesh(int nxy[2], T lxy[2], int *num_elements,
                              int *num_nodes, int **element_nodes, T **xloc,
                              int *ndof_bcs = nullptr,
                              int **dof_bcs = nullptr) {
  int nx = nxy[0], ny = nxy[1];
  T lx = lxy[0], ly = lxy[1];
  int _num_elements = nx * ny;
  int _num_nodes = (nx + 1) * (ny + 1);
  int _num_nodes_per_elem = 4;
  int *_element_nodes = new int[_num_elements * _num_nodes_per_elem];
  T *_xloc = new T[2 * _num_nodes];
  int _ndof_bcs = 2 * (ny + 1);
  int *_dof_bcs = new int[_ndof_bcs];

  int idx = 0;
  for (int j = 0; j < ny + 1; j++) {
    int node = (nx + 1) * j;
    _dof_bcs[idx] = 2 * node;
    idx++;
    _dof_bcs[idx] = 2 * node + 1;
    idx++;
  }

  // Set X
  for (int j = 0; j < ny + 1; j++) {
    for (int i = 0; i < nx + 1; i++) {
      int node = i + (nx + 1) * j;

      _xloc[2 * node] = lx * T(i) / T(nx);
      _xloc[2 * node + 1] = ly * T(j) / T(ny);
    }
  }

  // Set connectivity
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      int elem = i + nx * j;

      int conn_coord[4];
      for (int jj = 0, index = 0; jj < 2; jj++) {
        for (int ii = 0; ii < 2; ii++, index++) {
          conn_coord[index] = (i + ii) + (nx + 1) * (j + jj);
        }
      }

      // Convert to the correct connectivity
      _element_nodes[4 * elem + 0] = conn_coord[0];
      _element_nodes[4 * elem + 1] = conn_coord[1];
      _element_nodes[4 * elem + 2] = conn_coord[3];
      _element_nodes[4 * elem + 3] = conn_coord[2];
    }
  }

  *num_elements = _num_elements;
  *num_nodes = _num_nodes;
  *element_nodes = _element_nodes;
  *xloc = _xloc;
  if (ndof_bcs) {
    *ndof_bcs = _ndof_bcs;
  }
  if (dof_bcs) {
    *dof_bcs = _dof_bcs;
  }
}

template <typename T>
void create_3d_box_tet_mesh(int nxyz[3], T lxyz[3], int *num_elements,
                            int *num_nodes, int **element_nodes, T **xloc) {
  int nx = nxyz[0], ny = nxyz[1], nz = nxyz[2];
  T lx = lxyz[0], ly = lxyz[1], lz = lxyz[2];
  int _num_elements = nx * ny * nz * 6;
  int _num_nodes = (2 * nx + 1) * (2 * ny + 1) * (2 * nz + 1);
  int _num_nodes_per_elem = 10;
  int *_element_nodes = new int[_num_elements * _num_nodes_per_elem];
  T *_xloc = new T[3 * _num_nodes];

  auto get_node = [nx, ny](int i, int j, int k) {
    return i + (2 * nx + 1) * j + (2 * nx + 1) * (2 * ny + 1) * k;
  };

  // Set X
  for (int k = 0; k < 2 * nz + 1; k++) {
    for (int j = 0; j < 2 * ny + 1; j++) {
      for (int i = 0; i < 2 * nx + 1; i++) {
        int node = get_node(i, j, k);

        _xloc[3 * node] = lx * T(i) / T(nx);
        _xloc[3 * node + 1] = ly * T(j) / T(ny);
        _xloc[3 * node + 2] = lz * T(k) / T(nz);
      }
    }
  }

  // Set connectivity
  int *conn_ptr = _element_nodes;
  for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
        int pt[27];
        for (int kk = 0, index = 0; kk < 3; kk++) {
          for (int jj = 0; jj < 3; jj++) {
            for (int ii = 0; ii < 3; ii++, index++) {
              pt[index] = get_node(2 * i + ii, 2 * j + jj, 2 * k + kk);
            }
          }
        }

        conn_ptr[0] = pt[0];
        conn_ptr[1] = pt[20];
        conn_ptr[2] = pt[24];
        conn_ptr[3] = pt[18];
        conn_ptr[4] = pt[10];
        conn_ptr[5] = pt[22];
        conn_ptr[6] = pt[12];
        conn_ptr[7] = pt[9];
        conn_ptr[8] = pt[19];
        conn_ptr[9] = pt[21];
        conn_ptr += 10;

        conn_ptr[0] = pt[0];
        conn_ptr[1] = pt[2];
        conn_ptr[2] = pt[6];
        conn_ptr[3] = pt[24];
        conn_ptr[4] = pt[1];
        conn_ptr[5] = pt[4];
        conn_ptr[6] = pt[3];
        conn_ptr[7] = pt[12];
        conn_ptr[8] = pt[13];
        conn_ptr[9] = pt[15];
        conn_ptr += 10;

        conn_ptr[0] = pt[24];
        conn_ptr[1] = pt[0];
        conn_ptr[2] = pt[2];
        conn_ptr[3] = pt[20];
        conn_ptr[4] = pt[12];
        conn_ptr[5] = pt[1];
        conn_ptr[6] = pt[13];
        conn_ptr[7] = pt[22];
        conn_ptr[8] = pt[10];
        conn_ptr[9] = pt[11];
        conn_ptr += 10;

        conn_ptr[0] = pt[8];
        conn_ptr[1] = pt[26];
        conn_ptr[2] = pt[24];
        conn_ptr[3] = pt[20];
        conn_ptr[4] = pt[17];
        conn_ptr[5] = pt[25];
        conn_ptr[6] = pt[16];
        conn_ptr[7] = pt[14];
        conn_ptr[8] = pt[23];
        conn_ptr[9] = pt[22];
        conn_ptr += 10;

        conn_ptr[0] = pt[24];
        conn_ptr[1] = pt[2];
        conn_ptr[2] = pt[8];
        conn_ptr[3] = pt[20];
        conn_ptr[4] = pt[13];
        conn_ptr[5] = pt[5];
        conn_ptr[6] = pt[16];
        conn_ptr[7] = pt[22];
        conn_ptr[8] = pt[11];
        conn_ptr[9] = pt[14];
        conn_ptr += 10;

        conn_ptr[0] = pt[6];
        conn_ptr[1] = pt[2];
        conn_ptr[2] = pt[8];
        conn_ptr[3] = pt[24];
        conn_ptr[4] = pt[4];
        conn_ptr[5] = pt[5];
        conn_ptr[6] = pt[7];
        conn_ptr[7] = pt[15];
        conn_ptr[8] = pt[13];
        conn_ptr[9] = pt[16];
        conn_ptr += 10;
      }
    }
  }

  *num_elements = _num_elements;
  *num_nodes = _num_nodes;
  *element_nodes = _element_nodes;
  *xloc = _xloc;
}

#endif  // XCGD_MESHER_H
