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
          nodeSets[currentSetName] = NodeSet{currentSetName};
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