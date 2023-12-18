#pragma once

#include <owl/owl.h>
#include <string>
#include <vector>

#define GRID_SIZE 10000.0f // this has to be smaller than the TRIANGLEX_THRESHOLD in hostCode.cpp
#define THRESHOLD 0.5f
#define GRAVITATIONAL_CONSTANT .0001f
#define MAX_POINTS_PER_LEAF 32

using namespace std;

namespace owl {

  struct Point {
    float x;
    float y;
    float z;
    float vel_x;
    float vel_y;
    float vel_z;
    float mass;
    int idX;
  };


  struct Node {
    float quadrantX;
    float quadrantY;
    float quadrantZ;
    float mass;
    float s;
    // uint8_t numPoints;
    // int pointsIdx[MAX_POINTS_PER_LEAF];
    float centerOfMassX;
    float centerOfMassY;
    float centerOfMassZ;
    Node* children[8];
    // Node* nw;
    // Node* ne;
    // Node* sw;
    // Node* se;
    bool isLeaf;
    int pointID;
    uint32_t dfsIndex;

    Node(float x, float y, float z, float s);

    Node() {
      mass = 0;
      s = 0;
      centerOfMassX = 0;
      centerOfMassY = 0;
      centerOfMassZ = 0;
      dfsIndex = 0;
      for(int i = 0; i < 8; i++) {
        children[i] = nullptr;
      }
      isLeaf = false;
      pointID = 0;
    }

  };

  class BarnesHutTree {
    private:
      Node* root;
      float theta;
      float gridSize;

      void splitNode(Node* node);
      //void calculateCenterOfMass(Node* node);

    public:
      BarnesHutTree(float theta, float gridSize);
      ~BarnesHutTree();

      void insertNode(Node* node, const Point& point);
      void printTree(Node* root, int depth, std::string corner);
      void computeForces(Node* node, std::vector<Point> points, std::vector<float>& cpuComputedForces);
      //void calculateCenterOfMass();
  };
}




