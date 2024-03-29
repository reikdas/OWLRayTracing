#pragma once

#include <owl/owl.h>
#include <string>
#include <vector>

#define GRID_SIZE 10000.0f // this has to be smaller than the TRIANGLEX_THRESHOLD in hostCode.cpp
#define THRESHOLD 0.6f
#define GRAVITATIONAL_CONSTANT .1f
#define BUCKET_SIZE 32
#define ERRORING_POINT 2467409
#define PRIM_ID 890988

using namespace std;

typedef enum _bh_node_type {
  bhNonLeafNode,
	bhLeafNode
} bh_node_type;

namespace owl {
  typedef struct _vec3f {
    float x;
    float y;
    float z;
  } vec3float;

  struct Point {
    vec3float pos;
    //vec3float vel;
    float mass;
    int idX;
  };

  struct Node {
    bh_node_type type;
    float quadrantX;
    float quadrantY;
    float quadrantZ;
    float mass;
    float s;
    vec3float cofm;
    Node* children[8];
    std::vector<int> particles;
    int pointID;
    uint32_t dfsIndex;

    Node(float x, float y, float z, float s, int pointID);

    Node() {
      mass = 0.0f;
      s = 0.0f;
      cofm.x = 0.0f;
      cofm.y = 0.0f;
      cofm.z = 0.0f;
      dfsIndex = 0;
      for(int i = 0; i < 8; i++) {
        children[i] = nullptr;
      }
      particles.reserve(BUCKET_SIZE);
      type = bhLeafNode;
      pointID = -1;
    }

  };

  class BarnesHutTree {
    private:
      Node* root;
      float theta;
      float gridSize;

      //void calculateCenterOfMass(Node* node);

    public:
      BarnesHutTree(float theta, float gridSize);
      ~BarnesHutTree();

      void insertNode(Node* node, Node* point, float s);
      void computeCOM(Node *root);
      void printTree(Node* root, int depth);
      void computeForces(Node* node, std::vector<Point> &points, std::vector<float>& cpuComputedForces);
      void traverseOctreeDFS(Node* node, std::vector<Node*>& leafNodes, float *minS);
  };
}




