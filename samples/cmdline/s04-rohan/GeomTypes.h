// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "Materials.h"
#include "barnesHutTree.h"

namespace owl {

	struct Storage {
		int *neigh;
	};

  // ==================================================================
  /* the raw geometric shape of a sphere, without material - this is
     what goes into intersection and bounds programs */
  // ==================================================================
  struct Sphere {
    vec3f center;
    float mass;
    bool isLeaf; 
  };

  struct SpheresGeom {
    Sphere *prims;
    float rad;
  };
 

  // ==================================================================
  /* and finally, input for raygen and miss programs */
  // ==================================================================
  struct RayGenData
  {
    uint32_t *fbPtr;
    vec2i  fbSize;
    OptixTraversableHandle *worlds;
    int sbtOffset;
    //Sphere *internalSpheres;
    Point *points;
    
    struct {
      vec3f origin;
      vec3f lower_left_corner;
      vec3f horizontal;
      vec3f vertical;
    } camera;
  };

  struct MissProgData
  {
    /* nothing in this example */
  };

	struct MyGlobals 
	{	
		//Sphere *spheres;	
		//int *frameBuffer;
		//DisjointSet *ds;
		//Sphere *frameBuffer;
		//int **frameBuffer;
	};

}
