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

// The Ray Tracing in One Weekend scene.
// See https://github.com/raytracing/InOneWeekend/releases/ for this free book.

// public owl API
#include <owl/owl.h>
#include <owl/DeviceMemory.h>
// our device-side data structures
#include "GeomTypes.h"
// external helper stuff for image output

#include <vector>
#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include <random>
#include<ctime>
#include<chrono>
#include<algorithm>
#include<set>



#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#owl.sample(main): " << message << std::endl;    \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#owl.sample(main): " << message << std::endl;    \
  std::cout << OWL_TERMINAL_DEFAULT;

const vec3f lookFrom(13, 2, 3);
const vec3f lookAt(0, 0, 0);
const vec3f lookUp(0.f,1.f,0.f);
const float fovy = 20.f;

std::vector<Sphere> Spheres;
std::vector<int> neighbors;
std::vector<DisjointSet> ds;
extern "C" char deviceCode_ptx[];
//DisjointSet find

int find(int x, const DisjointSet *d)
{

	if (d[x].parent != x) 
	  return(find(d[x].parent,d));
	return 	d[x].parent;
}

/*Command-line args
argv[1] = input file
argv[2] = number of points to read from file
argv[3] = start radius
argv[4] = minPts
argv[5] = file to write execution time
*/

int main(int ac, char **av) {

  // ##################################################################
  // pre-owl host-side set-up
  // ##################################################################

  int dim = 3;
  std::string line;
  std::ifstream myfile;
  myfile.open(av[1]);
  // myfile.open("/home/min/a/nagara16/fast-cuda-gpu-dbscan/CUDA_DCLUST_datasets/3D_iono.txt");
  // myfile.open("/home/min/a/nagara16/ArborX/build/examples/dbscan/porto.txt");
  // myfile.open("/home/min/a/nagara16/Downloads/owl/samples/cmdline/s01-simpleTriangles/testing/3droad_full.csv");
  if (!myfile.is_open()) {
    perror("Error open");
    exit(EXIT_FAILURE);
  }
  std::vector<float> vect;
  int count = atof(av[2]) * dim;

  while (getline(myfile, line) && count > 0) {
    std::stringstream ss(line);
    float i;
    while (ss >> i) {
      vect.push_back(i);
      count--;
      // std::cout << i <<'\n';
      if (ss.peek() == ',')
        ss.ignore();
    }
  }

  // ##################################################################
  // Create scene
  // ##################################################################

  // Select minPts,epsilon
  float radius = atof(av[3]);
  int minPts = atof(av[4]);

  //If dataset is 2D, set z dimension to 0	
  if (dim == 2) {
    for (int i = 0, j = 0; i < vect.size(); i += 2, j += 1) {
      // Spheres.push_back(Sphere{vec3f(vect.at(i),vect.at(i+1),0),-1});
      Spheres.push_back(Sphere{vec3f(vect.at(i), vect.at(i + 1), 0), -1});
      ds.push_back(DisjointSet{j, 0, 0});
    }
  }

  if (dim == 3) {
    for (int i = 0, j = 0; i < vect.size(); i += 3, j += 1) {
      Spheres.push_back(
          Sphere{vec3f(vect.at(i), vect.at(i + 1), vect.at(i + 2)), -1});
      ds.push_back(DisjointSet{j, 0, 0});
    }
  }

  // Init Frame Buffer. Don't need 2D threads, so just use x-dim for threadId
  const vec2i fbSize(Spheres.size(), 1);

  // createScene();
  LOG_OK(" Executing DBSCAN");
  LOG_OK(" dataset size: " << Spheres.size());

  // ##################################################################
  // init owl
  // ##################################################################

  OWLContext context = owlContextCreate(nullptr, 1);
  OWLModule module = owlModuleCreate(context, deviceCode_ptx);

  // ##################################################################
  // set up all the *GEOMETRY* graph we want to render
  // ##################################################################

  OWLVarDecl SpheresGeomVars[] = {
      {"prims", OWL_BUFPTR, OWL_OFFSETOF(SpheresGeom, prims)},
      {"rad", OWL_FLOAT, OWL_OFFSETOF(SpheresGeom, rad)},
      {/* sentinel to mark end of list */}};

  OWLGeomType SpheresGeomType = owlGeomTypeCreate(
      context, OWL_GEOMETRY_USER, sizeof(SpheresGeom), SpheresGeomVars, -1);
  /*owlGeomTypeSetClosestHit(SpheresGeomType,0,
                           module,"Spheres");*/
  owlGeomTypeSetIntersectProg(SpheresGeomType, 0, module, "Spheres");
  owlGeomTypeSetBoundsProg(SpheresGeomType, module, "Spheres");

  // make sure to do that *before* setting up the geometry, since the
  // user geometry group will need the compiled bounds programs upon
  // accelBuild()
  owlBuildPrograms(context);
  LOG_OK("BUILD prog DONE\n");

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  // LOG("building geometries ...");

  /*OWLBuffer frameBuffer
    = owlHostPinnedBufferCreate(context,OWL_INT,
                            Spheres.size());*/
  /*OWLBuffer frameBuffer
   = owlHostPinnedBufferCreate(context,OWL_USER_TYPE(ds[0]),
                           Spheres.size());  */

  OWLBuffer frameBuffer = owlManagedMemoryBufferCreate(
      context, OWL_USER_TYPE(ds[0]), ds.size(), ds.data());

  OWLBuffer SpheresBuffer = owlDeviceBufferCreate(
      context, OWL_USER_TYPE(Spheres[0]), Spheres.size(), Spheres.data());

  OWLGeom SpheresGeom = owlGeomCreate(context, SpheresGeomType);
  owlGeomSetPrimCount(SpheresGeom, Spheres.size());
  owlGeomSetBuffer(SpheresGeom, "prims", SpheresBuffer);
  owlGeomSet1f(SpheresGeom, "rad", radius);

  // ##################################################################
  // Params
  // ##################################################################

  OWLVarDecl myGlobalsVars[] = {
      {"frameBuffer", OWL_BUFPTR, OWL_OFFSETOF(MyGlobals, frameBuffer)},
      {"callNum", OWL_INT, OWL_OFFSETOF(MyGlobals, callNum)},
      {"minPts", OWL_INT, OWL_OFFSETOF(MyGlobals, minPts)},
      {/* sentinel to mark end of list */}};

  OWLParams lp = owlParamsCreate(context, sizeof(MyGlobals), myGlobalsVars, -1);
  owlParamsSetBuffer(lp, "frameBuffer", frameBuffer);
  owlParamsSet1i(lp, "minPts", minPts);

  LOG_OK("Geoms and Params DONE\n");

  // ##################################################################
  // set up all *ACCELS* we need to trace into those groups
  // ##################################################################
 
  OWLGeom userGeoms[] = {SpheresGeom};

  auto start_b = std::chrono::steady_clock::now();
  OWLGroup spheresGroup = owlUserGeomGroupCreate(context, 1, userGeoms);
  owlGroupBuildAccel(spheresGroup);

  OWLGroup world = owlInstanceGroupCreate(context, 1, &spheresGroup);
  owlGroupBuildAccel(world);

  LOG_OK("Group build DONE\n");

  auto end_b = std::chrono::steady_clock::now();
  auto elapsed_b =
      std::chrono::duration_cast<std::chrono::microseconds>(end_b - start_b);
  std::cout << "Build time: " << elapsed_b.count() / 1000000.0 << " seconds."
            << std::endl;

  // ##################################################################
  // set miss and raygen programs
  // ##################################################################

  // -------------------------------------------------------
  // set up miss prog
  // -------------------------------------------------------
  /*OWLVarDecl missProgVars[] = {
    {  }
  };
  // ........... create object  ............................
  OWLMissProg missProg
    = owlMissProgCreate(context,module,"miss",sizeof(MissProgData),
                        missProgVars,-1);
  owlMissProgSet(context,0,missProg);*/

  // -------------------------------------------------------
  // set up ray gen program
  // -------------------------------------------------------
  OWLVarDecl rayGenVars[] = {
      {"spheres", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, spheres)},
      {"fbSize", OWL_INT2, OWL_OFFSETOF(RayGenData, fbSize)},
      {"world", OWL_GROUP, OWL_OFFSETOF(RayGenData, world)},
      {"camera.org", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.origin)},
      {"camera.llc", OWL_FLOAT3,
       OWL_OFFSETOF(RayGenData, camera.lower_left_corner)},
      {"camera.horiz", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.horizontal)},
      {"camera.vert", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.vertical)},
      {/* sentinel to mark end of list */}};

  // ........... create object  ............................
  OWLRayGen rayGen = owlRayGenCreate(context, module, "rayGen",
                                     sizeof(RayGenData), rayGenVars, -1);

  // ........... compute variable values  ..................
  const float vfov = fovy;
  const vec3f vup = lookUp;
  const float aspect = fbSize.x / float(fbSize.y);
  const float theta = vfov * ((float)M_PI) / 180.0f;
  const float half_height = tanf(theta / 2.0f);
  const float half_width = aspect * half_height;
  const float focusDist = 10.f;
  const vec3f origin = lookFrom;
  const vec3f w = normalize(lookFrom - lookAt);
  const vec3f u = normalize(cross(vup, w));
  const vec3f v = cross(w, u);
  const vec3f lower_left_corner = origin - half_width * focusDist * u -
                                  half_height * focusDist * v - focusDist * w;
  const vec3f horizontal = 2.0f * half_width * focusDist * u;
  const vec3f vertical = 2.0f * half_height * focusDist * v;

  // ----------- set variables  ----------------------------
  owlRayGenSetBuffer(rayGen, "spheres", SpheresBuffer);
  owlRayGenSet2i(rayGen, "fbSize", (const owl2i &)fbSize);
  owlRayGenSetGroup(rayGen, "world", world);
  owlRayGenSet3f(rayGen, "camera.org", (const owl3f &)origin);
  owlRayGenSet3f(rayGen, "camera.llc", (const owl3f &)lower_left_corner);
  owlRayGenSet3f(rayGen, "camera.horiz", (const owl3f &)horizontal);
  owlRayGenSet3f(rayGen, "camera.vert", (const owl3f &)vertical);

  // programs have been built before, but have to rebuild raygen and
  // miss progs

  owlBuildPrograms(context);
  owlBuildPipeline(context);
  owlBuildSBT(context);

  // ##################################################################
  // DBSCAN Start
  // ##################################################################

  std::ofstream ofile;
  ofile.open(av[5], std::ios::app);

  ////////////////////////////////////////////////////Call-1////////////////////////////////////////////////////////////////////////////

  //Core point identification
  auto start1 = std::chrono::steady_clock::now();

  owlParamsSet1i(lp, "callNum", 1);
  owlLaunch2D(rayGen, fbSize.x, fbSize.y, lp);
  cudaDeviceSynchronize();
  auto end1 = std::chrono::steady_clock::now();
  auto elapsed1 =
      std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
  std::cout << "Core points time: " << elapsed1.count() / 1000000.0
            << " seconds." << std::endl;

  ////////////////////////////////////////////////////Call-2////////////////////////////////////////////////////////////////////////////

  //Cluster formation
  auto start = std::chrono::steady_clock::now();
  owlParamsSet1i(lp, "callNum", 2);
  owlLaunch2D(rayGen, fbSize.x, fbSize.y, lp);
  auto end = std::chrono::steady_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  const DisjointSet *fb =
      (const DisjointSet *)owlBufferGetPointer(frameBuffer, 0);

  auto tot = (elapsed.count() / 1000000.0) + (elapsed_b.count() / 1000000.0) +
             (elapsed1.count() / 1000000.0);
  std::cout << "Execution time: " << elapsed.count() / 1000000.0 << " seconds."
            << std::endl;
  std::cout << "Total time = " << tot << '\n';

  int temp;

  //Write cluster results to file
  // ofile << "ind" << '\t'<< "cluster"<< std::endl;
  /*for(int i = 0; i < Spheres.size(); i++)
  {
          temp = find(fb[i].parent,fb);

                  ofile << i << '\t'<< temp<< std::endl;
          //cout<<i<<'\t'<<find(fb[i].parent,fb)<<'\n';
  }*/

  //Write execution time
  ofile << tot << std::endl;

  // ##################################################################
  // and finally, clean up
  // ##################################################################

  LOG("destroying devicegroup ...");
  owlContextDestroy(context);

  
}