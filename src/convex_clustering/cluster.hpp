#include<iostream>
#include<sstream>
#include<fstream>
#include<cstdlib>
#include<math.h>
#include<vector>
#include<string.h>
#ifdef USE_GPU
#include<CL/cl.h>
#include<CL/cl.hpp>
#endif

using namespace std;


class cluster_t:public proxmap_t{
public:
  cluster_t();
  ~cluster_t();
private:
  void allocate_memory(string configfile);
  void parse_config_line(string & key, istringstream & iss);
  void iterate();
  float get_map_distance();
  float evaluate_obj();
  void print_output();
  bool in_feasible_region();

  bool init(const char * config_file);
  //void init(float rho, const char * genofile, const char * weightsfile);
  void coalesce();
  void load_compact_geno(const char * genofile);
  void load_compact_weights(const char * weightsfile);
  void print_cluster(ostream & os);
  bool run_gpu;
#ifdef USE_GPU
  bool debug_opencl;
  cl_int err;
  cl::Context * context;
  cl::CommandQueue * commandQueue;
  cl::Program * program;
  vector<cl::Device> devices;
  cl::Kernel * kernel_store_U_project;
  cl::Buffer * buffer_offsets;
  cl::Buffer * buffer_genotypes;
  cl::Buffer * buffer_U;
  cl::Buffer * buffer_U_project;
  cl::Buffer * buffer_U_project_orig;
  cl::Buffer * buffer_weights;
  cl::Buffer * buffer_V_project_coeff;
  //cl::Buffer * buffer_;
  
  void init_opencl();
  void createKernel(const char * name, cl::Kernel * & kernel);
  void runKernel(const char * name, cl::Kernel * & kernel,int wg_x,int wg_y, int wg_z, int wi_x,int wi_y, int wi_z);
  template<class T> void createBuffer(int rw,int dim,const char * label,cl::Buffer * & buf);
  template<typename T> void setArg(cl::Kernel * & kernel,int & index,T arg,const char * label);
  template<typename T> void writeToBuffer(cl::Buffer * & buffer,int dim,T hostArr,const char * label);
  template<typename T> void readFromBuffer(cl::Buffer * & buffer,int dim,T hostArr,const char * label);

#endif
  int print_index;
  bool coeff_defined;
  float map_distance;
  float dist_func;
  float large;
  float * U;
  float * U_project;
  float * U_project_orig;
  float * V_project_coeff;
  float mu;
  float * genotypes;
  float * weights;
  int n,n2,p;
  int iter;
  float delta,current_vnorm,last_vnorm;
  int triangle_dim;
  int * offsets;

  void initialize(float mu);
  void load_into_triangle(const char * filename,float * & mat,int rows, int cols);

  float infer_rho();
  void init_v_project_coeff();
  // if returns true, it is non zero
  bool get_updated_v(int index1,int index2, float * v);
  void update_projection();
  void update_projection_nonzero();
  void update_map_distance();
  void update_u();
  void check_constraint();
  
};
