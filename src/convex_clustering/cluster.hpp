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
  void init_opencl();
  void parse_config_line(string & key, istringstream & iss);
  void iterate();
  void finalize_iteration();
  float get_map_distance();
  float evaluate_obj();
  void evaluate_obj_gpu();
  void print_output();
  bool in_feasible_region();
  void update_weights_gpu();
  void get_U_gpu();

  bool init(const char * config_file);
  //void init(float rho, const char * genofile, const char * weightsfile);
  void coalesce();
  void load_compact_geno(const char * genofile);
  void load_compact_weights(const char * weightsfile);
  void print_cluster(ostream & os);
#ifdef USE_GPU
  cl::Kernel * kernel_store_U_project;
  cl::Kernel * kernel_store_U_project_prev;
  cl::Kernel * kernel_init_U;
  cl::Kernel * kernel_update_U;
  cl::Kernel * kernel_update_map_distance;
  cl::Kernel * kernel_init_v_project_coeff;
  cl::Kernel * kernel_iterate_projection;
  cl::Kernel * kernel_evaluate_obj;
  cl::Buffer * buffer_n_norms;
  cl::Buffer * buffer_n2_norms;
  cl::Buffer * buffer_dist_func;
  cl::Buffer * buffer_rho;
  cl::Buffer * buffer_unweighted_lambda;
  cl::Buffer * buffer_U;
  cl::Buffer * buffer_U_project;
  cl::Buffer * buffer_U_project_orig;
  cl::Buffer * buffer_U_project_prev;
  cl::Buffer * buffer_V_project_coeff;
  cl::Buffer * buffer_rawdata;
  cl::Buffer * buffer_weights;
  cl::Buffer * buffer_offsets;
  cl::Buffer * buffer_variable_block_norms1;
  cl::Buffer * buffer_variable_block_norms2;
  cl::Buffer * buffer_subject_variable_block_norms;
#endif
  int variable_blocks;
  int print_index;
  bool coeff_defined;
  float map_distance;
  float dist_func;
  float large;
  float * sub_fnorm;
  float * U;
  float * U_prev;
  float * U_project;
  float * U_project_orig;
  float * U_project_prev;
  float * V_project_coeff;
  float * rawdata;
  float * weights;
  int * offsets;
  float * norm1_arr;
  float * norm2_arr;
  float mu;
  int n,n2,p;
  int iter;
  float current_vnorm,last_vnorm;
  int triangle_dim;

  void initialize(float mu);
  void initialize_gpu(float mu);
  void load_into_triangle(const char * filename,float * & mat,int rows, int cols);

  float infer_rho();
  void init_v_project_coeff();
  void init_v_project_coeff_gpu();
  // if returns true, it is non zero
  bool get_updated_v(int index1,int index2, float * v);
  void store_U_projection_gpu();
  void update_projection();
  void update_projection_gpu();
  void update_projection_nonzero();
  void update_map_distance();
  void update_map_distance_gpu();
  void update_u();
  void update_u_gpu();
  void check_constraint();
  
};
