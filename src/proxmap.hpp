using namespace std;
#include<iostream>
#include<fstream>
#include<sstream>
#include<cstdlib>
#include<math.h>
#include<gsl/gsl_matrix.h>
#include<gsl/gsl_linalg.h>
//#include<random_access.hpp>
//#include<plink_data.hpp>

#ifdef USE_GPU
#include<CL/cl.h>
#include<CL/cl.hpp>
//#include<ocl_wrapper.hpp>
#endif


struct config_t{
  config_t();
  int burnin;
  bool verbose;
  float rho_min; // proximal distance parameter
  float rho_scale_fast; // proximal distance parameter
  float rho_scale_slow; // proximal distance parameter
  float rho_max;
  float epsilon_max;
  float epsilon_scale_fast; // proximal distance parameter
  float epsilon_scale_slow; // proximal distance parameter
  float epsilon_min;
  float rho_distance_ratio;
  float obj_epsilon;
  float mapdist_epsilon;
  float mapdist_threshold;
  int max_iter;

  float mu_min;  // penalty parameter
  float mu_increment;  // penalty parameter
  string mu_incrementer; // geometric/additive
  float mu_max;  // penalty parameter

  bool use_gpu;
  bool use_cpu;
  bool enable_qn;
  int qn_secants;
  int platform_id;
  int device_id;
  string kernel_base;
  string genofile;
  string output_path;

  // convex cluster settings
  string weightsfile;
  int datapoints;
  int variables;
  float u_delta_rho_cap;
  float print_threshold;
  string geno_format;
  string geno_order;

  // regression settings
  //int total_observations;
  //int total_variables;
  //float frobenius_norm;
  float beta_epsilon;
  float spectral_norm;
  int max_landweber;
  string bin_geno_file;
  string xxi_file;
  string xxi_inv_file;
  string snp_bed_file;
  //string subject_bed_file;
  string bim_file;
  string fam_file;
  //float nu;  // penalty parameter
  int top_k_min;
  int top_k_max;
  bool debug_mpi;
  bool single_run;
};


class proxmap_t{
public:
  proxmap_t();
  virtual void init(string configfile);
  virtual void allocate_memory();
  void run();
  virtual ~proxmap_t();

  //utility functions
  
  //void handle_gsl_error(int status,const char * comment);
  static void invert(float * mat, float * outmat,int rows, int cols);
  static void invert(gsl_matrix * mat, gsl_matrix * outmat);
  static void invert(gsl_matrix * mat, gsl_matrix * outmat, gsl_permutation * perm);
  static int colcount(const char * filename);
  static int linecount(const char * filename);
  static void  load_into_matrix(const char * filename,float * & mat,int rows, int cols);
  static void  load_into_matrix(const char * filename,int * & mat,int rows, int cols);
//  template<typename T,int size> static void load_into_matrix(const char * filename,T(&)[size],int rows, int cols);
  static void mmultiply(float *  a,int a_rows, int a_cols, float *  b,int b_cols, float *  c);
  static void mmultiply(float *  a,int a_rows, int a_cols, float *  c);
  static float norm(float * mat,int size);
protected:
  // utility functions
  virtual void parse_config_line(string & key, istringstream & iss);
  // proximal distance functions
  virtual void initialize()=0;
  virtual float get_map_distance()= 0;
  virtual void update_map_distance()= 0;
  virtual bool in_feasible_region()=0;
  virtual float evaluate_obj()= 0;
  virtual void print_output()=0;
  virtual void iterate()=0;
  virtual bool finalize_inner_iteration()=0;
  virtual bool finalize_iteration()=0;
  virtual float infer_rho()=0;
  virtual float infer_epsilon()=0;
  float get_prox_dist_penalty();


  config_t * config;
  float last_mu,last_rho,rho,last_epsilon,epsilon,rho_distance_ratio;
  float map_distance;
  float dist_func;
  float mu,obj,last_obj;
  int iter_mu,iter_rho_epsilon;
  bool run_gpu;
  bool run_cpu;
  bool bypass_downhill_check;

  // for Quasi Newton acceleration
  //
  float * qn_U;
  float * qn_V;

  int qn_param_length;
  int qn_secants;
  virtual int get_qn_parameter_length()=0;
  virtual void get_qn_current_param(float * params)=0;
  virtual void store_qn_current_param(float * params)=0;
  virtual bool proceed_qn_commit() = 0;
  
#ifdef USE_GPU
  virtual void init_opencl();
  bool debug_opencl;
  cl_int err;
  cl::Context * context;
  cl::CommandQueue * commandQueue;
  cl::Program * program;
  vector<cl::Device> devices;
  void createKernel(const char * name, cl::Kernel * & kernel);
  void runKernel(const char * name, cl::Kernel * & kernel,int wg_x,int wg_y, int wg_z, int wi_x,int wi_y, int wi_z);
  template<class T> void createBuffer(int rw,int dim,const char * label,cl::Buffer * & buf);
  template<typename T> void setArg(cl::Kernel * & kernel,int & index,T arg,const char * label);
  template<typename T> void writeToBuffer(cl::Buffer * & buffer,int dim,T hostArr,const char * label);
  template<typename T> void readFromBuffer(cl::Buffer * & buffer,int dim,T hostArr,const char * label);
#endif
private:
  gsl_matrix * gsl_u;
  gsl_matrix * gsl_v;
  gsl_matrix * gsl_uv_delta;
  gsl_matrix * gsl_u_uv_delta;
  gsl_matrix * gsl_uuv_inverse;
  gsl_matrix * gsl_next_delta;
  gsl_matrix * gsl_part1;
  gsl_matrix * gsl_part2;
  gsl_matrix * gsl_part3;
  gsl_permutation * perm;
  static int pseudo_inverse(gsl_matrix * mat,gsl_matrix * mat_inv);
  void get_qn_proposed_param(float * current_param, float * next_param, float * proposed_param);
  float iterate_with_obj();
};
