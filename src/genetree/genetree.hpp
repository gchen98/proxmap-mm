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

const int MAX_FEATURES = 100;
const int MAX_CHILDREN = 10;
  struct node_t{
    int annotation[MAX_FEATURES];
    node_t * children[MAX_CHILDREN];
    //node_t * left_child;
    //node_t * right_child;
    int row_index;
    node_t * parent;
    bool is_leaf;
    node_t(){
      parent = NULL;
      is_leaf = true;
      for(int i=0;i<MAX_CHILDREN;++i){
        children[i] = NULL;
      }  
    }
  };

class genetree_t:public proxmap_t{
public:
  void init(string configfile);
  void allocate_memory();
  genetree_t();
  ~genetree_t();
private:
  int genes; // total genes
  int features; // total features
  node_t ** node_arr;

  string annotation_file; // input matrix file of genes x features. 9=missing
  string mutation_file; // input matrix file of genes x features. 9=missing
  string ancestry_file; // input matrix file of genes x genes. 1=self or ancestor
  float * Y;  // observed annotation matrix
  float * X;  // imputed annotation matrix
  float * X_projection;  // projection of imputed annotation matrix
  float * M;  // observed mutation matrix
  float * mut; // imputed mutation matrix
  float * mut_projection; // projection of imputed mutation matrix
  float * A;  // gene tree encoded as ancestry matrix
  float * A_inv;  // gene tree encoded as ancestry matrix
  float * AX;  // gene tree encoded as ancestry matrix
  float * Z; // the surrogate parameter for observed annotations
  float * Q; // the surrogate parameter for observed mutations

  int rank_X; // tuning parameter: rank of X
  int max_mutations; // tuning parameter: max non-zeros in mu
  float last_x_norm;
  float last_mut_norm;
  float current_x_norm;
  float current_mut_norm;

  // GSL stuff
  gsl_matrix * gsl_A;
  gsl_matrix * gsl_mut_M;
  gsl_matrix * gsl_A_mut_M;
  gsl_matrix * gsl_U; // gene x feature
  gsl_vector * gsl_S; // features
  gsl_matrix * gsl_V; // feature x feature
  gsl_vector * gsl_svd_work; // features
  gsl_matrix * gsl_lowrank; // gene x feature
 
  void project_X(); // projection of X on low rank representation
  void project_mut(); // projection of mu on constraints of mutation count
  void update_X();
  void update_mut();
  void update_Z();
  void update_Q();

  void update_map_distance();


 // void init_opencl();
  void parse_config_line(string & key, istringstream & iss);
  void iterate();
  bool finalize_iteration();
  bool finalize_inner_iteration();
//  void finalize_iteration_gpu();
  float get_map_distance();
  float evaluate_obj();
//  void evaluate_obj_gpu();
  void print_output();
  bool in_feasible_region();
//  void update_weights_gpu();
//  void get_U_gpu();
//
//  bool init(const char * config_file);
//  //void init(float rho, const char * genofile, const char * weightsfile);
//  void coalesce();
//  void load_compact_geno(const char * genofile);
//  void load_compact_weights(const char * weightsfile);
//  void print_genetree(ostream & os);
//#ifdef USE_GPU
//  cl::Kernel * kernel_store_U_project;
//  cl::Kernel * kernel_store_U_project_prev;
//  cl::Kernel * kernel_init_U;
//  cl::Kernel * kernel_update_U;
//  cl::Kernel * kernel_update_map_distance;
//  cl::Kernel * kernel_init_v_project_coeff;
//  cl::Kernel * kernel_iterate_projection;
//  cl::Kernel * kernel_evaluate_obj;
//  cl::Kernel * kernel_get_U_norm_diff;
//  cl::Buffer * buffer_n_norms;
//  cl::Buffer * buffer_n2_norms;
//  cl::Buffer * buffer_dist_func;
//  cl::Buffer * buffer_rho;
//  cl::Buffer * buffer_unweighted_lambda;
//  cl::Buffer * buffer_U;
//  cl::Buffer * buffer_U_prev;
//  cl::Buffer * buffer_U_project;
//  cl::Buffer * buffer_U_project_orig;
//  cl::Buffer * buffer_U_project_prev;
//  cl::Buffer * buffer_V_project_coeff;
//  cl::Buffer * buffer_rawdata;
//  cl::Buffer * buffer_weights;
//  cl::Buffer * buffer_offsets;
//  cl::Buffer * buffer_variable_block_norms1;
//  cl::Buffer * buffer_variable_block_norms2;
//  cl::Buffer * buffer_subject_variable_block_norms;
//#endif
//  int variable_blocks;
//  int print_index;
//  bool coeff_defined;
//  float large;
//  float * sub_fnorm;
//  float * U;
//  float * U_prev;
//  float U_norm_diff;
//  float * U_project;
//  float * U_project_orig;
//  float * U_project_prev;
//  float * V_project_coeff;
//  float * rawdata;
//  float * weights;
//  int * offsets;
//  float * norm1_arr;
//  float * norm2_arr;
//  int n,n2,p;
//  int iter;
//  float current_vnorm,last_vnorm;
//  int triangle_dim;
//
  void initialize();
//  void initialize_gpu();
//  void load_into_triangle(const char * filename,float * & mat,int rows, int cols);
//
  float infer_epsilon();
  float infer_rho();
//  void init_v_project_coeff();
//  void init_v_project_coeff_gpu();
//  // if returns true, it is non zero
//  bool get_updated_v(int index1,int index2, float * v);
//  void store_U_projection_gpu();
//  void update_projection();
//  void update_projection_gpu();
//  void update_projection_nonzero();
//  void update_map_distance_gpu();
//  void update_u();
//  void update_u_gpu();
//  void check_constraint();
//
//  // QN acceleration  
//  float min_rho,max_rho;
  int get_qn_parameter_length();
  void get_qn_current_param(float * params);
  void store_qn_current_param(float * params);
  bool proceed_qn_commit();
  void infer_y_matrix();

  void read_sparse_matrix(const char * filename, float * mat, int rows, int cols);
  
  int get_annot(node_t * node,int feature);
  
//  int total_iter;
//  
};
