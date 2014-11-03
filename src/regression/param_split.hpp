#include<assert.h>
#include<set>
#ifdef USE_MPI
#include<mpi.h>
#endif
//#ifdef USE_GPU
//#include<ocl_wrapper.hpp>
//#endif

using namespace std;

class cross_validation_t;

class param_split_t:public proxmap_t{
public:
  param_split_t(bool single_run);
  ~param_split_t();
  void init(string configfile);
  void allocate_memory();
  friend class cross_validation_t;
private:
#ifdef USE_MPI
  MPI_Status mpi_stat;
  MPI_Datatype floatSubjectsArrayType;
  MPI_Datatype floatVariablesArrayType;
#endif

  // MPI related variables
  int mpi_rank;
  uint slaves;
  int slave_id;
  int mpi_numtasks;
  int mpi_rc;
  int * snp_node_sizes;
  int * snp_node_offsets;
  int * subject_node_sizes;
  int * subject_node_offsets;

  // flags
  bool single_run; // flag to know if this is embedded in cross validation
  int total_iterations;
  int current_top_k;
  bool track_residual;
  float last_residual;
  float current_BIC;
  float last_BIC;
  float last_mapdist;
  float residual;
  float landweber_constant;
  float current_mapdist_threshold;
  int BLOCK_WIDTH;

  // dimensions
  int sub_observations; // is all subjects for master, and subset for slaves
  int observations; // is all subjects for all nodes
  int variables; // is all snps for master, and subset for slaves
  int all_variables; // is all snps for all nodes
  

  float * all_y; // outcome same dimension on all nodes
  float * y; // dimension dependent variable for outcome
  float * Xbeta_full;

  float * means;
  float * precisions;
  float * beta; // variable dimension depending on node
  float * last_beta;// variable dimension depending on node
  float * beta_project;// variable dimension depending on node
  float * constrained_beta;// variable dimension depending on node

  float * all_beta; // same dimension on all nodes

  float * lambda; // defined on master with dimension observations
  float * theta; // these have length dependent on sub observations
  float * theta_project; // these have length dependent on sub observations

  float * XXI; // the cached data of (X^T %*% X + I)^-1

  // model selection containers
  float * grid_bic;
  float * grid_beta;

  // IO variables
  plink_data_t * plink_data_X_subset;
  packedgeno_t * packedgeno_snpmajor;
  int packedstride_snpmajor;
  plink_data_t * plink_data_X_subset_subjectmajor;
  packedgeno_t * packedgeno_subjectmajor;
  int packedstride_subjectmajor;
  random_access_t * random_access_XXI_inv;
  random_access_t * random_access_XXI;
#ifdef USE_GPU
  ocl_wrapper_t * ocl_wrapper;
#endif
  ofstream ofs_debug;
  // chunks for GPU workspaces
  int subject_chunks; // for SNP major
  int snp_chunks; // for subject major

  void parse_config_line(string & key, istringstream & iss);
  void read_dataset();
  inline float c2g(char c,int shifts);
  void parse_fam_file(const char * infile, bool * mask,int len,float * y);
  void parse_bim_file(const char * infile, bool * mask,int len,float * mean, float * sd);
  void init_xxi_inv();

  void update_constrained_beta();
  void iterate();
  bool finalize_iteration();
  bool finalize_inner_iteration();
  float get_map_distance();
  void update_map_distance();
  float evaluate_obj();
  bool in_feasible_region();
  void print_output();
  void initialize();
  void init_gpu();
  float infer_rho();
  float infer_epsilon();
  
  void project_theta();
  void project_beta();
  void update_theta();
  void update_beta();
  void update_lambda();
  // QN acceleration
  int get_qn_parameter_length();
  void get_qn_current_param(float * params);
  void store_qn_current_param(float * params);
  bool proceed_qn_commit();

  //float compute_marginal_beta(float * xvec);
  //void init_marginal_screen();
  //void compute_XX();
  //void load_matrix_data(const char *  filename, float * & mat, int in_rows, int in_cols,int out_rows, int out_cols, bool * row_mask, bool * col_mask, bool filerequired,float defaultVal);
  //void load_random_access_data(random_access_t *   random_access, float * & mat, int in_variables, int in_observations,int out_observations, int out_variables, bool * observation_mask, bool * variables_mask);
};

inline float param_split_t::c2g(char c,int shifts){
  int val = (static_cast<int>(c))>>(2*shifts) & 3;
  assert(val<4);
  return plink_data_t::plink_geno_mapping[val];
}
