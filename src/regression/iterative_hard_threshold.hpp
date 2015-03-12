#include<assert.h>
#include<set>
#include<map>
#ifdef USE_MPI
#include<mpi.h>
#endif
//#include<random_access.hpp>
//#include<plink_data.hpp>

using namespace std;

class cross_validation_t;


class iterative_hard_threshold_t:public proxmap_t{
public:
  iterative_hard_threshold_t(bool single_run);
  ~iterative_hard_threshold_t();
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
  float spectral_norm;
  float current_mapdist_threshold;
  int BLOCK_WIDTH;

  // dimensions
  int sub_observations; // is all subjects for master, and subset for slaves
  int observations; // is all subjects for all nodes
  int variables; // is all snps for master, and subset for slaves
  int all_variables; // is all snps for all nodes
  bool logistic;

  int train_n;
  int validation_n;
  float * X_active_train;
  float * Xt_active_train;
  float * X_active_validate;
  float * Xt_active_validate;
  float * XtY;
  float * XtXbeta;
  float * y; // dimension dependent variable for outcome
  float * residuals; // dimension dependent variable for outcome
  float * Xbeta_full;
  float * Xbeta_old;
  float * negative_gradient;

  float * means;
  float * precisions;
  float * gradient;
  int * mask_n;
  int * mask_p;

  float * beta_increment;
  float * temp_n;
  float * temp_p;
  // Convention for Conjugate Gradient is Ax=b
  float * newton_A1;
  float * newton_A1_inv;
  float *  cg_x0;
  float *  cg_x;
  float *  bd_all_y0;
  float  *  bd_all_y;

  //float * cg_b;
  //float * cg_x;
  float * A3x;
  float * cg_residuals;
  float * cg_conjugate_vec;
  float * cg_Ap;
  float * beta; // variable dimension depending on node
  float * beta_old;// variable dimension depending on node
  float * last_beta;// variable dimension depending on node
  float * constrained_beta;// variable dimension depending on node
  float gradient_weight;
  float frobenius_norm;

  // model selection containers
  float * grid_bic;
  float * grid_beta;
  int total_active;
  int last_total_active;
  int * last_active_indices;
  int * active_indices;
  int * inactive_indices;
  float * residual_weights;

  // IO variables
  //random_access_t * random_access_geno;
  plink_data_t * plink_data_X_subset;
  packedgeno_t * packedgeno_snpmajor;
  int packedstride_snpmajor;
  plink_data_t * plink_data_X_subset_subjectmajor;
  packedgeno_t * packedgeno_subjectmajor;
  int packedstride_subjectmajor;
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
  // in is p, out is n
  void update_Xbeta();
  void update_Xbeta(int * mask_n,int * mask_p);
  void compute_x_times_vector(float * in_vec,int * mask_n,int * mask_p,float * out_vec,bool debug); 
  // in is n, out is p
  void compute_xt_times_vector(float * in_vec,float * out_vec);
  void compute_xt_times_vector(float * in_vec,int * mask_n, int * mask_p,float * out_vec, float  scaler);

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
  void update_residual_weights(int * mask_n);
  
  void update_beta_iterative_hard_threshold();
  void update_beta_landweber();
  void update_beta_CG();
  // QN acceleration
  int get_qn_parameter_length();
  void get_qn_current_param(float * params);
  void store_qn_current_param(float * params);
  bool proceed_qn_commit();
  float compute_marginal_beta(float * xvec);
  void update_X_matrices();
  bool run_landweber;
  void compute_X_active_train_vector(float * vector_p,float * vector_n);
  void compute_X_active_validate_vector(float * vector_p,float * vector_n);

};

inline float iterative_hard_threshold_t::c2g(char c,int shifts){
  int val = (static_cast<int>(c))>>(2*shifts) & 3;
  assert(val<4);
  return plink_data_t::plink_geno_mapping[val];
}
