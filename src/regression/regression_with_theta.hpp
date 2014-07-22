#include<set>
#ifdef USE_MPI
#include<mpi.h>
#endif

using namespace std;

class cross_validation_t;

class regression_with_theta_t:public proxmap_t{
public:
  regression_with_theta_t(bool single_run);
  ~regression_with_theta_t();
  void init(string configfile);
  void allocate_memory();
  friend class cross_validation_t;
private:
#ifdef USE_MPI
  MPI_Status mpi_stat;
  MPI_Datatype floatSubjectsArrayType;
  MPI_Datatype floatVariablesArrayType;
#endif
  bool single_run;
  //static const float BETA_EPSILON = 1e-6;
  ofstream ofs_debug;
  bool * active_set;
  int active_set_size;
  int top_k;
  int mpi_rank;
  int slave_id;
  int mpi_numtasks;
  uint slaves;
  int mpi_rc;
  int * node_sizes;
  int * node_offsets;
  float beta_distance;
  float theta_distance;
  
  void read_dataset();
  float compute_marginal_beta(float * xvec);
  void init_marginal_screen();
  void init_xxi_inv();
  void parse_config_line(string & key, istringstream & iss);
  void iterate();
  bool finalize_iteration();
  bool finalize_inner_iteration();

  float get_map_distance();
  void update_map_distance();
  float evaluate_obj();
  bool in_feasible_region();
  void print_output();
  void initialize();
  float infer_rho();
  float infer_epsilon();
  


  float private_rho,private_epsilon;
  int total_iterations;
  int observations,variables;
  //int observations,n,variables,p;
  int blocks;
  float w;
  float maxr,maxc;
  float * XXI_inv;
  //float * singular_vectors;
  float * nu;
  float * beta_project;
  float * beta;
  float * last_beta;
  float * constrained_beta;
  float * theta_project;
  float * theta;
  float * lambda;
  float L;
  float * X;
  float * XT;
  float * XXTI;
  float * y;
  float * Xbeta;
  float last_residual;
  float residual;
  bool track_residual;
  
  void update_Xbeta(float * beta);
  void update_XXT();
  void project_beta();
  void project_theta();
  void update_theta();
  void update_beta();
  void update_lambda();
  void loss();
  void check_constraints();
  void load_matrix_data(const char *  filename, float * & mat, int in_rows, int in_cols,int out_rows, int out_cols, bool * row_mask, bool * col_mask, bool filerequired,float defaultVal);
  // QN acceleration
  int get_qn_parameter_length();
  void get_qn_current_param(float * params);
  void store_qn_current_param(float * params);
  bool proceed_qn_commit();

};

