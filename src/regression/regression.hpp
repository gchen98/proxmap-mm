#include<set>
#ifdef USE_MPI
#include<mpi.h>
#endif

using namespace std;

class regression_t:public proxmap_t{
public:
  regression_t();
  ~regression_t();
private:
#ifdef USE_MPI
  MPI_Status mpi_stat;
  MPI_Datatype floatSubjectsArrayType;
  MPI_Datatype floatVariablesArrayType;
#endif
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
  void init_xxi_inv();
  void allocate_memory(string configfile);
  void parse_config_line(string & key, istringstream & iss);
  void iterate();
  bool finalize_iteration();
  float get_map_distance();
  void update_map_distance();
  float evaluate_obj();
  bool in_feasible_region();
  void print_output();
  void initialize();
  float infer_rho();

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
  float * constrained_beta;
  float * theta_project;
  float * theta;
  float * lambda;
  float L;
  float * X;
  float * XT;
  float * XXT;
  float * XXTI;
  float * y;
  float * Xbeta;
  
  void update_Xbeta();
  void update_XXT();
  void project_beta();
  void project_theta();
  void update_theta();
  void update_beta();
  void update_lambda();
  void loss();
  void check_constraints();
  void load_matrix_data(const char *  filename, float * & mat, int in_rows, int in_cols,int out_rows, int out_cols, bool * row_mask, bool * col_mask, bool filerequired,float defaultVal);

};

