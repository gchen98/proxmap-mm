#ifdef USE_MPI
#include<mpi.h>
#endif

class cross_validation_t{
public:
  cross_validation_t();
  ~cross_validation_t();
  void allocate_memory(string config_file); 
  void run(string proxmap_config);
private:
  int rows,cols;
  int mpi_rank;
  int folds;
  int * fold_id;
  float * trained_betas;
  int min_k,max_k,k_increment;
  regression_t * regression;
  string genofile;
  string traitfile;
  string uid;

  void init_folds();
  void generate_train_test_files(config_t * & config,int fold_id);
  float eval_k(int k,vector<int>  & k_vec, vector<float> & mse_vec, string regression_config_file);
  float compute_mse();
};
