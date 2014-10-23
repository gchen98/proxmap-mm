#include<gsl/gsl_matrix.h>
#include<gsl/gsl_linalg.h>
#include<gsl/gsl_blas.h>
#include<gsl/gsl_errno.h>
#include"proxmap.hpp"

config_t::config_t(){
  //rho = 0;
  platform_id = 0;
  device_id = 0;
}

proxmap_t::proxmap_t(){
  config = new config_t;
}
  
proxmap_t::~proxmap_t(){
  if(config->enable_qn && qn_param_length){    
    delete[] qn_U;
    delete[] qn_V;    
    gsl_matrix_free(gsl_u);
    gsl_matrix_free(gsl_v);
    gsl_matrix_free(gsl_uv_delta);
    gsl_matrix_free(gsl_u_uv_delta);
    gsl_matrix_free(gsl_uuv_inverse);
    gsl_matrix_free(gsl_next_delta);
    gsl_matrix_free(gsl_part1);
    gsl_matrix_free(gsl_part2);
    gsl_matrix_free(gsl_part3);
    gsl_permutation_free(perm);
  }
  delete config;
}



void proxmap_t::parse_config_line(string & token,istringstream & iss){
//cerr<<"token: "<<token<<"\n";
  if (token.compare("USE_GPU")==0){
    iss>>config->use_gpu;
  }else if (token.compare("USE_CPU")==0){
    iss>>config->use_cpu;
  }else if (token.compare("PLATFORM_ID")==0){
    iss>>config->platform_id;
  }else if (token.compare("DEVICE_ID")==0){
    iss>>config->device_id;
  }else if (token.compare("KERNELS")==0){
    iss>>config->kernel_base;
  }else if (token.compare("VERBOSE")==0){
    iss>>config->verbose;
  }else if (token.compare("MAX_ITER")==0){
    iss>>config->max_iter;
  }else if (token.compare("ENABLE_QN")==0){
    iss>>config->enable_qn;
  }else if (token.compare("QN_SECANTS")==0){
    iss>>config->qn_secants;
  }else if (token.compare("GENOTYPES")==0){
    iss>>config->genofile;
  }else if (token.compare("OUTPUT_PATH")==0){
    iss>>config->output_path;
  }else if (token.compare("BURN_IN")==0){
    iss>>config->burnin;
  }else if (token.compare("RHO_MIN")==0){
    iss>>config->rho_min;
  }else if (token.compare("RHO_SCALE_FAST")==0){
    iss>>config->rho_scale_fast;
  }else if (token.compare("RHO_SCALE_SLOW")==0){
    iss>>config->rho_scale_slow;
  }else if (token.compare("RHO_MAX")==0){
    iss>>config->rho_max;
  }else if (token.compare("RHO_DISTANCE_RATIO")==0){
    iss>>config->rho_distance_ratio;
  }else if (token.compare("EPSILON_MAX")==0){
    iss>>config->epsilon_max;
  }else if (token.compare("EPSILON_SCALE_FAST")==0){
    iss>>config->epsilon_scale_fast;
  }else if (token.compare("EPSILON_SCALE_SLOW")==0){
    iss>>config->epsilon_scale_slow;
  }else if (token.compare("EPSILON_MIN")==0){
    iss>>config->epsilon_min;
  }else if (token.compare("OBJ_EPSILON")==0){
    iss>>config->obj_epsilon;
  }else if (token.compare("MAPDIST_EPSILON")==0){
    iss>>config->mapdist_epsilon;
  }else if (token.compare("MAPDIST_THRESHOLD")==0){
    iss>>config->mapdist_threshold;
  }else if (token.compare("MU_MIN")==0){
    iss>>config->mu_min;
  }else if (token.compare("MU_INCREMENT")==0){
    iss>>config->mu_increment;
  }else if (token.compare("MU_INCREMENTER")==0){
    iss>>config->mu_incrementer;
    if (config->mu_incrementer.compare("geometric")==0||config->mu_incrementer.compare("additive")==0){
      //we're OK
      //cerr<<"Using an incrementer style of "<<config->mu_incrementer<<endl;
    }else{
      cerr<<"Incrementer style of "<<config->mu_incrementer<<" invalid.  Use 'geometric' or 'additive'"<<endl;
      exit(1);
    }
  }else if (token.compare("MU_MAX")==0){
    iss>>config->mu_max;
  }
}

void proxmap_t::allocate_memory(){
  if(config->verbose)cerr<<"Allocating base class memory\n";
  if (config->enable_qn){
    qn_secants = config->qn_secants;
    qn_param_length = get_qn_parameter_length();
    if (qn_param_length){
      //cerr<<"QN parameter length is "<<qn_param_length<<endl;
      qn_U = new float[qn_param_length * qn_secants]; 
      qn_V = new float[qn_param_length * qn_secants]; 
      gsl_u = gsl_matrix_alloc(qn_param_length,qn_secants);
      gsl_v = gsl_matrix_alloc(qn_param_length,qn_secants);
      gsl_uv_delta = gsl_matrix_alloc(qn_param_length,qn_secants);
      gsl_u_uv_delta = gsl_matrix_alloc(qn_secants,qn_secants);  
      gsl_uuv_inverse = gsl_matrix_alloc(qn_secants,qn_secants);
      gsl_next_delta = gsl_matrix_alloc(qn_param_length,1);
      gsl_part1 = gsl_matrix_alloc(qn_param_length,qn_secants);
      gsl_part2 = gsl_matrix_alloc(qn_secants,1);
      gsl_part3 = gsl_matrix_alloc(qn_param_length,1);
      perm = gsl_permutation_alloc(qn_secants);
    }
  }
}

void proxmap_t::init(string config_file){
  ifstream ifs_config(config_file.data());
  if (!ifs_config.is_open()){
    cerr<<"Cannot open the file "<<config_file<<".\n";
    throw "IO error";
  }
  config->burnin = 0;
  config->obj_epsilon = 1e-4;
  config->mapdist_epsilon = 1e-3;
  config->enable_qn = false;
  config->qn_secants = 1;
  config->verbose = false;
  config->max_iter = static_cast<int>(1e5);
  config->epsilon_min = 1e-4;
  config->mu_min = .5;
  config->mu_max = 1e10;
  config->mu_increment = .5;
  config->mu_incrementer = "additive";
  config->rho_distance_ratio = 100;
  config->output_path=".";
  string line,token,val;
  while (getline(ifs_config,line)){
    istringstream iss(line);
    iss>>token;
    parse_config_line(token,iss);
  }
  ifs_config.close();
  run_gpu = config->use_gpu;
  run_cpu = config->use_cpu;

}

int proxmap_t::colcount(const char * filename){
  ifstream ifs(filename);
  if (!ifs.is_open()){
    cerr<<"Cannot open "<<filename<<" for line count\n";
    exit(1);
  }
  string line,token;
  int count = 0;
  getline(ifs,line);
  istringstream iss(line);
  while(iss>>token){
    ++count;
  }
  ifs.close();
  return count;
}

int proxmap_t::linecount(const char * filename){
  ifstream ifs(filename);
  if (!ifs.is_open()){
    cerr<<"Cannot open "<<filename<<" for line count\n";
    exit(1);
  }
  string line;
  int count = 0;
  while(getline(ifs,line)){
    ++count;
  }
  ifs.close();
  return count;
}

void proxmap_t::load_into_matrix(const char * filename,int * & mat,int rows, int cols){
  ifstream ifs(filename);
  if (!ifs.is_open()){
    cerr<<"Cannot open "<<filename<<" for matrix load.\n";
    exit(1);
  }
  string line;
  for(int i=0;i<rows;++i){
    getline(ifs,line);
    istringstream iss(line);
    for(int j=0;j<cols;++j){
      iss>>mat[i*cols+j];
    }
  }
  ifs.close();
}

void proxmap_t::load_into_matrix(const char * filename,float * & mat,int rows, int cols){
  ifstream ifs(filename);
  if (!ifs.is_open()){
    cerr<<"Cannot open "<<filename<<" for matrix load.\n";
    exit(1);
  }
  string line;
  for(int i=0;i<rows;++i){
    getline(ifs,line);
    istringstream iss(line);
    for(int j=0;j<cols;++j){
      iss>>mat[i*cols+j];
    }
  }
  ifs.close();
}

float proxmap_t::norm(float * mat, int size){
  float norm = 0;
  for(int i=0;i<size;++i){
    norm+=mat[i]*mat[i];   
    //cerr<<i<<": val is "<<mat[i]<<endl;
  }
  return sqrt(norm);
}



void proxmap_t::mmultiply(float *  a,int a_rows, int a_cols, float *  c){
  for(int outrow = 0;outrow<a_rows;++outrow){
    float tempvec[a_cols];
    for(int i=0;i<a_cols;++i) tempvec[i] = a[outrow*a_cols+i];
    for(int outcol = 0;outcol<a_rows;++outcol){
      c[outrow*a_rows+outcol]=0;
      for(int i=0;i<a_cols;++i){
        c[outrow*a_rows+outcol]+=tempvec[i]*a[outcol*a_cols+i];
      }
    }
  }

}

void proxmap_t::mmultiply(float *  a,int a_rows, int a_cols, float *  b,int b_cols, float *  c){
  for(int i=0;i<a_rows;++i){
    for(int k=0;k<b_cols;++k){
      c[i*b_cols+k] = 0;
      for(int j=0;j<a_cols;++j){
        c[i*b_cols+k] += a[i*a_cols+j] * b[j*b_cols+k];
      }
    }
  }
}

//void proxmap_t::handle_gsl_error(int status, const char * comment){
//  if (status) {
//    cerr<<"Status code was "<<status<<" from "<<comment<<endl; 
//  }
//}

void dump_mat(gsl_matrix* mat){
  for(uint i=0;i<mat->size1;++i){
    for(uint j=0;j<mat->size2;++j){
      if(j) cerr<<" ";
      cerr<<gsl_matrix_get(mat,i,j);
    }
    cerr<<endl;
  }
}
int proxmap_t::pseudo_inverse(gsl_matrix * mat,gsl_matrix * mat_inv){
  int status = 0;
  //int rows = mat->size1;
  int cols = mat->size2;
  gsl_matrix * tempv = gsl_matrix_alloc(cols,cols);
  gsl_vector * temps = gsl_vector_alloc(cols);
  gsl_vector * work  = gsl_vector_alloc(cols);
  //cerr<<"A\n";
  //dump_mat(mat);
  status = gsl_linalg_SV_decomp(mat,tempv,temps,work);
  if(status!=0) cerr<<"GSL SVD failed\n";
  //cerr<<"U\n";
  //dump_mat(mat);
  //cerr<<"V\n";
  //dump_mat(tempv);
  //cerr<<"S:"<<endl;
  double s[cols];
  for(int i=0;i<cols;++i){
    //cerr<<" "<<gsl_vector_get(temps,i);
    s[i] = gsl_vector_get(temps,i);
    if(s[i]>.00001)  s[i] = 1./s[i];
  }
  //cerr<<endl;
  for(int i=0;i<cols;++i){
    for(int j=0;j<cols;++j){
      gsl_matrix_set(tempv,i,j,s[j] * gsl_matrix_get(tempv,i,j));
    }
  }
  gsl_blas_dgemm(CblasNoTrans,CblasTrans,1.0,tempv,mat,0,mat_inv);
  //cerr<<endl;
  gsl_vector_free(work);
  gsl_vector_free(temps);
  gsl_matrix_free(tempv);
  return status;
}

void proxmap_t::invert(float * mat, float * outmat,int rows, int cols){
  gsl_matrix * gsl_mat = gsl_matrix_alloc(rows,cols);
  for(int i=0;i<rows;++i){
    for(int j=0;j<cols;++j){
      //if (j) cerr<<" ";
      //cerr<<mat[i*cols+j];
      gsl_matrix_set(gsl_mat,i,j,mat[i*cols+j]);
    }
    //cerr<<endl;
  }
  gsl_matrix * gsl_mat_inv = gsl_matrix_alloc(rows,cols);
  invert(gsl_mat, gsl_mat_inv);
  for(int i=0;i<rows;++i){
    for(int j=0;j<cols;++j){
      outmat[i*cols+j] = gsl_matrix_get(gsl_mat_inv,i,j);
    }
  }
  gsl_matrix_free(gsl_mat);
  gsl_matrix_free(gsl_mat_inv);
}

void proxmap_t::invert(gsl_matrix * gsl_mat, gsl_matrix * gsl_mat_inv){
  //int rows = gsl_mat->size1;
  int cols = gsl_mat->size2;
  gsl_permutation * perm = gsl_permutation_alloc(cols);
  invert(gsl_mat, gsl_mat_inv, perm);
  gsl_permutation_free(perm);
}

void proxmap_t::invert(gsl_matrix * gsl_mat, gsl_matrix * gsl_mat_inv, gsl_permutation * perm){
  int rows = gsl_mat->size1;
  int cols = gsl_mat->size2;
  gsl_set_error_handler_off();
  int signum;
  double orig[rows*cols];
  for(int i=0;i<rows;++i){
    for(int j=0;j<cols;++j){
      orig[i*cols+j] = gsl_matrix_get(gsl_mat,i,j);
    }
  }
  int status1 =  gsl_linalg_LU_decomp(gsl_mat,perm, &signum);
  //if (config->verbose) cerr<<"Signum: "<<signum<<endl;
  int status2 = gsl_linalg_LU_invert(gsl_mat,perm, gsl_mat_inv);
  //cerr<<"Status "<<status1<<" "<<status2<<endl;
  int status3 = 0;
  if (status1!=0 || status2!=0){
  //if (1==1){
    cerr<<"Error in LU decomp/inversion, attempting pseudoinverse\n";
    for(int i=0;i<rows;++i){
      for(int j=0;j<cols;++j){
        //if (j) cerr<<" ";
        //cerr<<mat[i*cols+j];
        gsl_matrix_set(gsl_mat,i,j,orig[i*cols+j]);
      }
      //cerr<<endl;
    }
    //int signum;
    status3 = pseudo_inverse(gsl_mat,gsl_mat_inv);
  }else{
    //cerr<<"Inversion OK\n";
  }
  //gsl_set_error_handler(NULL);
  if(status3){
    cerr<<"Pseudo inverse failed. Exiting.\n";
    exit(1);
  }
}


void proxmap_t::get_qn_proposed_param(float * current_param, float * next_param, float * proposed_param){

  // FORMULA is PROPOSED = NEXT - V %*% (U^T %*% (U-V))^-1 %*% U^T %*% (CURRENT - NEXT)
  //gsl_matrix * gsl_u = gsl_matrix_alloc(qn_param_length,qn_secants);
  //gsl_matrix * gsl_v = gsl_matrix_alloc(qn_param_length,qn_secants);
  //gsl_matrix * gsl_uv_delta = gsl_matrix_alloc(qn_param_length,qn_secants);
  
  // prepare the term (U-V)  
  for(int j=0;j<qn_secants;++j){
    for(int i=0;i<qn_param_length;++i){   
      gsl_matrix_set(gsl_u,i,j,qn_U[i*qn_secants+j]);
      gsl_matrix_set(gsl_v,i,j,qn_U[i*qn_secants+j]);
      gsl_matrix_set(gsl_uv_delta,i,j,qn_U[i*qn_secants+j] - qn_V[i*qn_secants+j]);
    }
  }
  // do the inverse of (U^T %*% (U-V))
  //gsl_matrix * gsl_u_uv_delta = gsl_matrix_alloc(qn_secants,qn_secants);  
  gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,gsl_u,gsl_uv_delta,0,gsl_u_uv_delta);
  //gsl_matrix * gsl_uuv_inverse = gsl_matrix_alloc(qn_secants,qn_secants);
  //gsl_permutation * perm = gsl_permutation_alloc(qn_secants);
  invert(gsl_u_uv_delta, gsl_uuv_inverse,perm);
  //int s;
  //gsl_linalg_LU_decomp(gsl_u_uv_delta,perm, &s);
  //gsl_linalg_LU_invert(gsl_u_uv_delta,perm, gsl_uuv_inverse);

  // prepare the last difference term
  //gsl_matrix * gsl_next_delta = gsl_matrix_alloc(qn_param_length,1);
  for(int i=0;i<qn_param_length;++i){
    gsl_matrix_set(gsl_next_delta,i,0,current_param[i]-next_param[i]);
  }
  // compute part 1 as V %*% (U^T %*% (U-V))^-1
  //gsl_matrix * gsl_part1 = gsl_matrix_alloc(qn_param_length,qn_secants);
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,gsl_v,gsl_uuv_inverse,0,gsl_part1);
  // compute part 2 as U^T %*% (CURRENT - NEXT)
  //gsl_matrix * gsl_part2 = gsl_matrix_alloc(qn_secants,1);
  gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,gsl_u,gsl_next_delta,0,gsl_part2);
  // compute part 3 as part 1 %*% part 2
  //gsl_matrix * gsl_part3 = gsl_matrix_alloc(qn_param_length,1);
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,gsl_part1,gsl_part2,0,gsl_part3);
  // compute part 4 as next - part 3
  for(int i=0;i<qn_param_length;++i){
    proposed_param[i] = next_param[i] - gsl_matrix_get(gsl_part3,i,0);
  }
  bool debug = false;
  if (debug){
    ofstream ofs_U("debug_U.txt");
    ofstream ofs_V("debug_V.txt");
    ofstream ofs_current("debug_current.txt");
    ofstream ofs_next("debug_next.txt");
    ofstream ofs_proposed("debug_proposed.txt");
    for(int i=0;i<qn_param_length;++i){   
      for(int j=0;j<qn_secants;++j){
        if(j){
          ofs_U<<"\t";
          ofs_V<<"\t";
        }
        ofs_U<< qn_U[i*qn_secants+j];
        ofs_V<< qn_V[i*qn_secants+j];
      }
      ofs_U<<endl; ofs_V<<endl;
      ofs_current<<current_param[i]<<endl;
      ofs_next<<next_param[i]<<endl;
      ofs_proposed<<proposed_param[i]<<endl;
    }
    ofs_U.close();
    ofs_V.close();
    ofs_current.close();
    ofs_next.close();
    ofs_proposed.close();
    exit(0);
  }
}

float proxmap_t::iterate_with_obj(){
  rho = infer_rho();
  epsilon = infer_epsilon();
  iterate();
  return evaluate_obj();
}

void proxmap_t::run(){
  
  float mu_min = config->mu_min;
  float mu_increment = config->mu_increment;
  float mu_max = config->mu_max;
  if (mu_increment==0) ++mu_increment;
  if (mu_max==mu_min) ++mu_max;
  iter_mu = 0;
  mu = 0;
  bool verbose = config->verbose;
  float * qn_last_param = NULL;
  float * qn_current_param = NULL;
  float * qn_next_param= NULL;
  float * qn_proposed_param= NULL;
  if (config->enable_qn && qn_param_length>0){
    qn_last_param = new float[qn_param_length];
    qn_current_param = new float[qn_param_length];
    qn_next_param = new float[qn_param_length];
    qn_proposed_param = new float[qn_param_length];
  }
  bool proceed_mu = true;
  do{ // loop over mu
    rho_distance_ratio = config->rho_distance_ratio;
    //epsilon = 0.1;
    initialize();
    //cerr<<"Verbosity: "<<config->verbose<<endl;
    //rho = rho_min;
    iter_rho_epsilon = 0;
    bool converged = false;
    last_obj=1e10;
    int burnin = config->burnin;
    int max_iter = config->max_iter;
    if(config->verbose) cerr<<"Max iterations: "<<max_iter<<endl;
    bool qn_matrices_init= false;
    bool qn_last_iterate_valid = false;  
    int qn_column_counter = 0;
    int current_secant = 0;
    float qn_obj_improvement = 0;
    float obj_improvement = 0;
    float qn_acceptance = 0;
    int qn_iterates = 0;
    bool proceed_inner = true;
    bypass_downhill_check = false;
    while(proceed_inner && !converged && iter_rho_epsilon<max_iter){
      if(config->verbose)cerr<<"Inner iterate: "<<iter_rho_epsilon<<endl;
      if (iter_rho_epsilon>=burnin){
        obj = iterate_with_obj(); 
        if(obj<last_obj){
          if (config->enable_qn ){
            get_qn_current_param(qn_current_param);
            if (qn_matrices_init){
              // do the QN update
              // output contents for debug
              bool debug_qn1 = false;
              if(debug_qn1){
                cerr<<"Let's do a QN update!\n";
                cerr<<"U contents:\n";
                for(int i=0;i<qn_param_length;++i){
                  for(int j=0;j<qn_secants;++j){
                    if(j) cerr<<" ";
                    cerr<<qn_U[i*qn_secants+j];
                  }
                  cerr<<endl;
                }
                cerr<<"V contents:\n";
                for(int i=0;i<qn_param_length;++i){
                  for(int j=0;j<qn_secants;++j){
                    if(j) cerr<<" ";
                    cerr<<qn_V[i*qn_secants+j];
                  }
                  cerr<<endl;
                }
              }
              // do one proposed iteration
              float next_obj = iterate_with_obj();
              if (next_obj<obj){
                get_qn_current_param(qn_next_param);
                if (qn_param_length) get_qn_proposed_param(qn_current_param, qn_next_param, qn_proposed_param);
                store_qn_current_param(qn_proposed_param);
                ++qn_iterates;
                int col_index = qn_column_counter % qn_secants;
                //cerr<<"Updating QN matrix col "<<col_index<<endl;
                for(int i = 0;i<qn_param_length;++i){
                  qn_U[i*qn_secants+col_index] = qn_current_param[i] - qn_last_param[i];              
                }              
                float proposed_obj = iterate_with_obj();
                // make sure the QN improvement is not too ambitious!
                if (proposed_obj < next_obj && proceed_qn_commit()){
                  if (verbose)cerr<<"Proposed QN objective of "<<proposed_obj<<" is lower than standard obj "<<next_obj<<endl;
                  qn_obj_improvement+=(obj-proposed_obj);
                  obj_improvement+=(obj-proposed_obj);
                  ++qn_acceptance;
                  // update the entries in the U and V matrix              
                
                  // update the current_param
                  for(int i=0;i<qn_param_length;++i){
                    qn_V[i*qn_secants+col_index] = qn_proposed_param[i] - qn_current_param[i];
                    qn_current_param[i] = qn_proposed_param[i];
                  }
                  obj = proposed_obj;
                }else{
                  if(verbose) cerr<<"Proposed QN objective of "<<proposed_obj<<" is higher than standard obj "<<next_obj<<" or the commit failed!"<<endl;
                    //if(proposed_obj<next_obj) cerr<<"Jump was "<< (next_obj-proposed_obj)/next_obj<<endl;
                  //}
                  // rollback parameter values
                  store_qn_current_param(qn_next_param);
                  // update the current_param
                  for(int i=0;i<qn_param_length;++i){
                    qn_V[i*qn_secants+col_index] = qn_next_param[i] - qn_current_param[i];                  
                    qn_current_param[i] = qn_next_param[i];
                  }
                  float backtracked_obj = iterate_with_obj();
                  obj_improvement+=(obj-backtracked_obj);
                }
              
              //for(int i = 0;i<qn_param_length;++i){
                //qn_U[i*qn_secants+col_index] = qn_current_param[i] - qn_last_param[i];
                //qn_V[i*qn_secants+col_index] = qn_proposed_param[i] - qn_current_param[i];                  
              //}              
                ++qn_column_counter;
              }else{
                //cerr<<"QN next objective is not lower, skipping\n";
              }
            }else{ // still working on initializing the U and V matrices
              // initialize the U matrix
              if (qn_last_iterate_valid){
                if (current_secant < qn_secants){
                  for(int i=0;i<qn_param_length;++i){
                    //qn_delta_param[i] = 
                    qn_U[i*qn_secants+current_secant] = qn_current_param[i] - 
                    qn_last_param[i];
                  }
                }else if (current_secant==qn_secants){
                  for(int i=0;i<qn_param_length;++i){
                    //qn_delta_param[i] = 
                    qn_V[i*qn_secants+current_secant-1] = qn_current_param[i] - 
                    qn_last_param[i];
                  }
                  for(int i=0;i<qn_param_length;++i){
                    for(int j=0;j<qn_secants-1;++j){
                      //qn_delta_param[i] = 
                      qn_V[i*qn_secants+j] = qn_U[i*qn_secants+j+1];
                    }
                  }
                  qn_matrices_init = true;
                }
                ++current_secant; 
              }
            }
            for(int i=0;i<qn_param_length;++i){
              qn_last_param[i] = qn_current_param[i];
            }
            qn_last_iterate_valid = true;
          } // end if QN accelerated
          if (bypass_downhill_check){
            if (verbose){
              cerr<<"Bypassing downhill check! (last: "<<last_obj<<" current: "<<obj<<endl;
            }
          //}else if(last_obj<=obj){
          }else if(fabs(last_obj-obj)/last_obj<config->obj_epsilon){
            converged=true; 
            if (verbose)cerr<<"Converged!\n";
          }else{
            if (verbose){
              cerr<<"Proceeding to next iteration! (last: "<<last_obj<<" current: "<<obj<<")"<<endl;
              //print_output();
            }
          }
        }else{
          converged = true;
          if (verbose)cerr<<"Objective function is not changing or going uphill from "<<last_obj<<" to "<<obj<<", aborting.\n";
        } // if (obj < last_obj)
      }else{
        if (verbose)cerr<<"Ignoring objective on burnin iteration "<<iter_rho_epsilon<<"\n";
      }
      last_obj = obj;
      last_rho = rho;
      last_epsilon = epsilon;
      ++iter_rho_epsilon;
      proceed_inner = finalize_inner_iteration();
    }
    if(config->enable_qn && qn_param_length && qn_acceptance>0){
      cerr<<"QN acceptance ratio: "<<(qn_acceptance/qn_iterates*1.)<<" QN improvement: "<<qn_obj_improvement<<"/"<<obj_improvement<<"="<<(qn_obj_improvement/obj_improvement)<<endl;
    }

    proceed_mu = finalize_iteration();
    print_output();
    last_mu = mu;
    if (mu==0) mu = mu_min;
    else {
      if(config->mu_incrementer.compare("additive")==0){
        mu+=mu_increment;
      }else if(config->mu_incrementer.compare("geometric")==0){
        mu*=mu_increment;
      }else{
        mu = mu_max+1;
      }
    }
    ++iter_mu;
  }while(proceed_mu && mu<=mu_max);
  
  if (config->enable_qn && qn_param_length>0){
    delete[] qn_last_param;
    delete[] qn_current_param;
    delete[] qn_next_param;
    delete[] qn_proposed_param;
  }
}

float proxmap_t::get_prox_dist_penalty(){
  float mapdist = get_map_distance();
  float penalty = (.5*rho*mapdist)/(sqrt(mapdist+epsilon));
  //cerr<<"DEBUG rho "<<rho<<" mapdist "<<mapdist<<" epsilon "<<epsilon<<endl;
  if(config->verbose) cerr<<"PROXDIST_PEN:"<<"rho:"<<rho<<",mapdist:"<<mapdist<<",epsilon:"<<epsilon<<",penalty:"<<penalty<<endl;
  return penalty;
}
