#include<set>
#include<gsl/gsl_blas.h>
#include"../cl_constants.h"
#include"../proxmap.hpp"
#include"genetree.hpp"

struct mut_t{
  int index;
  float val;
  float absval;

  mut_t(int index,float val){
    this->index = index;
    this->val = val;
    this->absval = fabs(val);
  };
};

struct byValDesc{
  bool operator()(const mut_t & a,const mut_t & b){
    return a.absval>b.absval;
  }
};

struct byIndexAsc{
  bool operator()(const mut_t & a,const mut_t & b){
    if(a.absval==b.absval){
      return a.index<b.index;
    }else{
      return a.absval>b.absval;
    }
  }
};


genetree_t::genetree_t(){
}

genetree_t::~genetree_t(){
  for(int i=0;i<genes;++i){
    delete node_arr[i];
  }
  delete [] node_arr;
}

void genetree_t::parse_config_line(string & token,istringstream & iss){
  proxmap_t::parse_config_line(token,iss);
  if (token.compare("GENES")==0){
    iss>>genes;
  }else if (token.compare("FEATURES")==0){
    iss>>features;
  }else if (token.compare("ANNOTATION_FILE")==0){
    iss>>annotation_file;
  }else if (token.compare("MUTATION_FILE")==0){
    iss>>mutation_file;
  }else if (token.compare("ANCESTRY_FILE")==0){
    iss>>ancestry_file;
  }else if (token.compare("RANK_X")==0){
    iss>>rank_X;
  }else if (token.compare("MAX_MUTATIONS")==0){
    iss>>max_mutations;
  }
}

void genetree_t::init(string config_file){
  proxmap_t::init(config_file);
}



int genetree_t::get_annot(node_t * node,int feature){
  if(config->verbose) cerr<<"Feature "<<feature<<" arriving at node "<<node->row_index<<endl;
  //bool missing = true;
  int total_annot = 0;
  int total_children = 0;
  int total_null = 0;
  int total_func = 0;
  for(int child=0;child<MAX_CHILDREN;++child){
    node_t * child_node = node->children[child];
    if(child_node!=NULL){
      int annot = get_annot(child_node,feature);
      if(annot!=9) {
        total_null+=annot==0;
        total_func+=annot==1;
      }
      ++total_children;
    }
  }
  if(node->is_leaf) {
    if (config->verbose) cerr<<"At leaf returning "<<node->annotation[feature]<<"\n";
    return (node->annotation[feature]);
    //return (node->annotation[feature]==1);
  }else{
    //if (total_annot>19){
    //if ((total_leaves==total_children && total_annot>0) || total_annot==total_children){
    if(1.*total_func/total_children>=.5) node->annotation[feature] = 1;
    else if(1.*total_null/total_children>=.5) node->annotation[feature] = 0;
    else node->annotation[feature] = 9;
    int ret = node->annotation[feature];
    //int ret = total_annot;
    //if (missing){
      //node->annotation[feature] = 9;
    //}else{
      //node->annotation[feature] = 1;
    //}
    if (config->verbose) cerr<<"At internal for "<<node->row_index<<" with total_annot "<<total_annot<<" and node annot: "<<node->annotation[feature]<<" returning "<<ret<<"\n";
    return ret;
  }
}

void genetree_t::infer_y_matrix(){
  for(int i=0;i<genes;++i){
    node_arr[i] = new node_t;
    node_arr[i]->row_index = i;
    for(int j=0;j<features;++j){
      node_arr[i]->annotation[j] = Y[i*features+j];
    }
    for(int anc=i-1;anc>=0;--anc){
      if(A[i*genes+anc]==1){
        //cerr<<"Setting parent of "<<i<<" to "<<anc<<endl; 
        node_arr[i]->parent = node_arr[anc];
        for(int child = 0;child<MAX_CHILDREN;++child){
          if (node_arr[anc]->children[child] == NULL){
            node_arr[anc]->children[child] = node_arr[i];
            //cerr<<"Setting child "<<child<<" of "<<anc<<" to "<<i<<endl;
            node_arr[anc]->is_leaf = false;
            break;
          }
        }
        break;
      }
    }
  }

  for(int j=0;j<features;++j){
    get_annot(node_arr[0],j);
    //node_arr[0]->annotation[j] = get_annot(node_arr[0],j)>1?1:9;
  }

  if(config->verbose) {
    cerr<<"INFERRED SPARSE Y\n";
    for(int i=0;i<genes;++i){
      node_t * node = node_arr[i];
      cerr<<i;
      for(int j=0;j<features;++j){
        if(node->annotation[j]!=9){
          cerr<<"\t"<<j<<" "<<node->annotation[j];
        }
      }
      cerr<<endl;
    }
  }
  for(int i=0;i<genes;++i){
    node_t * node = node_arr[i];
    for(int j=0;j<features;++j){
      if(node->annotation[j]!=9){
        Y[i*features+j] = node->annotation[j];
      }
    }
  }
  //exit(0);
}

void genetree_t::read_sparse_matrix(const char * filename, float * mat,int rows,int cols){
  ifstream ifs(filename);
  if(!ifs.is_open()){
    cerr<<"Cannot open "<<filename<<endl;
    throw "I/O error";
  }
  string line;
  while(getline(ifs,line)){
    istringstream iss(line);
    int index,target,val;
    iss>>index;
    while(iss>>target){
      iss>>val;
      mat[index*cols+target] = val;
    }
  }
  ifs.close();
}

void genetree_t::allocate_memory(){
  Y = new float[genes*features];
  X = new float[genes*features];
  X_projection = new float[genes*features];
  Z = new float[genes*features];
  mut = new float[genes*features];
  mut_projection = new float[genes*features];
  Q = new float[genes*features];

  A = new float[genes*genes];
  A_inv = new float[genes*genes];
  AX = new float[genes*features];
  M = new float[genes*features];
  this->node_arr = new node_t * [genes];

  for(int i=0;i<genes;++i){
    for(int j=0;j<features;++j){
      Y[i*features+j] = 9;
      M[i*features+j] = 0;
    }
  }
  if(config->verbose) cerr<<"Reading in input files\n";
  read_sparse_matrix(annotation_file.data(),Y,genes,features);
  //load_into_matrix(annotation_file.data(),Y,genes,features);
  read_sparse_matrix(mutation_file.data(),M,genes,features);
  //load_into_matrix(mutation_file.data(),M,genes,features);
  for(int i=0;i<genes;++i){
    for(int j=0;j<genes;++j){
      //A[i*genes+j] = 0;
      A[i*genes+j] = i==j?1:0;
    }
  }
  read_sparse_matrix(ancestry_file.data(),A,genes,genes);
  //ofstream ofs_rdebug1("ancestry_rdebug.txt");
  if(config->verbose) cerr<<"BEGIN ANCESTRY\n";
  for(int i=0;i<genes;++i){
    if(config->verbose) cerr<<i;
    for(int j=0;j<genes;++j){
      if(config->verbose)cerr<<" "<<A[i*genes+j];
//      if(j) ofs_rdebug1<<" ";
 //     ofs_rdebug1<<A[i*genes+j];
    }
  //  ofs_rdebug1<<endl;
    if(config->verbose)cerr<<endl;
  }
  //ofs_rdebug1.close();
  if(config->verbose)cerr<<"END ANCESTRY\n";
  infer_y_matrix();
  //exit(0);
  //load_into_matrix(ancestry_file.data(),A,genes,genes);
  bool has_function[features];
  for(int j=0;j<features;++j) has_function[j] = false;
  for(int i=0;i<genes;++i){
    for(int j=0;j<features;++j){
      if(Y[i*features+j]!=9) has_function[j] = true;
    }
  }
  if(config->verbose){
    for(int j=0;j<features;++j) cerr<<"FUNCTION "<<j<<": "<<has_function[j]<<endl;
  }
//  ofstream ofs_rdebug2("x_rdebug.txt");
  
  for(int i=0;i<genes;++i){
    for(int j=0;j<features;++j){
      mut[i*features+j] = 0;
      X[i*features+j] = Y[i*features+j] !=9 ? Y[i*features+j] : 0.5;
      //if(j) ofs_rdebug2<<" ";
      //ofs_rdebug2<<X[i*features+j];
      //mut[i*features+j] = (i==0 && has_function[j]) ? 1 : 0;  // assume root as acquired mutations
      //X[i*features+j] = Y[i*features+j] !=9 ? Y[i*features+j] : has_function[j];
    }
//    ofs_rdebug2<<endl;
  }
  if(config->verbose){
    cerr<<"X INIT\n";
    for(int i=0;i<genes;++i){
      cerr<<i;
      for(int j=0;j<features;++j){
        cerr<<" "<<X[i*features+j];
      }
      cerr<<endl;
    }
  }
  
//  ofs_rdebug2.close();
  // prepare GSL stuff
  gsl_lowrank = gsl_matrix_alloc(genes,features);
  gsl_U = gsl_matrix_alloc(genes,features);
  gsl_S = gsl_vector_alloc(features);
  gsl_V = gsl_matrix_alloc(features,features);
  gsl_svd_work = gsl_vector_alloc(features);
  gsl_A = gsl_matrix_alloc(genes,genes);
  gsl_mut_M = gsl_matrix_alloc(genes,features);
  gsl_A_mut_M = gsl_matrix_alloc(genes,features);
  gsl_matrix * gsl_A_inv = gsl_matrix_alloc(genes,genes);
  for(int i=0;i<genes;++i){
    for(int j=0;j<genes;++j){
      //if(i==j) gsl_matrix_set(gsl_A,i,j,9);
      //else gsl_matrix_set(gsl_A,i,j,0);
      gsl_matrix_set(gsl_A,i,j,A[i*genes+j]);
      //cerr<<" "<<gsl_matrix_get(gsl_A,i,j);
    }
    //cerr<<endl;
  }
  gsl_permutation * gsl_p = gsl_permutation_alloc(genes);
  int sign;
  gsl_linalg_LU_decomp(gsl_A, gsl_p, &sign);
  gsl_linalg_LU_invert (gsl_A, gsl_p, gsl_A_inv);
  //cerr<<"Return codes: "<<rc1<<" "<<rc2<<endl;
  if(config->verbose)cerr<<"BEGIN A INVERSE:\n";
  for(int i=0;i<genes;++i){
    if(config->verbose) cerr<<i;
    for(int j=0;j<genes;++j){
      A_inv[i*genes+j] =  gsl_matrix_get(gsl_A_inv,i,j);
      if(config->verbose) cerr<<" "<<gsl_matrix_get(gsl_A_inv,i,j);
      //cerr<<" "<<A_inv[i*genes+j];
    }
    if(config->verbose)cerr<<endl;
  }
  if(config->verbose)cerr<<"END A INVERSE:\n";
  gsl_matrix_free(gsl_A_inv);  
  gsl_permutation_free(gsl_p);
  update_Z();
  update_Q();
  last_x_norm = 1e10;
  last_mut_norm = 1e10;
  current_x_norm = 0;
  current_mut_norm = 0;
  //exit(0);
}

void genetree_t::project_X(){
  for(int i=0;i<genes;++i){
    for(int j=0;j<features;++j){
      gsl_matrix_set(gsl_U,i,j,X[i*features+j]);
    }
  }
  if(config->verbose){
//    cerr<<"PROJECT X: X original\n";
//    for(int i=0;i<genes;++i){
//      for(int j=0;j<features;++j){
//        if(j) cerr<<" ";
//        cerr<<X[i*features+j];
//      }
//      cerr<<endl;
//    }
  }
  gsl_linalg_SV_decomp(gsl_U,gsl_V,gsl_S,gsl_svd_work);
  if(config->verbose){
    for(int i=0;i<features;++i){
      cerr<<"PROJECT X: eigenvalue "<<i<<": "<<gsl_vector_get(gsl_S,i)<<endl;
    }
  }
  for(int r=rank_X;r<features;++r){
    gsl_vector_set(gsl_S,r,0);
  }
  for(int i=0;i<genes;++i){
    for(int j=0;j<features;++j){
      gsl_matrix_set(gsl_U,i,j,gsl_matrix_get(gsl_U,i,j)*gsl_vector_get(gsl_S,j));
    }
  }
  gsl_blas_dgemm(CblasNoTrans,CblasTrans,1,gsl_U,gsl_V,0,gsl_lowrank);
  //bool standardize = gsl_vector_get(gsl_S,0)>0;
  float min_vals[features];
  float max_vals[features];
  //set<float> sorted_vals[features];
  for(int j=0;j<features;++j) min_vals[j]=1e10;
  for(int j=0;j<features;++j) max_vals[j]=-1e10;
  for(int i=0;i<genes;++i){
    for(int j=0;j<features;++j){
      X_projection[i*features+j] = gsl_matrix_get(gsl_lowrank,i,j);
      if(X_projection[i*features+j]<1e-5) X_projection[i*features+j] = 0;
      //sorted_vals[j].insert(X_projection[i*features+j]);
      if(X_projection[i*features+j]<min_vals[j]) min_vals[j] = X_projection[i*features+j];
      if(X_projection[i*features+j]>max_vals[j]) max_vals[j] = X_projection[i*features+j];
    }
  }
  float ranges[features];
  //float medians[features];
  for(int j=0;j<features;++j){
    ranges[j]=max_vals[j]-min_vals[j];
  //  set<float>::iterator it = sorted_vals[j].begin();
  //  for(int i=0;i<genes/2;++i) it++;
  //  medians[j] = *it;
  //  //if(config->verbose)cerr<<"Median for "<<j<<": "<<medians[j]<<endl;
  }
  
  bool standardize_vec[features]; 
  for(int j=0;j<features;++j){
    standardize_vec[j] = gsl_vector_get(gsl_S,j)>0 && ranges[j]>0;
    //standardize_vec[j] = false;
  }
  for(int i=0;i<genes;++i){
    for(int j=0;j<features;++j){
      if (standardize_vec[j]){
        X_projection[i*features+j]+= -1.*min_vals[j];
        X_projection[i*features+j]/= ranges[j];
      }
    }
  }
  if(config->verbose){
    cerr<<"PROJECT X: X_PROJECTION:\n";
    for(int i=0;i<genes;++i){
      cerr<<i;
      for(int j=0;j<features;++j){
        cerr<<" ";
        cerr<<X_projection[i*features+j];
      }
      cerr<<endl;
    }
  }
}

void genetree_t::project_mut(){
  multiset<mut_t,byIndexAsc> sorted_mut_arr[features];
  for(int i=0;i<genes;++i){
    for(int j=0;j<features;++j){
      float mut_val = mut[i*features+j];
      mut_t m(i*features+j,mut_val);
      sorted_mut_arr[j].insert(m);
    }
  }
  float cutoff = 0.5;
  for(int j=0;j<features;++j){
    int k=0;
    for(multiset<mut_t,byValDesc>::iterator it=sorted_mut_arr[j].begin();it!=sorted_mut_arr[j].end();it++){
      mut_t m = *it;
      if (k<max_mutations &&  m.absval>cutoff){
        mut_projection[m.index] = m.val>0?1:-1;
        ++k;
      }else{
        mut_projection[m.index] = 0;
      }
    }
  }
  if(config->verbose){
    cerr<<"PROJECT MU: mu projection\n";
    for(int i=0;i<genes;++i){
      cerr<<i;
      for(int j=0;j<features;++j){
        cerr<<" ";
        if(fabs(mut_projection[i*features+j])<1e-5) cerr<<"0";
        else cerr<<mut_projection[i*features+j];
      }
      cerr<<endl;
    }
  }
}

void genetree_t::update_Z(){
  for(int i=0;i<genes;++i){
    for(int j=0;j<features;++j){
      gsl_matrix_set(gsl_mut_M,i,j,mut[i*features+j]+M[i*features+j]);
    }
  }
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1,gsl_A,gsl_mut_M,0,gsl_A_mut_M);
  for(int i=0;i<genes;++i){
    for(int j=0;j<features;++j){
      Z[i*features+j] =(Y[i*features+j]!=9)?Y[i*features+j]:gsl_matrix_get(gsl_A_mut_M,i,j);
    }
  }
  if(config->verbose){
    cerr<<"UPDATE Z: begin\n";
    for(int i=0;i<genes;++i){
      cerr<<i;
      for(int j=0;j<features;++j){
        cerr<<" ";
        cerr<<Z[i*features+j];
      }
      cerr<<endl;
    }
  }
}

void genetree_t::update_X(){
  float X_coeff = 1./(1.+dist_func);
  float proj_coeff = dist_func/(1.+dist_func);
  for(int i=0;i<genes;++i){
    for(int j=0;j<features;++j){
      X[i*features+j] = X_coeff*Z[i*features+j] + proj_coeff*X_projection[i*features+j];
    }
  }
  if(config->verbose){
    cerr<<"UPDATE X: distfunc: "<<dist_func<<" coeff: "<<X_coeff<<" proj_coeff: "<<proj_coeff<<" X MATRIX:\n";
    for(int i=0;i<genes;++i){
      cerr<<i;
      for(int j=0;j<features;++j){
        cerr<<" ";
        cerr<<X[i*features+j];
      }
      cerr<<endl;
    }
  }
  update_Q();
}

void genetree_t::update_Q(){
  mmultiply(A_inv,genes,genes,X,features, AX);
  for(int i=0;i<genes;++i){
    for(int j=0;j<features;++j){
      Q[i*features+j] = AX[i*features+j] - M[i*features+j];
    }
  }
  if(config->verbose){
    cerr<<"UPDATE Q: begin\n";
    for(int i=0;i<genes;++i){
      cerr<<i;
      for(int j=0;j<features;++j){
        cerr<<" ";
        cerr<<Q[i*features+j];
      }
      cerr<<endl;
    }
  }
}

void genetree_t::update_mut(){
  float mut_coeff = 1./(1.+dist_func);
  float proj_coeff = dist_func/(1.+dist_func);
  for(int i=0;i<genes;++i){
    for(int j=0;j<features;++j){
      mut[i*features+j] = mut_coeff*Q[i*features+j] + proj_coeff*mut_projection[i*features+j];
    }
  }
  if(config->verbose){
    cerr<<"UPDATE mut: mut_coeff: "<<mut_coeff<<" proj_coeff: "<<proj_coeff<<" mut MATRIX:\n";
    for(int i=0;i<genes;++i){
      cerr<<i;
      for(int j=0;j<features;++j){
        cerr<<" ";
        cerr<<mut[i*features+j];
      }
      cerr<<endl;
    }
  }
  update_Z();
}

void genetree_t::update_map_distance(){
  float X_dist = 0;
  for(int i=0;i<genes;++i){
    for(int j=0;j<features;++j){
      float dev = X[i*features+j] - X_projection[i*features+j];
      X_dist+=dev*dev;
    }
  }
  float mut_dist = 0;
  for(int i=0;i<genes;++i){
    for(int j=0;j<features;++j){
      float dev = mut[i*features+j] - mut_projection[i*features+j];
      mut_dist+=dev*dev;
    }
  }
  if(config->verbose)cerr<<"UPDATE_MAP_DIST: X_dist: "<<X_dist<<" mut_dist: "<<mut_dist<<endl;
  this->map_distance = X_dist+mut_dist;
  this->dist_func = rho/(map_distance+epsilon);
}

void genetree_t::initialize(){
  cerr<<"INITIALIZE: current regularization level mu:"<<mu<< ",last rho:"<<last_rho<<endl;
}

float genetree_t::infer_epsilon(){ 
  return 1;
}

float genetree_t::infer_rho(){ 
  float new_rho = last_rho;
  if(mu==0){
    new_rho = config->rho_min;
  }else{
    if(iter_rho_epsilon==0) new_rho+=config->rho_scale_slow;
  }
  return new_rho;
}

float genetree_t::get_map_distance(){
  return this->map_distance;
}

void genetree_t::print_output(){
  if(in_feasible_region()){
    ofstream ofs_annot("annotation_matrix.txt"); 
    ofs_annot<<"GENE_INDEX";
    for(int j=0;j<features;++j){
      ofs_annot<<"\t"<<"FEATURE_"<<j;
    }
    ofs_annot<<endl;
    for(int i=0;i<genes;++i){
      ofs_annot<<i;
      for(int j=0;j<features;++j){
        //if(Y[i*features+j]!=9) ofs_annot<<"\t"<<Y[i*features+j];
        //else {
          ofs_annot<<"\t";
          if(fabs(X[i*features+j])<1e-5) ofs_annot<<"0";
          else ofs_annot<<X[i*features+j];
        //}
      }
      ofs_annot<<endl;
    }
    ofs_annot.close();
    ofstream ofs_mut("mutation_matrix.txt"); 
    ofs_mut<<"GENE_INDEX";
    for(int j=0;j<features;++j){
      ofs_mut<<"\t"<<"FEATURE_"<<j;
    }
    ofs_mut<<endl;
    for(int i=0;i<genes;++i){
      ofs_mut<<i;
      for(int j=0;j<features;++j){
        ofs_mut<<"\t";
        if(fabs(mut[i*features+j])<1e-5) ofs_mut<<"0";
        else ofs_mut<<mut[i*features+j];
      }
      ofs_mut<<endl;
    }
    ofs_mut.close();
  }
}

bool genetree_t::in_feasible_region(){ 
  float mapdist = get_map_distance();
  bool ret= (mapdist>0 && mapdist<config->mapdist_epsilon);
  return ret;
}

void genetree_t::iterate(){
  cerr<<"Begin iteration "<<iter_rho_epsilon<<"\n";
  project_X();
  project_mut();
  update_X();
  update_mut();
  update_map_distance();
}
//
//// compute the value of the augmented objective function
//
float genetree_t::evaluate_obj(){ 
  last_x_norm = current_x_norm;
  last_mut_norm = current_mut_norm;
  current_x_norm = 0;
  for(int i=0;i<genes;++i){
    for(int j=0;j<features;++j){
      float dev = Z[i*features+j]-X[i*features+j];
      current_x_norm+=dev*dev;
    }
  }
  current_mut_norm = 0;
  for(int i=0;i<genes;++i){
    for(int j=0;j<features;++j){
      float dev = Q[i*features+j]-mut[i*features+j];
      current_mut_norm+=dev*dev;
    }
  }
  float prox_pen = get_prox_dist_penalty();
  float obj = current_x_norm+current_mut_norm+prox_pen;
  cerr<<"EVAL OBJ: X_NORM: "<<current_x_norm<<" MAPDIST: "<<map_distance<<" MU_NORM: "<<current_mut_norm<<" PENALTY: "<<prox_pen<<" OBJ: "<<obj<<endl;
  return obj;
}
bool genetree_t::finalize_inner_iteration(){
  return true;
}
bool genetree_t::finalize_iteration(){ 
  bool cont =  !in_feasible_region();
  //bool cont =  (!in_feasible_region()||((last_x_norm-current_x_norm)>1e-3||fabs(last_mut_norm-current_mut_norm)>1e-3);
  return cont;
}
int genetree_t::get_qn_parameter_length(){ return 0;}
void genetree_t::get_qn_current_param(float * params){
}

void genetree_t::store_qn_current_param(float * params){

}
bool genetree_t::proceed_qn_commit(){
  return true;
}
