#include"proxmap.hpp"
#include"regression/regression.hpp"
#include"convex_clustering/cluster.hpp"
#include"cross_validation.hpp"

cross_validation_t::cross_validation_t(){
#ifdef USE_MPI
  MPI::Init();
  this->mpi_rank = MPI::COMM_WORLD.Get_rank();
#endif
}

cross_validation_t::~cross_validation_t(){
  if(mpi_rank==0){
    delete[] fold_id;
    delete[] trained_betas;
  }
#ifdef USE_MPI
  MPI::Finalize();
#endif
}

void cross_validation_t::allocate_memory(string config_file){
  ifstream ifs_config(config_file.data());
  if (!ifs_config.is_open()){
    cerr<<"Cannot open the file "<<config_file<<".\n";
    throw "IO error";
  }
  string line,token,val;
  while (getline(ifs_config,line)){
    istringstream iss(line);
    iss>>token;
    if (token.compare("FOLDS")==0){
      iss>>this->folds;
    }else if (token.compare("UID")==0){
      iss>>this->uid;
    }else if (token.compare("GENOTYPES")==0){
      iss>>this->genofile;
    }else if (token.compare("TRAIT")==0){
      iss>>this->traitfile;
    }else if (token.compare("MIN_K")==0){
      iss>>this->min_k;
    }else if (token.compare("MAX_K")==0){
      iss>>this->max_k;
    }else if (token.compare("K_INCREMENT")==0){
      iss>>this->k_increment;
    }
  }
  ifs_config.close();
}

void cross_validation_t::init_folds(){
  int variables = proxmap_t::colcount(this->genofile.data());
  int observations = proxmap_t::linecount(this->traitfile.data());
  this->rows = observations;
  this->cols = variables;
  this->trained_betas = new float[folds * variables];
  cerr<<"Dimensions for dataset are "<<observations<<" by "<<variables<<endl;
  fold_id = new int[observations];
  for(int i=0;i<observations;++i) fold_id[i] = -1;
  int total_inserted = 0;
  int current_fold_id = 0;
  int observations_per_fold = observations / this->folds;
  int fold_count = 0;
  while(total_inserted<observations){
    if(current_fold_id<(folds-1) && fold_count==observations_per_fold){
      ++current_fold_id;
      fold_count = 0;
    }
    bool inserted = false;
    do{
      int randindex = static_cast<int>(rand()/(RAND_MAX+1.) * observations);
      if (fold_id[randindex]==-1){
        fold_id[randindex] = current_fold_id;
        ++fold_count;
        ++total_inserted;
        //cerr<<"Inserted "<<current_fold_id<<" at "<<randindex<<". Total inserted is "<<total_inserted<<endl;
        inserted = true;
      }
    }while(!inserted);
  }
  for(int i=0;i<observations;++i){
    cerr<<fold_id[i];
  }
  cerr<<endl; 
}

bool test_exists(const char * filename){
  ifstream test_file(filename);
  bool flag = false;
  if (test_file.is_open()){
    flag = true;
    test_file.close();
  }
  return flag;
}

void cross_validation_t::generate_train_test_files(config_t * & config, int current_fold_id){
  //cerr<<"Original X: "<<config->genofile<<endl;
  ostringstream oss_X_train,oss_X_test,oss_y_train,oss_y_test;
  ostringstream oss_xxi_inv;
  oss_X_train<<"uid."<<this->uid<<"."<<this->genofile<<".fold."<<current_fold_id<<".train";
  oss_y_train<<"uid."<<this->uid<<"."<<this->traitfile<<".fold."<<current_fold_id<<".train";
  oss_X_test<<"uid."<<this->uid<<"."<<this->genofile<<".fold."<<current_fold_id<<".test";
  oss_y_test<<"uid."<<this->uid<<"."<<this->traitfile<<".fold."<<current_fold_id<<".test";
  oss_xxi_inv<<"uid."<<this->uid<<"."<<config->xxi_inv_file_prefix<<".fold."<<current_fold_id;
  string outfile_X_train = oss_X_train.str();
  string outfile_X_test = oss_X_test.str();
  string outfile_y_train = oss_y_train.str();
  string outfile_y_test = oss_y_test.str();
  // rewrite the configuration files for all nodes
  config->genofile = outfile_X_train;
  config->traitfile = outfile_y_train;
  config->xxi_inv_file_prefix = oss_xxi_inv.str();
  if(mpi_rank>0) return;
  int checks = 0;
  checks+=test_exists(outfile_X_train.data());
  checks+=test_exists(outfile_X_test.data());
  checks+=test_exists(outfile_y_train.data());
  checks+=test_exists(outfile_y_test.data());
  if (checks==4) return; // do not need to regenerate these files
  ofstream ofs_X_train(outfile_X_train.data());
  ofstream ofs_X_test(outfile_X_test.data());
  ofstream ofs_y_train(outfile_y_train.data());
  ofstream ofs_y_test(outfile_y_test.data());
  ifstream ifs_X(this->genofile.data());
  ifstream ifs_y(this->traitfile.data());
  for(int i=0;i<rows;++i){
    string line_X,line_y;
    getline(ifs_X,line_X);
    getline(ifs_y,line_y);
    if (current_fold_id==this->fold_id[i]){
      ofs_X_test<<line_X<<endl;
      ofs_y_test<<line_y<<endl;
    }else{
      ofs_X_train<<line_X<<endl;
      ofs_y_train<<line_y<<endl;
    }
  }
  ofs_X_train.close();ofs_y_train.close();
  ofs_X_test.close();ofs_y_test.close();
  ifs_X.close();ifs_y.close();
}

float cross_validation_t::compute_mse(){
  float mse = 0;
  for(int fold = 0;fold<this->folds;++fold){
    ostringstream oss_X_test,oss_y_test;
    oss_X_test<<"uid."<<this->uid<<"."<<this->genofile<<".fold."<<fold<<".test";
    oss_y_test<<"uid."<<this->uid<<"."<<this->traitfile<<".fold."<<fold<<".test";
    ifstream ifs_X(oss_X_test.str().data());
    ifstream ifs_y(oss_y_test.str().data());
    float deviance = 0;
    rows = proxmap_t::linecount(oss_y_test.str().data());
    for(int row=0;row<rows;++row){
      string line_X;
      string line_y;
      getline(ifs_X,line_X);
      getline(ifs_y,line_y);
      istringstream iss_y(line_y);
      istringstream iss_X(line_X);
      float y,x;
      iss_y>>y;
      float fitted = 0;
      for(int col=0;col<cols;++col){
        iss_X>>x;
        fitted+=trained_betas[fold*cols+col]*x;
      }
      cerr<<row<<": Y "<<y<<" fitted "<<fitted<<endl;
      deviance+=(y-fitted)*(y-fitted);
    }
    cerr<<"Deviance for fold "<<fold<<" is "<<deviance<<endl;
    ifs_X.close();
    ifs_y.close();
    mse+=deviance;
  }
  mse/=folds;
  return mse;
}

float cross_validation_t::eval_k(int k,vector<int>  & k_vec, vector<float> & mse_vec, string regression_config_file){
  float mse = 0;
#ifdef USE_MPI
  for(int fold=0;fold<this->folds;++fold){
    proxmap_t * proxmap = NULL;
    proxmap = new regression_t(false);
    if (proxmap!=NULL){
      proxmap->init(regression_config_file);
      regression_t * regression = static_cast<regression_t * >(proxmap);
      generate_train_test_files(regression->config,fold);
      regression->config->top_k = k;
      if(mpi_rank==0){
        cerr<<"K "<<k<<" fold "<<fold<<endl;
        cerr<<"New files for fold are: "<<regression->config->genofile<<","<<regression->config->traitfile<<","<<regression->config->xxi_inv_file_prefix<<endl;
      }
      // barrier ensures that any new files are done writing to disk
      MPI_Barrier(MPI_COMM_WORLD);
      cerr<<"MPI RANK "<<mpi_rank<<" at barrier\n";
      proxmap->allocate_memory();
      proxmap->run();
      if(mpi_rank==0){
        for(int j=0;j<cols;++j){
          trained_betas[fold*cols+j] = fabs(regression->beta[j])>regression->config->beta_epsilon?regression->beta[j]:0;
          if(trained_betas[fold*cols+j]!=0) cerr<<"Fold "<<fold<<" j "<<j<<" val "<<trained_betas[fold*cols+j]<<endl;
        }
      }
      delete proxmap;
    }
  }
  if(mpi_rank==0){
    mse = compute_mse();
    cerr<<"MSE for k="<<k<<" is "<<mse<<endl;
    k_vec.push_back(k);
    mse_vec.push_back(mse);
  }
  MPI_Bcast(&mse,1,MPI_FLOAT,0,MPI_COMM_WORLD);
#endif
  return mse;
}

void cross_validation_t::run(string regression_config_file){
#ifdef USE_MPI
  if(mpi_rank==0){
    init_folds();
  }
  //float last_mse = 1e10;
  vector<float> mse_vec;
  vector<int> k_vec;
  int bracket_left = -1;
  int bracket_right = -1;
  if(bracket_right>max_k)bracket_right = max_k;
  int last_k=-1;
  float last_mse=1e10;
  // broad search
  for(int k=min_k;k<=max_k;k+=k_increment){
    float mse = eval_k(k,k_vec,mse_vec,regression_config_file);
    if (mse>last_mse){
      // set brackets
      bracket_left = last_k+1;
      bracket_right = k-1;
      if (bracket_right>max_k) bracket_right = max_k;
      //if (bracket_right<bracket_left) bracket_right = -1;
      cerr<<"Brackets set to "<<bracket_left<<" to "<<bracket_right<<endl;
      // terminate here
      break;
    }
    last_k = k;
    last_mse = mse;
  }
  // fine search
  if (bracket_left>-1 && bracket_right>-1){
    for(int k=bracket_left;k<=bracket_right;++k){
      eval_k(k,k_vec,mse_vec,regression_config_file);
    } //bracket loop
  }

  int best_k;
  if(mpi_rank==0){
    float min_mse = 1e10;
    int best_index = -1;
    for(uint index = 0;index<k_vec.size();++index){
      if (mse_vec[index]<min_mse){
        best_index = index;
        min_mse = mse_vec[index];
      }
    }
    best_k = k_vec[best_index];
    cerr<<"Best K was "<<best_k<<endl;
  }
  MPI_Bcast(&best_k,1,MPI_INT,0,MPI_COMM_WORLD);
  // do one final fit on the best K
  proxmap_t * proxmap = NULL;
  proxmap = new regression_t(false);
  if (proxmap!=NULL){
    proxmap->init(regression_config_file);
    regression_t * regression = static_cast<regression_t * >(proxmap);
    regression->config->genofile = this->genofile;
    regression->config->traitfile = this->traitfile;
    regression->config->top_k = best_k;
    proxmap->allocate_memory();
    cout<<"FINAL_MODEL:\n";
    proxmap->run();
    delete proxmap;
  }
#endif
}

int main(int argc, char * argv[]){
  try{
    int arg=0;
    if (argc<3){
      cerr<<"Usage: <cross_validation config> <proxmap configfile>\n";
      return 1;
    }
    string config_cross = argv[++arg];
    string configfile = argv[++arg];

    cross_validation_t * cross_validation = new cross_validation_t();
    cross_validation->allocate_memory(config_cross);
    cross_validation->run(configfile);
    delete cross_validation;

  }catch(const char * & estr){
    cerr<<"Exception caught of message: "<<estr<<endl;
    return -1;
  }
  return 0;
}

