#include"proxmap.hpp"
#include"regression/regression.hpp"
#include"convex_clustering/cluster.hpp"

config_t::config_t(){
  //rho = 0;
  platform_id = 0;
  device_id = 0;
}

proxmap_t::proxmap_t(){
  config = new config_t;
}
  
proxmap_t::~proxmap_t(){
  delete config;
}

void proxmap_t::init(string configfile){
  allocate_memory(configfile);
}

void proxmap_t::parse_config_line(string & token,istringstream & iss){
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
  }else if (token.compare("GENOTYPES")==0){
    iss>>config->genofile;
  }else if (token.compare("RHO_DISTANCE_RATIO")==0){
    iss>>config->rho_distance_ratio;
  }else if (token.compare("RHO_MIN")==0){
    iss>>config->rho_min;
  }else if (token.compare("RHO_SCALE_FAST")==0){
    iss>>config->rho_scale_fast;
  }else if (token.compare("RHO_SCALE_SLOW")==0){
    iss>>config->rho_scale_slow;
  }else if (token.compare("RHO_MAX")==0){
    iss>>config->rho_max;
  }else if (token.compare("EPSILON_MAX")==0){
    iss>>config->epsilon_max;
  }else if (token.compare("EPSILON_SCALE_FAST")==0){
    iss>>config->epsilon_scale_fast;
  }else if (token.compare("EPSILON_SCALE_SLOW")==0){
    iss>>config->epsilon_scale_slow;
  }else if (token.compare("EPSILON_MIN")==0){
    iss>>config->epsilon_min;
  }else if (token.compare("MU_MIN")==0){
    iss>>config->mu_min;
  }else if (token.compare("MU_INCREMENT")==0){
    iss>>config->mu_increment;
  }else if (token.compare("MU_MAX")==0){
    iss>>config->mu_max;
  }
}

void proxmap_t::allocate_memory(string config_file){
  cerr<<"Allocating base class memory\n";
  ifstream ifs_config(config_file.data());
  if (!ifs_config.is_open()){
    cerr<<"Cannot open the file "<<config_file<<".\n";
    throw "IO error";
  }
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

void proxmap_t::run(){
  float epsilon_max = config->epsilon_max;
  //float epsilon_scaler_fast = config->epsilon_scale_fast;
  //float epsilon_scaler_slow = config->epsilon_scale_slow;
  //float epsilon_min = config->epsilon_min;
  //float rho_max = config->rho_max;
  //float rho_scaler_fast = config->rho_scale_fast;
  //float rho_scaler_slow = config->rho_scale_slow;
  //float rho_min = config->rho_min;
  float mu_min = config->mu_min;
  float mu_increment = config->mu_increment;
  float mu_max = config->mu_max;
  float mu = mu_min;
  int iter_mu = 0;
  do{ // loop over mu
    cerr<<"Mu iterate: "<<iter_mu<<" mu="<<mu<<endl;
    rho_distance_ratio = config->rho_distance_ratio;
    epsilon = 0.01;
    initialize(mu);
    //rho = rho_min;
    int iter_rho_epsilon = 0;
    bool converged = false;
    float last_obj=1e10;
    //float last_rho = 0;
    //bool move_rho = true;
    int burnin = 1;
    
    int max_iter = 1000;
    while(!converged && iter_rho_epsilon<max_iter){
      cerr<<"Inner iterate: "<<iter_rho_epsilon<<endl;
      rho = infer_rho();
      //if (iter_rho_epsilon>0) rho = last_rho;
      //if (rho<last_rho) {
        //rho = last_rho;
        //cerr<<"Backtracking to previous rho of "<<rho<<endl;
      //}
      //if (rho>rho_max) rho = rho_max;
      //if (epsilon<epsilon_min) epsilon = epsilon_min;
      cerr<<" New rho="<<rho<<", epsilon="<<epsilon<<endl;
      iterate();
      float obj = evaluate_obj();
      if (iter_rho_epsilon>burnin){
        if(last_obj<=obj){
          converged = true;
          cerr<<"Objective function is not changing or going uphill from "<<last_obj<<" to "<<obj<<", aborting.\n";
        }else if(fabs(last_obj-obj)/last_obj<1e-6){
          converged=true; 
          cerr<<"Converged with last obj "<<last_obj<<" current "<<obj<<"!\n";
        }else{
          cerr<<"Continuing with last obj "<<last_obj<<" current "<<obj<<"!\n";
        }
      }else{
        cerr<<"Ignoring objective on warmup iteration\n";
      }
      bool feasible = in_feasible_region();
      cerr<<"In feasible region?: "<<feasible<<endl;
      last_obj = obj;
      //last_rho = rho;
      ++iter_rho_epsilon;
    }
    print_output();
    //while(epsilon>epsilon_min || rho<rho_max);
    mu+=mu_increment;
    ++iter_mu;
  }while(mu<mu_max);
}

float proxmap_t::get_prox_dist_penalty(){
  float mapdist = get_map_distance();
  float penalty = (.5*rho*mapdist)/(sqrt(mapdist+epsilon));
  //cerr<<"Penalty is "<<penalty<<endl;
  return penalty;
}


int main(int argc, char * argv[]){
  try{
    int arg=0;
    if (argc<3){
      cerr<<"Usage: <analysis: [cluster|regression]> <configfile>\n";
      return 1;
    }
    string analysis = argv[++arg];
    string configfile = argv[++arg];
    proxmap_t * proxmap = NULL;
    if (analysis.compare("cluster")==0){ 
      proxmap = new cluster_t();
    }else if (analysis.compare("regression")==0){ 
      proxmap = new regression_t();
    }
    if (proxmap!=NULL){
      proxmap->init(configfile);
      proxmap->run();
      delete proxmap;
    }
  }catch(const char * & estr){
    cerr<<"Exception caught of message: "<<estr<<endl;
    return -1;
  }
  return 0;
}
