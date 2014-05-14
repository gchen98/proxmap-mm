#include"../cl_constants.h"
#include"../proxmap.hpp"
#include"cluster.hpp"

cluster_t::cluster_t(){
}

cluster_t::~cluster_t(){
}

void cluster_t::parse_config_line(string & token,istringstream & iss){
  proxmap_t::parse_config_line(token,iss);
  if (token.compare("EARLY_WEIGHTS")==0){
    iss>>config->early_weightsfile;
  }else if (token.compare("LATE_WEIGHTS")==0){
    iss>>config->late_weightsfile;
  }else if (token.compare("DATAPOINTS")==0){
    iss>>config->datapoints;
  }else if (token.compare("VARIABLES")==0){
    iss>>config->variables;
  }else if (token.compare("GENO_FORMAT")==0){
    iss>>config->geno_format;
  }else if (token.compare("GENO_ORDER")==0){
    iss>>config->geno_order;
  }else if (token.compare("U_DELTA_RHO_CAP")==0){
    iss>>config->u_delta_rho_cap;
  }
}

void cluster_t::allocate_memory(string config_file){
  print_index = 100000;
  proxmap_t::allocate_memory(config_file);
  cerr<<"Allocating class memory\n";
  // figure out the dimensions
  n = config->datapoints;
  //n = linecount(config->genofile.data());
  n2 = n*n;
  p = config->variables;
  this->variable_blocks = (p/BLOCK_WIDTH + (p%BLOCK_WIDTH!=0));
  this->sub_fnorm = new float[n*variable_blocks];
  //p = colcount(config->genofile.data());
  cerr<<"Subjects: "<<n<<" and predictors: "<<p<<endl;
  offsets = new int[n];
  offsets[0] = 0;
  //cerr<<"Offset for "<<0<<" is "<<offsets[0]<<endl;
  int last_width = n;
  for(int i=1;i<n;++i){
    offsets[i] = offsets[i-1]+last_width;
    //cerr<<"Offset for "<<i<<" is "<<offsets[i]<<endl;
    --last_width;
  }
  triangle_dim = offsets[n-1]+1;
  cerr<<"Triangle dim is "<<triangle_dim<<endl;
  cerr<<"Allocating arrays\n";
  rawdata = new float[n*p];
  U  = new float[n*p];
  U_prev  = new float[n*p];
  U_norm_diff = 0;
  U_project = new float[n*p];
  U_project_orig = new float[n*p];
  U_project_prev = new float[n*p];
  weights = new float[triangle_dim];
  V_project_coeff = new float[triangle_dim];
  norm1_arr = new float[n];
  norm2_arr = new float[triangle_dim];
  cerr<<"Reading in input files\n";
  if(config->geno_format.compare("verbose")==0){
    load_into_matrix(config->genofile.data(),rawdata,n,p);
  }else if(config->geno_format.compare("compact")==0){
    load_compact_geno(config->genofile.data());
  }else {
    throw "Invalid genoformat specified.";
  }
  weights = new float[triangle_dim];
  V_project_coeff = new float[triangle_dim];
  cerr<<"Reading in input files\n";
  if(config->geno_format.compare("verbose")==0){
    load_into_matrix(config->genofile.data(),rawdata,n,p);
  }else if(config->geno_format.compare("compact")==0){
    load_compact_geno(config->genofile.data());
  }else {
    throw "Invalid genoformat specified.";
  }
  large = 1e10;
  iter = 0;
  if(run_gpu){
    init_opencl();
  }
}

void cluster_t::initialize(){
  float transition_mu = config->mu_max/2;
  cerr<<"Transition mu for shifting weights is "<<transition_mu<<endl;
  if (mu==transition_mu){
    load_into_triangle(config->late_weightsfile.data(),weights,n,n);
    if(run_gpu) update_weights_gpu();
  }else if (mu==0){
    load_into_triangle(config->early_weightsfile.data(),weights,n,n);
    if(run_gpu) update_weights_gpu();
    if (run_cpu){
      cerr<<"Initializing U and its projection\n";
      bool debug_cpu = false;
      for(int i=0;i<n;++i){
        for(int j=0;j<p;++j){
          //U[i*p+j] = i;
          U[i*p+j] = rawdata[i*p+j];
          U_project_orig[i*p+j] =  U_project[i*p+j] = rawdata[i*p+j];
          if (debug_cpu && i==(n-10) && j>(p-10)){
            cerr<<"CPU: U_project_orig "<<i<<","<<j<<": "<<U_project_orig[i*p+j]<<endl;
          }
        }
      }
    }
    if (run_gpu){
      initialize_gpu();
    }
    update_map_distance();
  }
}

void cluster_t::load_compact_geno(const char * filename){
  ifstream ifs(filename);
  if (!ifs.is_open()){
    cerr<<"Cannot open "<<filename<<" for compact geno load.\n";
    exit(1);
  }
  cerr<<"Genotypes loading...\n";
  string line;
  int rows=0,cols=0;
  bool snp_major = true;
  if(config->geno_order.compare("snp_major")==0){
    rows = p;
    cols = n;
  }else if(config->geno_order.compare("subject_major")==0){
    rows = n;
    cols = p;
    snp_major = false;
  }else{
    throw "Invalid geno order. Valid choices are snp_major,subject_major";
  }
  for(int j=0;j<rows;++j){
    getline(ifs,line);
    for(int i=0;i<cols;++i){
      int index=snp_major?i*p+j:j*n+i;
      rawdata[index] = static_cast<int>(line[i]-'1');
    }
  }
  cerr<<"Genotypes loaded\n";
};



void cluster_t::load_into_triangle(const char * filename,float * & mat,int rows, int cols){
  ifstream ifs(filename);
  if (!ifs.is_open()){
    cerr<<"Cannot open "<<filename<<" for triangle load.\n";
    exit(1);
  }
  string line;
  for(int i=0;i<rows;++i){
    getline(ifs,line);
    istringstream iss(line);
    float val;
    int k = 0;
    for(int j=0;j<cols;++j){
      //cerr<<"At row "<<i<<" col "<<j<<endl;
      iss>>val;
      if (j>=i) {
        // just a test
        //cerr<<"Writing into element: "<<offsets[i]+k<<" of weightmatrix\n";
        //mat[offsets[i]+k] = 1;
        mat[offsets[i]+k] = val;
        ++k;
      }
    }
  }
  ifs.close();
}

float cluster_t::infer_rho(){
  float new_rho = 0;
  if (mu>config->mu_min && U_norm_diff<config->u_delta_rho_cap){
    new_rho = this->last_rho;
    cerr<<"INFER_RHO: Norm diff U is "<<U_norm_diff<<" suspending rho increase. "<<endl;
  }else{
    new_rho = mu*dist_func*rho_distance_ratio;
    cerr<<"INFER_RHO: Rho adjusted proportionally to mu\n";
  }
  cerr<<"INFER_RHO: last rho was "<<this->rho<<" proposed rho is "<<new_rho<<endl;
  return new_rho;
}


void cluster_t::init_v_project_coeff(){
  if (run_gpu){
    init_v_project_coeff_gpu();
  }
  if (run_cpu){
    //float small = 1e-10;
    int nearcoals = 0;
    int noncoals = 0;
    int coals = 0;
    int zeros = 0;
    for(int index1=0;index1<n-1;++index1){
      for(int index2=index1+1;index2<n;++index2){
        float & weight = weights[offsets[index1]+index2-index1];
        float & scaler  = V_project_coeff[offsets[index1]+(index2-index1)];
        if (weight == 0){
          // no shrinkage
          scaler = 1;
          ++zeros;
        }else{
          float l2norm_b = 0;
          for(int j=0;j<p;++j){
            //float v = 100;
            float v = (U_project_orig[index1*p+j]-U_project_orig[index2*p+j]); 
            l2norm_b+=(v*v);
          }
          if(l2norm_b==0){
            scaler = 1;
            ++coals;
          }else{
            l2norm_b = sqrt(l2norm_b);
            float lambda = mu * dist_func * weight / rho;
            if (lambda<l2norm_b){
              scaler = (1.-lambda/l2norm_b);
              ++noncoals;
            }else{
              scaler = 0;
              ++nearcoals;
            }
          }
        }
      }
    }
    bool debug_cpu = false;
    if (debug_cpu){
      for(int index1=0;index1<n-1;++index1){
        for(int index2=index1+1;index2<n;++index2){
          float & scaler  = V_project_coeff[offsets[index1]+(index2-index1)];
          if ( scaler!=0 && scaler!=1 )
          cerr<<"CPU Init_V Index: "<<index1<<","<<index2<<": "<<scaler<<endl;
        }
      }
      //exit(0);
    }
    cerr<<"INIT_V_COEFF: There are "<<zeros<<" zero weights "<<coals<<" coals and "<<noncoals<<" noncoals and "<<nearcoals<<" near coals.\n";
  }
}

bool cluster_t::get_updated_v(int index1,int index2, float * v){
  float & scaler  =  V_project_coeff[offsets[index1]+(index2-index1)];
  if (scaler==0) return false;
  //cerr<<"In get_updated_v\n";
  for(int j=0;j<p;++j){
    v[j] = scaler * (U_project_orig[index1*p+j]-U_project_orig[index2*p+j]);
    bool debug = j<-50;
    if(debug) cerr<<"person "<<index1<<","<<index2<<" variable "<<j<<": "<<(U_project_orig[index1*p+j]-U_project_orig[index2*p+j])<<","<<v[j]<<endl;
    //if(1==1){
    if(isnan(v[j])||isinf(v[j])){
      cerr<<"In get_updated_v between indices "<<index1<<","<<index2<<" variable "<<j<<" with scaler "<<scaler<<" taken from index "<<offsets[index1]+(index2-index1)<<"\n";
      cerr<<"U_project_orig: "<<U_project_orig[index1*p+j]<<","<<U_project_orig[index2*p+j] <<" rawdata: "<<rawdata[index1*p+j]<<","<<rawdata[index2*p+j]<<endl;
      exit(1);
    }
  }
  return true;
}

void cluster_t::update_projection(){
  if (run_gpu){
    store_U_projection_gpu();
  }
  if (run_cpu){
    bool debug_cpu1 = false;
    for(int i=0;i<n;++i){
      for(int j=0;j<p;++j){
        U_project_orig[i*p+j] = U_project[i*p+j];
        U_project[i*p+j] = U[i*p+j];
        if (debug_cpu1 && i>(n-3) && j>(p-3)) cerr<<"CPU: U_project: "<<i<<","<<j<<": "<<U_project[i*p+j]<<endl;
      }
    }
    //if (debug_cpu1) exit(0);
  }
  init_v_project_coeff();
  bool debug = false;
  float last_fnorm[n];
  for(int i=0;i<n;++i) last_fnorm[i] = 1e10;
  bool converged = false;
  int iter = 0;
  //int WIDTH = 2;
  //int BLOCKS = p/BLOCK_WIDTH+(p%BLOCK_WIDTH!=0);
  float left_summation[BLOCK_WIDTH];
  float right_summation[BLOCK_WIDTH];
  float all_summation[BLOCK_WIDTH];
  //this->variable_blocks = (p/BLOCK_BLOCK_WIDTH + (p%BLOCK_BLOCK_WIDTH!=0));
  int maxiter = 1000;
  while(!converged && iter<maxiter){
    float fnorm[n];
    for(int i=0;i<n;++i) fnorm[i] = 0;
    //if (debug) cerr<<"Projection iteration: "<<iter<<endl;
    //int U_project_changes = 0;
    //int U_project_remain = 0;

    if (run_cpu){
      for(int i=0;i<n*p;++i) U_project_prev[i] = U_project[i];
    }
    if (run_gpu){
      update_projection_gpu();
      //
      // 1) Run kernel to copy U_project into U_project_prev
      // 2) Run kernel to compute sub_fnorms and store in 
      // buffer_subject_variable_block_norms.
      // 3) Fetch sub_fnorm from GPU
      //
    }
    if (run_cpu){
      // loop over the grid Y dim
      for(int index = 0;index<n;++index){
        // loop over the grid X dim
        for(int block = 0;block<variable_blocks;++block){
          int left_neighbors=0,right_neighbors=0, neighbors = 0;
          for(int thread=0;thread<BLOCK_WIDTH;++thread){
            int j = block*BLOCK_WIDTH+thread;
            if(j<p){
              left_summation[thread] = 0;
              right_summation[thread] = 0;
              all_summation[thread] = 0;
            }
          }
          for(int i=0;i<n;++i){
            if(i!=index){
              int offset_index = index<i?offsets[index]+i-index: 
              offsets[i]+index-i;
              float weight = weights[offset_index];
              if(weight>0){
                if (i<index){
                  ++left_neighbors;
                  float scaler  =  V_project_coeff[offset_index];
                  //scaler = .5;
                  if (scaler>0){
                    for(int thread=0;thread<BLOCK_WIDTH;++thread){
                      int j = block*BLOCK_WIDTH+thread;
                      if(j<p){
                        left_summation[thread]+=scaler * (U_project_orig[i*p+j]-U_project_orig[index*p+j]);
                      }
                    }
                  }
                }else if (index<i){
                  ++right_neighbors;
                  float scaler  =  V_project_coeff[offset_index];
                  //scaler = .5; 
                  if (scaler>0){
                    for(int thread=0;thread<BLOCK_WIDTH;++thread){
                      int j = block*BLOCK_WIDTH+thread;
                      if(j<p){
                        right_summation[thread]+=scaler * (U_project_orig[index*p+j]-U_project_orig[i*p+j]);
                      }
                    }
                  }
                }
                for(int thread=0;thread<BLOCK_WIDTH;++thread){
                  int j = block*BLOCK_WIDTH+thread;
                  if(j<p){
                    all_summation[thread]+=U_project_prev[i*p+j];
                  }
                }
                ++neighbors;
              } // if positive weight
            } // if neighbor
          }  //loop over neighbors
          sub_fnorm[index*variable_blocks+block] = 0;
          for(int thread=0;thread<BLOCK_WIDTH;++thread){
            int j = block*BLOCK_WIDTH+thread;
            if(j<p){
              //float u =  (U[index*p+j])/(1.+neighbors);
              float u =  (U[index*p+j]+right_summation[thread]-left_summation[thread]+all_summation[thread])/(1.+neighbors);
              //float u =  (right_summation[thread]-left_summation[thread]+all_summation[thread])/(1.+neighbors);
              //float u =  (right_summation[thread]-left_summation[thread])/(1.+neighbors);
              //float u =  (U[index*p+j]+all_summation[thread])/(1.+neighbors);
              U_project[index*p+j] = u;
              sub_fnorm[index*variable_blocks+block] +=u*u;
              if(debug) cerr<<"UPDATE_PROJECTION: index "<<index<<", var "<<j<<" Neighbors: "<<neighbors<<" U point: "<<U[index*p+j]<<" U projection: "<<U_project[index*p+j]<<endl;
            }
          }
           //if (isnan(fnorm)||isinf(fnorm)) exit(1);
        } // loop over blocks
      } // subjects looop
      bool debug_cpu = false;
      if (debug_cpu){
        cerr<<"\nCPU iterate projection:\n";
        for(int i=n-5;i<n;++i){
          cerr<<i<<":";
          for(int j=0;j<p;++j){
            cerr<<" "<<U_project[i*p+j];
          }
          cerr<<endl;
        }
        for(int i=(0);i<n;++i){
          cerr<<i<<":";
          for(int j=0;j<variable_blocks;++j){
            cerr<<" "<<sub_fnorm[i*variable_blocks+j];
          }
          cerr<<endl;
        }
        //exit(1);
      }

    } // if run CPU
    for(int i=0;i<n;++i){
      for(int block=0;block<variable_blocks;++block){
        fnorm[i]+=sub_fnorm[i*variable_blocks+block];
      }
    }
    float max_fnorm_diff = 0;
    for(int i=0;i<n;++i){
      float f = fabs(last_fnorm[i]-fnorm[i]);
      if(max_fnorm_diff<f) max_fnorm_diff = f;
      //cerr<<"F: "<<i<<" : "<<f<<" max: "<<max_fnorm_diff<<endl;
      
    }
    if(debug){
      cerr<<"UPDATE_PROJECTION: fnorms:\n";
      for(int i=0;i<n;++i){
        cerr<<" index: "<<i<<": "<<last_fnorm[i]<<","<<fnorm[i]<<endl;
      }
      cerr<<"UPDATE_PROJECTION: max diff: "<<max_fnorm_diff<<endl;
    }
    converged = max_fnorm_diff<.001;
    //float prop_diff = fabs(last_fnorm-fnorm)/last_fnorm;
    //cerr<<"UPDATE_PROJECTION at iter "<<iter<<", U_project changes "<<U_project_changes<<", U_project_remain "<<U_project_remain<<", Last Frob norm is "<<last_fnorm<<" current is "<<fnorm<<" and diff proportion is "<< prop_diff <<endl;
    //converged = prop_diff<1e-2;
    //last_fnorm = fnorm;
    for(int i=0;i<n;++i){ last_fnorm[i] = fnorm[i];}
    ++iter;
    if(debug)cerr<<"UPDATE_PROJECTION: Iteration "<<iter<<endl;
  }
  if (converged){
    cerr<<"UPDATE_PROJECTION: Converged in "<<iter<<" iterations."<<endl;
  }else{
    cerr<<"UPDATE_PROJECTION: Failed to converged after "<<iter<<" iterations."<<endl;
  }
  for(int i=0;i<n;++i){
    //float norm = 0;
    for(int j=0;j<p;++j){
      //norm +=(U_project_orig[i*p+j]-U_project[i*p+j])*(U_project_orig[i*p+j]-U_project[i*p+j]);
      //U_project_orig[i*p+j] = U_project[i*p+j];
      //if (debug) cerr<<"i:"<<i<<",j:"<<j<<" "<<U_project_orig[i*p+j]<<endl;
    }
    //if(debug && (i<10||i>(n-10)))cerr<<"UPDATE_PROJECTION: PROGRESS index "<<i<<":"<<norm<<"\n";
    
  }
  update_map_distance();
}

void cluster_t::update_projection_nonzero(){
  bool debug = true;
  //bool converged = false;
  //float last_fnorm = -1e10;
  //float fnorm = 0;
  for(int i=0;i<n;++i){
    for(int j=0;j<p;++j){
      U_project_orig[i*p+j] = U_project[i*p+j];
    }
  }
  init_v_project_coeff();
  for(int index = 0;index<n;++index){
    //cerr<<"For subject "<<index<<"\n";
    float v_left[p];
    float v_right[p];
    float left_summation[p];
    float right_summation[p];
    float all_summation[p];
    for(int i=0;i<p;++i){
      left_summation[i] = 0;
      right_summation[i] = 0;
      all_summation[i] = 0;
    }
    int left_neighbors=0,right_neighbors=0, neighbors = 0;
    for(int i=0;i<n;++i){
      if(i!=index){
        if (i<index){
          ++left_neighbors;
          if (get_updated_v(i,index,v_left)){
            for(int j=0;j<p;++j){
              //left_summation[j]+=V[i*n*p+index*p+j];
              left_summation[j]+=v_left[j];
            }
          }
        }else if (index<i){
          ++right_neighbors;
          if (get_updated_v(index,i,v_right)){
            for(int j=0;j<p;++j){
              //right_summation[j]+=V[index*n*p+i*p+j];
              right_summation[j]+=v_right[j];
            }
          }
        }
        for(int j=0;j<p;++j){
          all_summation[j]+=U[i*p+j];
        }
      }
      ++neighbors;
    }
    for(int j=0;j<p;++j){
       U_project[index*p+j] = (U[index*p+j]+right_summation[j]-left_summation[j]+all_summation[j])/(n);
       if(debug) cerr<<"UPDATE_PROJECTION: index "<<index<<", var "<<j<<" n: "<<n<<" U point: "<<U[index*p+j]<<" right: "<<right_summation[j]<<" left: "<<left_summation[j]<<" all: "<<all_summation[j]<<" U projection: "<<U_project[index*p+j]<<endl;
    }
  } // subjects looop
}


float cluster_t::get_map_distance(){
  return this->map_distance;
}

void cluster_t::update_map_distance(){
  if(run_gpu){
    update_map_distance_gpu();
  }
  if (run_cpu){
    float norm1 = 0;
    float norm2 = 0;
    // compute the frobenius norm first
    float norm1_arr[variable_blocks];
    float norm2_arr[variable_blocks];
    for(int k=0;k<variable_blocks;++k){
      norm1_arr[k] = 0;
      norm2_arr[k] = 0;
      float norm1_temp[BLOCK_WIDTH];
      float norm2_temp[BLOCK_WIDTH];
      for(int t=0;t<BLOCK_WIDTH;++t){
        norm1_temp[t] = 0;
        norm2_temp[t] = 0;
        int j = k*BLOCK_WIDTH+t;
        if (j<p){
          for(int i=0;i<n;++i){
            norm1_temp[t]+=(U[i*p+j]-U_project[i*p+j])*(U[i*p+j]-U_project[i*p+j]);
            //cerr<<"U and U_project: "<<U[i*p+j]<<","<<U_project[i*p+j]<<endl;
            for(int i2=(i+1);i2<n;++i2){
              norm2_temp[t]+=((U[i*p+j]-U[i2*p+j]) - (U_project[i*p+j]-U_project[i2*p+j]))*
                 ((U[i*p+j]-U[i2*p+j]) - (U_project[i*p+j]-U_project[i2*p+j]));
            }
          }
        }
        norm1_arr[k]+=norm1_temp[t];
        norm2_arr[k]+=norm2_temp[t];
      }
      norm1+=norm1_arr[k];
      norm2+=norm2_arr[k];
      //cerr<<"CPU Block "<<k<<" norm: "<<norm1_arr[k]<<","<<norm2_arr[k]<<endl;
    }
    cerr<<"CPU Norm1 was "<<norm1<<" and norm2 was "<<norm2<<endl;
    float norm = norm1+norm2;
    this->map_distance = norm;
    this->dist_func = sqrt(this->map_distance+epsilon);
    cerr<<"GET_MAP_DISTANCE: New map distance is "<<norm<<" with U distance="<<norm1<<", V distance="<<norm2<<" dist_func: "<<dist_func<<endl;
  }
}

void cluster_t::update_u(){
  //dist_func = 0;rho=2;
  if(run_gpu){
    update_u_gpu();
  } 
  if (run_cpu){
    bool debug = false;
    int geno_changes = 0;
    int U_changes = 0;
    int mixes = 0;
    for(int i=0;i<n;++i){
      for(int j=0;j<p;++j){
        U[i*p+j] = dist_func*rawdata[i*p+j]/(dist_func+rho) + rho*U_project[i*p+j]/(dist_func+rho);
        if (debug && (i<10 || i>(n-10))) cerr<<"UPDATE_U: index: "<<i<<" geno: "<<rawdata[i*p+j]<<" projection: "<<U_project[i*p+j]<<" U: "<<U[i*p+j]<<endl;
        geno_changes+=(U[i*p+j]!=rawdata[i*p+j]);
        U_changes+=(U[i*p+j]!=U_project[i*p+j]);
        mixes+=((U[i*p+j]!=U_project[i*p+j]) &&(U[i*p+j]!=rawdata[i*p+j]));
      }
    }
    bool debug_cpu = false;
    if(debug_cpu){
      for(int i=0;i<n;++i){
        for(int j=0;j<p;++j){
          if(i==(n-10) && j>(p-10)){
            cerr<<"update U CPU: U: "<<i<<","<<j<<": "<<U[i*p+j]<<endl;
          }
        }
      }
      //exit(0);
    }

    cerr<<"UPDATE_U: genochanges: "<<geno_changes<<" U_project changes: "<<U_changes <<" mixtures: "<<mixes<<"\n";
  }
  update_map_distance();
}


void cluster_t::print_output(){
  //bool complete = false;
  bool print = false;
  cerr<<"PRINT_OUTPUT mu = "<<mu<<" last current "<<last_vnorm<<","<<current_vnorm<<endl;
  if (mu==0 || mu>=config->mu_max){
    print = true;
    last_vnorm = 1e10;
  }else{
    if (last_vnorm-current_vnorm>.01){
      print = true;
      last_vnorm = current_vnorm;
    }
  }
  if(print){
    if (run_gpu){
      get_U_gpu();
    }
    ostringstream oss;
    oss<<print_index<<"_rho"<<rho<<".epsilon"<<epsilon<<".mu"<<mu<<".clusters.txt";
    string filename=oss.str();
    ofstream ofs(filename.data());
    //ofs<<"DATAPOINT	CLUSTER\n";
    for(int i=0;i<n;++i){
      //ofs<<i<<"\t";
      for(int j=0;j<p;++j){
        if(j) ofs <<" ";
        ofs<<U[i*p+j];
      }
      ofs<<endl;
    }
    ofs.close();
  
    ostringstream oss2;
    oss2<<print_index<<"_rho"<<rho<<".epsilon"<<epsilon<<".mu"<<mu<<".coals.txt";
    filename=oss2.str();
    ofs.open(filename.data());
    for(int i1=0;i1<n-1;++i1){
      for(int i2=i1+1;i2<n;++i2){
        float norm = 0;
        for(int j=0;j<p;++j){
          norm+= (U[i1*p+j] - U[i2*p+j])*(U[i1*p+j] - U[i2*p+j]);
        }
        norm=sqrt(norm);
        if(norm<.05){
          ofs<<i1<<"\t"<<i2<<endl;
        }
      }
    }
    ofs.close();
    ++print_index;
  }
}

void cluster_t::print_cluster(ostream & os){
  for(int i=0;i<n;++i){
    for(int j=0;j<p;++j){
      if(j) cout <<"\t";
      os<<U[i*p+j];
    }
    os<<endl;
  }
}

void cluster_t::check_constraint(){
  in_feasible_region();
}

bool cluster_t::in_feasible_region(){
  return true;
  float norm = 0;
  int normalizer = 0;
  int clusters=0;
  int clusters_u=0;
  bool debug = false;
  float cluster_threshold = .05;
  float c_min=1e10;
  float c_max = -1e10;
  float c_mean = 0;
  for(int index=0;index<n-1;++index){
    for(int i=index+1;i<n;++i){
      float c = 0;
      float c_u = 0;
      float v_update[p];
      if(!get_updated_v(index,i,v_update)){
        for(int j=0;j<p;++j){ v_update[j]=0;}
      }
      float v_true[p];
      for(int j=0;j<p;++j){
        v_true[j] = U[index*p+j] - U[i*p+j]; 
      }
      //cerr<<"Checking pair "<<index<<","<<i<<endl;
      if(debug) cerr<<"IN_FEASIBLE_REGION: indices "<<index<<","<<i<<":"; 
      for(int j=0;j<p;++j){
        c+=(v_update[j]*v_update[j]);
        c_u+=(v_true[j]*v_true[j]);
        //cerr<<"True: "<<v_true[j]<<" update: "<<v_update[j]<<endl;
        if(debug){
          cerr<<" "<<j<<":"<<v_true[j]<<","<<v_update[j];
        }
        if (v_true[j]!=v_update[j]){
          norm+=(v_true[j]-v_update[j])*(v_true[j]-v_update[j]);
        }
        ++normalizer;
      }
      c=sqrt(c);
      c_u=sqrt(c_u);
      if (c<cluster_threshold){
        if (c<c_min) c_min = c;
        if(c>c_max)c_max=c;
        c_mean+=c;
        ++clusters;
      }
      if (c_u<cluster_threshold){
        ++clusters_u;
        //cerr<<"IN_FEASIBLE_REGION: Coalescent event at "<<index<<","<<i<<":\n";
        for(int j=0;j<p;++j){
          //cerr<<" "<<j<<":"<<U[index*p+j]<<","<<U[i*p+j]<<endl;
        }
      }
// cerr<<"IN_FEASIBLE_REGION: indices "<<index<<","<<i<<" with v norm: "<<c<<endl; 
      if(debug) cerr<<endl;
      //clusters+=(c==0);
    }
  }
  norm/=1.*normalizer;
  //cerr<<"IN_FEASIBLE_REGION: total V coalescent events: "<<clusters<<" U-U events "<<clusters_u<<" c_norm mean: "<<(c_mean*1./clusters)<<" range: ("<<c_min<<"-"<<c_max<<")"<<endl;
  bool ret = norm<1e-4;
  //cerr<<"IN_FEASIBLE_REGION: norm "<<norm<<" differences "<<normalizer<<" returning "<<ret<<endl;
  return ret;
}

// each iteration of the projection and the point updates

void cluster_t::iterate(){
  update_projection(); 
  update_u();
}

// compute the value of the augmented objective function

float cluster_t::evaluate_obj(){
  if (run_gpu){
    evaluate_obj_gpu();
  }
  if (run_cpu){
    for(int i=0;i<n;++i){
      norm1_arr[i]=0;
    }
    for(int i=0;i<triangle_dim;++i){
      norm2_arr[i]=0;
    }
    float norm1_temp[BLOCK_WIDTH];
    float norm2_temp[BLOCK_WIDTH];
    for(int i1=0;i1<n;++i1){
      for(int i2=i1;i2<n;++i2){
        if(i1==i2){
          // compute norm 1
          float single_norm = 0;
          for(int t=0;t<BLOCK_WIDTH;++t){
            norm1_temp[t] = 0;
          }
          // accumulate across variable blocks
          for(int k=0;k<variable_blocks;++k){
            for(int t=0;t<BLOCK_WIDTH;++t){
              int j = k*BLOCK_WIDTH+t;
              if (j<p){
                float dev = rawdata[i1*p+j]-U[i1*p+j];
                //dev = rawdata[i1*p+j]; 
                norm1_temp[t]+=dev*dev;
              }
            }
          }
          // reduction
          for(int t=0;t<BLOCK_WIDTH;++t){
            single_norm+=norm1_temp[t];
          }
          norm1_arr[i1] = single_norm;
        }else{
          int offset_index = offsets[i1]+i2-i1;
          float weight = weights[offset_index]; 
          //weight = 1;
          if (weight>0){
            float scaler = V_project_coeff[offset_index];
            //scaler = 1;
            if (scaler>0){
              float pair_norm = 0;
              for(int t=0;t<BLOCK_WIDTH;++t){
                norm2_temp[t] = 0;
              }
              // accumulate across variable blocks
              for(int k=0;k<variable_blocks;++k){
                for(int t=0;t<BLOCK_WIDTH;++t){
                  int j = k*BLOCK_WIDTH+t;
                  if (j<p){
                    float dev = scaler * 
                    (U_project[i1*p+j]-U_project[i2*p+j]);
                    //dev = U_project[i1*p+j];
                    norm2_temp[t]+=dev*dev;
                  }
                }
              }
              // reduction
              for(int t=0;t<BLOCK_WIDTH;++t){
                pair_norm+=norm2_temp[t];
              }
              norm2_arr[offset_index]=weight * sqrt(pair_norm);
            } // if scaler is positive
          } // if positive weight
        } // loop over true neightbors
      } // loop over possible neighbors 
    } // loop over first person
    bool debug_cpu = false;
    if(debug_cpu){
      cerr<<"CPU NORM1:";
      for(int i=0;i<n;++i){
        cerr<<" "<<norm1_arr[i];
      }
      cerr<<endl;
      for(int i1=0;i1<n-1;++i1){
        cerr<<"CPU NORM2["<<i1<<"]:";
        for(int i2=i1+1;i2<n;++i2){
          if (i2>i1){
            cerr<<" "<<norm2_arr[offsets[i1]+i2-i1];
          }
        }
        cerr<<endl;
      }
      //exit(1);
    }
  } // end run CPU
  float norm1 = 0;
  float norm2 = 0;
  for(int i=0;i<n;++i){
    norm1+=norm1_arr[i];
    for(int i2=i+1;i2<n;++i2){
      norm2+=norm2_arr[offsets[i]+i2-i];
    }
  }
  float penalty = get_prox_dist_penalty();
  float obj = .5*norm1+mu*norm2+penalty;
  current_vnorm = norm2;
  cerr<<"EVAL_OBJECTIVE: Objective is "<<obj<<" ||X-U|| is "<<norm1<<" and ||V|| is "<<norm2<<" mu penalized ||V||: "<<(mu*norm2)<<" proxdist penalty: "<<penalty<<endl;
  return obj;
}   

bool cluster_t::finalize_iteration(){
  if(run_gpu){
    finalize_iteration_gpu();
  }
  if(run_cpu){
    U_norm_diff = 0;
    for(int i=0;i<n;++i){
      for(int j=0;j<p;++j){
        float dev = (U[i*p+j]-U_prev[i*p+j]);
        U_norm_diff+=dev*dev;
      }
    }
    U_norm_diff=sqrt(U_norm_diff);
    cerr<<"FINALIZE_ITERATION: CPU U_norm_diff: "<<U_norm_diff<<endl;
    for(int i=0;i<n;++i){
      for(int j=0;j<p;++j){
        U_prev[i*p+j] = U[i*p+j];
      }
    }
  }
  return (mu<=config->mu_min || fabs(current_vnorm-0)>.0001);
}

int main_cluster(int argc,char * argv[]){
  if(argc<2){
    cerr<<"Usage: <config file>\n";
    return 1;
  }
  try{
    //cluster_t * cluster = new cluster_t();
    //if (!cluster->init(config_file)){
    //  throw "Failed to initialize.";
    //}
    //cluster->coalesce();
    //cluster->print_cluster(cout);
    //delete cluster;
  }catch(const char * & excp_str){
    cerr<<"Exception caught with message "<<excp_str<<endl;
    return 1;
  }

  cerr<<"Done!\n";
  return 0;
}

