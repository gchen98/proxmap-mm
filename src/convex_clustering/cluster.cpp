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
  genotypes = new float[n*p];
  U  = new float[n*p];
  U_project = new float[n*p];
  U_project_orig = new float[n*p];
  weights = new float[triangle_dim];
  V_project_coeff = new float[triangle_dim];
  cerr<<"Reading in input files\n";
  if(config->geno_format.compare("verbose")==0){
    load_into_matrix(config->genofile.data(),genotypes,n,p);
  }else if(config->geno_format.compare("compact")==0){
    load_compact_geno(config->genofile.data());
  }else {
    throw "Invalid genoformat specified.";
  }
  load_into_triangle(config->early_weightsfile.data(),weights,n,n);
  large = 1e10;
  iter = 0;

  run_gpu = config->use_gpu;
  if(run_gpu){
    init_opencl();
  }
}

void cluster_t::initialize(float mu){
  this->mu = mu;
  this->delta = 0.;
  //this->delta = .0001;
  if(mu==0){
    cerr<<"Initializing U and its projection\n";
    for(int i=0;i<n;++i){
      for(int j=0;j<p;++j){
        U[i*p+j] = genotypes[i*p+j];
        U_project_orig[i*p+j] =  U_project[i*p+j] = U[i*p+j];
        //cerr<<"U_project_orig "<<i<<","<<j<<": "<<U_project_orig[i*p+j]<<endl;
        //U_project_orig[i*p+j] =  U_project[i*p+j] = 0;
      }
    }
  }
  float transition_mu = config->mu_max/2;
  cerr<<"Transition mu for shifting weights is "<<transition_mu<<endl;
  if(mu==transition_mu){
    load_into_triangle(config->late_weightsfile.data(),weights,n,n);
  }
  update_map_distance();
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
      genotypes[index] = static_cast<int>(line[i]-'1');
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
  float new_rho = mu*dist_func*rho_distance_ratio;
  cerr<<"INFER_RHO: last rho was "<<this->rho<<" proposed rho is "<<new_rho<<endl;
  return new_rho;
//  //float small = 0;
//  //float delta = .001;
//  float max_ratio = 1e-10;
//  float min_ratio = 1e10;
//  bool debug = false;
//  bool updated = false;
//  float mean_l2_norm = 0;
//  float mean_scaled_l2_norm = 0;
//  float max_l2 = 0;
//  float min_l2 = 1e10;
//  float max_scaled_l2 = 0;
//  float min_scaled_l2 = 1e10;
//  float last_rho = rho;
//  float cutoff = .1;
//  float best_cutoff = 0;
//  int counts = 0;
//  bool debug_dist = false;
//  ofstream ofs_dist;
//  if(debug_dist)ofs_dist.open("b2_norm_dist.txt");
//  if(debug_dist)ofs_dist<<"L2_norm\tweight\tscaler\n";
//  bool do_search = true;
//  if (do_search){
//    for(int index1=0;index1<n-1;++index1){
//      for(int index2=index1+1;index2<n;++index2){
//        float & weight = weights[offsets[index1]+index2-index1];
//        float & scaler  = V_project_coeff[offsets[index1]+(index2-index1)];
//        if (weight>0){
//          //cerr<<"INFER_RHO: Weight for "<<index1<<" to "<<index2<<" is "<<weight<<endl;
//          float l2norm_b = 0;
//          for(int j=0;j<p;++j){
//            float v = (U_project_orig[index1*p+j]-U_project_orig[index2*p+j]); 
//            l2norm_b+=(v*v);
//          }
//          if (l2norm_b==0){
//            scaler=1e10;
//            cerr<<"INFER_RHO: L2 norm B was "<<l2norm_b<<" weight "<<weight<<". Setting scaler for "<<index1<<","<<index2<<" to 1e10\n";
//          }else{
//            // add in a smoothing factor
//            l2norm_b=sqrt(l2norm_b+delta);
//            float scaled_l2norm_b = l2norm_b/weight;
//            if (scaled_l2norm_b>max_scaled_l2) max_scaled_l2 = scaled_l2norm_b;
//            if (scaled_l2norm_b<min_scaled_l2) min_scaled_l2 = scaled_l2norm_b;
//            if (l2norm_b>max_l2) max_l2 = l2norm_b;
//            if (l2norm_b<min_l2) min_l2 = l2norm_b;
//            mean_scaled_l2_norm+=scaled_l2norm_b;
//            mean_l2_norm+=l2norm_b;
//            ++counts;
//            scaler = mu * dist_func/scaled_l2norm_b;
//            if (fabs(scaled_l2norm_b-cutoff)<fabs(best_cutoff-cutoff)){
//              best_cutoff = scaled_l2norm_b;
//            }
//            if(debug_dist)ofs_dist<<l2norm_b<<"\t"<<weight<<"\t"<<scaler<<"\n";
//            if (scaler>max_ratio) max_ratio = scaler;
//            if (scaler<min_ratio) min_ratio = scaler;
//            if (debug) cerr<<"INFER_RHO indices "<<index1<<","<<index2<<" mu: "<<mu<<" weights: "<< weights[offsets[index1]+index2-index1]<<" dist_func: "<<dist_func<<" Scaled L2 norm_B: "<<scaled_l2norm_b<<" scaler: "<<scaler<<endl;
//            updated = true;
//          }
//        }
//      }
//    }
//  }
//  if(debug_dist)ofs_dist.close();
//  cerr<<"INFER_RHO: Mean ||b_n|| was "<<(mean_l2_norm*1./counts)<<", range ("<<min_l2<<"-"<<max_l2<<", scaled ||b_n|| was "<<(mean_l2_norm*1./counts) <<", range ("<<min_scaled_l2<<"-"<<max_scaled_l2<<") with mu ("<<mu<<") c_n was "<<dist_func<<endl;
//  cerr<<"INFER_RHO: Min, max ratio was "<<min_ratio<<","<<max_ratio<<endl;
//  
//  float decay = rho_distance_ratio;
//  cerr<<"INFER_RHO: decay set to "<<decay<<endl;
//  //float new_rho = min_ratio/(1.-decay);
//  new_rho = updated?max_ratio*(rho_distance_ratio):last_rho;
//  cerr<<"INFER_RHO: Rho set to "<<new_rho<<endl;
//  new_rho = mu*dist_func*rho_distance_ratio;
  //new_rho = mu*dist_func/best_cutoff;
//  cerr<<"INFER_RHO: Best cutoff : "<<best_cutoff<<", Rho again set to "<<new_rho<<endl;
//  return new_rho;
}


void cluster_t::init_v_project_coeff(){
  float small = 1e-10;
  int zeros = 0;
  int nonzeros = 0;
  bool debug = false;
  //float epsilon = .1;
  //float max_scaler = 1e-10;
  //float min_scaler = 1e10;
  //float mean_scaler = 0;
  //int counts=0;
  //ofstream ofs_dist;
  for(int index1=0;index1<n-1;++index1){
    for(int index2=index1+1;index2<n;++index2){
      float & weight = weights[offsets[index1]+index2-index1];
      float & scaler  = V_project_coeff[offsets[index1]+(index2-index1)];
      if (weight == 0){
        // no shrinkage
        scaler = 1;
        ++nonzeros;
      }else{
        float l2norm_b = 0;
        for(int j=0;j<p;++j){
          float v = (U_project_orig[index1*p+j]-U_project_orig[index2*p+j]); 
          l2norm_b+=(v*v);
        }
        //if(l2norm_b==0){
        if(l2norm_b<small){
          scaler = 1;
          ++nonzeros;
        }else{
          l2norm_b = sqrt(l2norm_b);
          float lambda = mu * dist_func * weight / rho;
          if (lambda<l2norm_b){
            scaler = (1.-lambda/l2norm_b);
            ++nonzeros;
          }else{
            scaler = 0;
            ++zeros;
          }
        }
        //float rescaled=(scaler<rho)?(1.-scaler/rho):0;
        //if (rescaled>max_scaler) max_scaler = rescaled;
        //if (rescaled<min_scaler) min_scaler = rescaled;
        //mean_scaler+=rescaled;
        //if (debug) cerr<<"INIT_V_COEFF indices "<<index1<<","<<index2<<" raw scaler: "<<scaler<<" and rho "<<rho<<" and revised scaler: "<<rescaled<<" mean_scaler: "<<mean_scaler<<endl;
        //++counts;
        //scaler = rescaled;
        ////if(scaler<epsilon) scaler = epsilon;
        ////if (debug) cerr<<"INIT_V_COEFF indices "<<index1<<","<<index2<<" new scaler: "<<scaler<<endl;
        ////}
        //if (scaler==0){
        //  ++zeros;
        //}else{
        //  ++nonzeros;
       // }
      }
    }
  }
  if (nonzeros==0){
    cerr<<"INIT_V_COEFF: WARNING: all cluster differences are shrunken to zero!\n";
  }else if (zeros==0){
    cerr<<"INIT_V_COEFF: WARNING: no cluster differences are shrunken to zero!\n";
  }else{
    cerr<<"INIT_V_COEFF: There are "<<nonzeros<<" non-zero and "<<zeros<<" zero coefficients for V\n";
  }
  //cerr<<"INIT_V_COEFF: Mean scaler: "<<(mean_scaler*1./counts)<<" with range ("<<min_scaler<<"-"<<max_scaler<<")"<<endl;
}

bool cluster_t::get_updated_v(int index1,int index2, float * v){
  float & scaler  =  V_project_coeff[offsets[index1]+(index2-index1)];
  if (scaler==0) return false;
  //cerr<<"In get_updated_v\n";
  int diff = 0;
  for(int j=0;j<p;++j){
    diff+=U_project_orig[index1*p+j]!=U_project_orig[index2*p+j];
    v[j] = scaler * (U_project_orig[index1*p+j]-U_project_orig[index2*p+j]);
    bool debug = j<-50;
    if(debug) cerr<<"person "<<index1<<","<<index2<<" variable "<<j<<": "<<(U_project_orig[index1*p+j]-U_project_orig[index2*p+j])<<","<<v[j]<<endl;
    //if(1==1){
    if(isnan(v[j])||isinf(v[j])){
      cerr<<"In get_updated_v between indices "<<index1<<","<<index2<<" variable "<<j<<" with scaler "<<scaler<<" taken from index "<<offsets[index1]+(index2-index1)<<"\n";
      cerr<<"U_project_orig: "<<U_project_orig[index1*p+j]<<","<<U_project_orig[index2*p+j] <<" genotype: "<<genotypes[index1*p+j]<<","<<genotypes[index2*p+j]<<endl;
      exit(1);
    }
  }
  //if (diff>0) cerr<<"GET_UPDATED_V: "<<diff<<" Differences found for "<<index1<<" and "<<index2<<endl;
  return true;
}

void cluster_t::update_projection(){
  bool debug = false;
  bool converged = false;
  float last_fnorm[n];
  for(int i=0;i<n;++i) last_fnorm[i] = 1e10;
  for(int i=0;i<n;++i){
    for(int j=0;j<p;++j){
      U_project_orig[i*p+j] = U_project[i*p+j];
      U_project[i*p+j] = U[i*p+j];
      //if (debug) cerr<<"i:"<<i<<",j:"<<j<<" "<<U_project_orig[i*p+j]<<endl;
    }
  }
  init_v_project_coeff();
  int iter = 0;
  while(!converged){
    float fnorm[n];
    for(int i=0;i<n;++i) fnorm[i] = 0;
    //if (debug) cerr<<"Projection iteration: "<<iter<<endl;
    int U_project_changes = 0;
    int U_project_remain = 0;
    for(int index = 0;index<n;++index){
      //cerr<<"For subject "<<index<<"\n";
      float v_left[p];
      float v_right[p];
      float left_summation[p];
      float right_summation[p];
      float all_summation[p];
      for(int j=0;j<p;++j){
        left_summation[j] = 0;
        right_summation[j] = 0;
        all_summation[j] = 0;
      }
      int left_neighbors=0,right_neighbors=0, neighbors = 0;
      //if (debug) cerr<<"Evaluating neighbors of "<<index<<endl;
      for(int i=0;i<n;++i){
        if(i!=index){
          float weight = i<index?weights[offsets[i]+index-i]:weights[offsets[index]+i-index];
          if(weight>0){
            //cerr<<"Subjects "<<index<<","<<i<<" have nonzero weight\n";
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
              all_summation[j]+=U_project[i*p+j];
              //all_summation[j]+=U[i*p+j];
            }
            ++neighbors;
          }
        }
      }
      for(int j=0;j<p;++j){
         float old = U_project[index*p+j];
         U_project[index*p+j] = (U[index*p+j]+right_summation[j]-left_summation[j]+all_summation[j])/(1.+neighbors);
         U_project_changes+=(old!=U_project[index*p+j]);
         U_project_remain+=(old==U_project[index*p+j]);
         fnorm[index]+=U_project[index*p+j]*U_project[index*p+j];
         if(debug) cerr<<"UPDATE_PROJECTION: index "<<index<<", var "<<j<<" Neighbors: "<<neighbors<<" U point: "<<U[index*p+j]<<" right: "<<right_summation[j]<<" left: "<<left_summation[j]<<" all: "<<all_summation[j]<<" U projection: "<<U_project[index*p+j]<<endl;
         //if (isnan(fnorm)||isinf(fnorm)) exit(1);
      }
    } // subjects looop
    //fnorm = sqrt(fnorm);
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
  cerr<<"UPDATE_PROJECTION: Converged in "<<iter<<" iterations."<<endl;
  for(int i=0;i<n;++i){
    float norm = 0;
    for(int j=0;j<p;++j){
      norm +=(U_project_orig[i*p+j]-U_project[i*p+j])*(U_project_orig[i*p+j]-U_project[i*p+j]);
      U_project_orig[i*p+j] = U_project[i*p+j];
      //if (debug) cerr<<"i:"<<i<<",j:"<<j<<" "<<U_project_orig[i*p+j]<<endl;
    }
    if(i<10||i>(n-10))cerr<<"UPDATE_PROJECTION: PROGRESS index "<<i<<":"<<norm<<"\n";
    
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
  float norm1 = 0;
  float norm2 = 0;
  // compute the frobenius norm first
  for(int i=0;i<n;++i){
    for(int j=0;j<p;++j){
      norm1+=(U[i*p+j]-U_project[i*p+j])*(U[i*p+j]-U_project[i*p+j]);
//cerr<<"U and U_project: "<<U[i*p+j]<<","<<U_project[i*p+j]<<endl;
    }
    for(int i2=(i+1);i2<n;++i2){
      for(int j=0;j<p;++j){
        //norm+=(V[i*n*p+i2*p+j]-V_project[i*n*p+i2*p+j])*(V[i*n*p+i2*p+j]-V_project[i*n*p+i2*p+j]);
        norm2+=((U[i*p+j]-U[i2*p+j]) - (U_project[i*p+j]-U_project[i2*p+j]))*
             ((U[i*p+j]-U[i2*p+j]) - (U_project[i*p+j]-U_project[i2*p+j]));
      }
    }
  }
  float norm = norm1+norm2;
  this->map_distance = norm;
  this->dist_func = sqrt(this->map_distance+epsilon);
  cerr<<"GET_MAP_DISTANCE: New map distance is "<<norm<<" with U distance="<<norm1<<", V distance="<<norm2<<" dist_func: "<<dist_func<<endl;
}

void cluster_t::update_u(){
  bool debug = false;
  int geno_changes = 0;
  int U_changes = 0;
  int mixes = 0;
  
  for(int i=0;i<n;++i){
    for(int j=0;j<p;++j){
      U[i*p+j] = dist_func*genotypes[i*p+j]/(dist_func+rho) + rho*U_project[i*p+j]/(dist_func+rho);
      if (debug && (i<10 || i>(n-10))) cerr<<"UPDATE_U: index: "<<i<<" geno: "<<genotypes[i*p+j]<<" projection: "<<U_project[i*p+j]<<" U: "<<U[i*p+j]<<endl;
      geno_changes+=(U[i*p+j]!=genotypes[i*p+j]);
      U_changes+=(U[i*p+j]!=U_project[i*p+j]);
      mixes+=((U[i*p+j]!=U_project[i*p+j]) &&(U[i*p+j]!=genotypes[i*p+j]));
    }
  }
  cerr<<"UPDATE_U: genochanges: "<<geno_changes<<" U_project changes: "<<U_changes <<" mixtures: "<<mixes<<"\n";
  update_map_distance();
}

void cluster_t::print_output(){
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
  cerr<<"IN_FEASIBLE_REGION: total V coalescent events: "<<clusters<<" U-U events "<<clusters_u<<" c_norm mean: "<<(c_mean*1./clusters)<<" range: ("<<c_min<<"-"<<c_max<<")"<<endl;
  bool ret = norm<1e-4;
  cerr<<"IN_FEASIBLE_REGION: norm "<<norm<<" differences "<<normalizer<<" returning "<<ret<<endl;
  return ret;
}

// each iteration of the projection and the point updates

void cluster_t::iterate(){
  update_projection(); 
  //update_projection_nonzero(); 
  update_u();
}

// compute the value of the augmented objective function

float cluster_t::evaluate_obj(){
  float obj = 0;
  float norm1 = 0;
  int diff1 = 0;
  for(int i=0;i<n;++i){
    for(int j=0;j<p;++j){
      norm1+=(genotypes[i*p+j]-U[i*p+j])*(genotypes[i*p+j]-U[i*p+j]);
      diff1+=genotypes[i*p+j]!=U[i*p+j];
    }
  }
  float norm2 = 0;
  for(int i1=0;i1<n-1;++i1){
    for(int i2=i1+1;i2<n;++i2){
      if (weights[offsets[i1]+i2-i1]>0){
        float v_update[p];
        if(!get_updated_v(i1,i2,v_update)){
          for(int j=0;j<p;++j) v_update[j] = 0;
        }else{
          for(int j=0;j<p;++j){
          //cerr<<"EVALUATE_OBJ: i1,i2,j: "<<i1<<","<<i2<<","<<j<<": "<<v_update[j]<<endl;
            norm2+=v_update[j]*v_update[j];
          }
          norm2 += sqrt(norm2) * weights[offsets[i1]+i2-i1]; 
        }
      }
    }
  }
  float penalty = get_prox_dist_penalty();
  obj = .5*norm1+mu*norm2+penalty;
  last_vnorm = current_vnorm;
  current_vnorm = norm2;
  if (fabs(current_vnorm-last_vnorm)/current_vnorm < .0001 && delta<1){
    //cerr<<"EVAL_OBJECTIVE: current, last Vnorms: "<<current_vnorm<<","<<last_vnorm<<", Incremented delta threshold from "<<delta<<" to ";
    //delta*=2;
    //cerr<<delta<<endl;
    //cerr<<"EVAL_OBJECTIVE: current, last Vnorms: "<<current_vnorm<<","<<last_vnorm<<", Decremented rho_distance_ratio from "<<rho_distance_ratio<<" to ";
    //rho_distance_ratio*=.9;
    //cerr<<rho_distance_ratio<<endl;
  }
  cerr<<"EVAL_OBJECTIVE: Objective is "<<obj<<" ||X-U|| is "<<norm1<<" and ||V|| is "<<norm2<<" mu penalized ||V||: "<<(mu*norm2)<<" proxdist penalty: "<<penalty<<" deviations from X: "<<diff1<<"\n";
  return obj;
}   

int main_cluster(int argc,char * argv[]){
  if(argc<2){
    cerr<<"Usage: <config file>\n";
    return 1;
  }
  //int arg=0;
  //const char * config_file = argv[++arg];
  //const char * genofile = argv[++arg];
  //const char * weightsfile = argv[++arg];
  //float rho = atof(argv[++arg]);
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

