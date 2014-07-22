#include<set>
#include<vector>
#include<iostream>
#include<sstream>
#include<fstream>
#include<cstdlib>
#include<vector>
#include<math.h>
#include<string.h>
#include<gsl/gsl_blas.h>
#include<gsl/gsl_linalg.h>
#include<gsl/gsl_eigen.h>
#include"../proxmap.hpp"
#include"regression_with_theta.hpp"

struct beta_t{
  int index;
  float val;
  float absval;

  beta_t(int index,float val){
    this->index = index;
    this->val = val;
    this->absval = fabs(val);
  };
};

struct byValDesc{
  bool operator()(const beta_t & a,const beta_t & b){
    return a.absval>b.absval;
  }
};

regression_with_theta_t::regression_with_theta_t(bool single_run){
  this->single_run = single_run;
  this->total_iterations = 0;
}

regression_with_theta_t::~regression_with_theta_t(){
#ifdef USE_MPI
  if(slave_id>=0){
    delete [] X;
    delete [] XT;
    delete [] y;
    delete [] XXI_inv;
  }
  delete [] node_sizes;
  delete [] node_offsets;
  delete [] active_set;
  delete [] beta;
  delete [] beta_project;
  delete [] constrained_beta;
  delete [] theta;
  delete [] theta_project;
  delete [] Xbeta;
  delete [] lambda;
  if(this->single_run){
    MPI::Finalize();
  } 
  if(config->debug_mpi){
    ofs_debug.close();
  }
#endif
}


void regression_with_theta_t::update_Xbeta(float * beta){
  // PN operations
  bool active[variables];
  for(int j=0;j<variables;++j){
    active[j]= beta[j]!=0;
  }
  //mmultiply(X,observations,variables,beta,1,Xbeta);
  for(int i=0;i<observations;++i){
    Xbeta[i] = 0;
    for(int j=0;j<variables;++j){
      if (active[j]) Xbeta[i]+=X[i*variables+j] * beta[j];
    }
    
  }
}



void regression_with_theta_t::update_lambda(){
#ifdef USE_MPI
  //double start = clock();
  float theta_norm = 0;
  float deviance_norm = 0;
  float lambda_norm = 0;
  bool debug = true;
  if(slave_id>=0){
    //ofs_debug<<"Updating lambda\n";
    update_Xbeta(this->beta);
    //update_Xbeta(this->constrained_beta);
    float xbetatheta[observations];
    for(int i=0;i<observations;++i){
      xbetatheta[i] = Xbeta[i]-theta[i];
      if (debug) ofs_debug<<"UPDATE_LAMBDA: "<<i<<": "<<Xbeta[i]<<","<<theta[i]<<","<<xbetatheta[i]<<endl;
    }
    // NN operations
    mmultiply(XXI_inv,observations,observations,xbetatheta,1,lambda);
    for(int i=0;i<observations;++i){
      //if(total_iterations<=1 && i<10) cerr<<i<<":"<<xbetatheta[i]<<","<<lambda[i]<<endl;
      deviance_norm+=(Xbeta[i]-theta[i])*(Xbeta[i]-theta[i]);
      lambda_norm+=lambda[i]*lambda[i];
      theta_norm+=theta[i]*theta[i];
    }
    deviance_norm=sqrt(deviance_norm);
    lambda_norm=sqrt(lambda_norm);
    theta_norm=sqrt(theta_norm);
    // total operations: PN + NN
  }
  float lambda_norms[mpi_numtasks];
  MPI_Gather(&lambda_norm,1,MPI_FLOAT,lambda_norms,1,MPI_FLOAT,0,MPI_COMM_WORLD);
  float deviance_norms[mpi_numtasks];
  MPI_Gather(&deviance_norm,1,MPI_FLOAT,deviance_norms,1,MPI_FLOAT,0,MPI_COMM_WORLD);
  float theta_norms[mpi_numtasks];
  MPI_Gather(&theta_norm,1,MPI_FLOAT,theta_norms,1,MPI_FLOAT,0,MPI_COMM_WORLD);
  //if(config->verbose) cerr<<"BENCHMARK UPDATE_LAMBDA: "<<(clock()-start)/CLOCKS_PER_SEC<<endl;
  for(int i=1;i<mpi_numtasks;++i){
    if(config->verbose)cerr<<"UPDATE_LAMBDA: Norms for node "<<i<<" theta: "<<theta_norms[i]<<" lambda: "<<lambda_norms[i]<<" deviance: "<<deviance_norms[i]<<endl;
  }
  // this method implements a Landweber update
  //float gradient[observations];
  //for(int iter=0;iter<10;++iter){
  //  mmultiply(XXTI,observations,observations,lambda,1,gradient);
  //  for(int i=0;i<observations;++i){
  //    gradient[i] += theta[i] - Xbeta[i];
  //    lambda[i]-=gradient[i]/L;
  //    //if (i<3) ofs_debug<<"Gradient/Lambda "<<i<<" is "<<gradient[i]<<"/"<<lambda[i]<<endl;
  //  }  
 // }
#endif
}


void regression_with_theta_t::project_beta(){
  //double start = clock();
  bool debug = true;
  if(slave_id>=0){
    if (1==2){
    //if (in_feasible_region()){
      //cerr<<"PROJECT_BETA: slave "<<slave_id<<" is feasible\n";
      for(int j=0;j<variables;++j){
        float xt_lambda = 0;
        if(constrained_beta[j]!=0){
          for(int i=0;i<observations;++i){
            xt_lambda+=XT[j*observations+i] * lambda[i];
          }
          beta_project[j] = beta[j]-xt_lambda;
        }else{
          beta_project[j] = 0;
        }
        //if (debug) ofs_debug<<"PROJECT_BETA: var "<<j<<" is "<<beta_project[j]<<endl;
      }
    }else{
      //cerr<<"PROJECT_BETA: slave "<<slave_id<<" is not feasible\n";
      float xt_lambda[variables];
      // NP operations
      mmultiply(XT,variables,observations,lambda,1,xt_lambda);
      //float before_norm = 0, after_norm = 0;
      for(int j=0;j<variables;++j){
        //before_norm+=fabs(beta_project[j]);
        beta_project[j] = beta[j]-xt_lambda[j];
        //cerr<<"PROJECT_BETA: var "<<j<<" is "<<beta_project[j]<<endl;
     
        //after_norm+=fabs(beta_project[j]);
        //beta[j] = slave_id*variables+j;
        if (debug) ofs_debug<<"PROJECT_BETA: var "<<j<<" is "<<beta_project[j]<<endl;
      }
      // total operations: NP
      //cerr<<slave_id<<":PROJECTBETA: before,after: "<<before_norm<<","<<after_norm<<endl;
    }
  }
  //MPI_Barrier(MPI_COMM_WORLD);
  //if(config->verbose) cerr<<"BENCHMARK PROJECT_BETA: "<<(clock()-start)/CLOCKS_PER_SEC<<endl;
}

void regression_with_theta_t::project_theta(){
#ifdef USE_MPI
  //double start = clock();
  bool debug = true;
  float theta_project_norm = 0;
  if(slave_id>=0){
    for(int i=0;i<observations;++i){
      theta_project[i] = theta[i]+lambda[i];
      if (debug) ofs_debug<<"PROJECT_THETA: Subject "<<i<<" is "<<theta_project[i]<<endl;
      //if(i<10)cerr<<"Thetaproject "<<i<<" "<<theta_project[i]<<endl;
      theta_project_norm+=theta_project[i]*theta_project[i];
    }
    theta_project_norm = sqrt(theta_project_norm);
  }
  float theta_project_norms[mpi_numtasks];
  MPI_Gather(&theta_project_norm,1,MPI_FLOAT,theta_project_norms,1,MPI_FLOAT,0,MPI_COMM_WORLD);
  for(int i=1;i<mpi_numtasks;++i){
    if(config->verbose)cerr<<"PROJECT_THETA: For node "<<i<<" theta project norm: "<<theta_project_norms[i]<<endl;
  }
  //if(config->verbose) cerr<<"BENCHMARK PROJECT_THETA: "<<(clock()-start)/CLOCKS_PER_SEC<<endl;
#endif
}

void regression_with_theta_t::update_map_distance(){
#ifdef USE_MPI
  //double start = clock();
  beta_distance = 0;
  theta_distance = 0;
  if(slave_id>=0){
    for(int j=0;j<variables;++j){
      float dev = (beta[j]-beta_project[j]);
      beta_distance+=(dev*dev);
    }
    ofs_debug<<"UPDATE_MAP_DISTANCE: beta_distance: "<<beta_distance<<endl;
    for(int i=0;i<observations;++i){
      float dev = (theta[i]-theta_project[i]);
      theta_distance+=(dev*dev);
    }
    ofs_debug<<"UPDATE_MAP_DISTANCE: theta_distance: "<<theta_distance<<endl;
  }
  float beta_distance_reduce,theta_distance_reduce;
  MPI_Reduce(&beta_distance,&beta_distance_reduce,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Reduce(&theta_distance,&theta_distance_reduce,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Gatherv(beta,variables,MPI_FLOAT,beta,node_sizes,node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
  //ofs_debug<<"UPDATE_MAP_DISTANCE: sorting constrained beta\n";
  if(mpi_rank==0){
    multiset<beta_t,byValDesc> sorted_beta;
    for(int j=0;j<variables;++j){
      beta_t b(j,beta[j]);
      sorted_beta.insert(b);
    }
    int j=0;
    float constraint_dev=0;
    for(multiset<beta_t,byValDesc>::iterator it=sorted_beta.begin();it!=sorted_beta.end();it++){
      beta_t b = *it;
      if (j<top_k){
        constrained_beta[b.index] = b.val;
      }else{
        constrained_beta[b.index] = 0;
        //cerr<<"Adding "<<b.val<<" for var "<<b.index<<endl;
        constraint_dev+=b.val*b.val;
      }
      ++j;
    }
    //float inflater = 1;
    if (rho>=config->rho_max){
      //constraint_dev*=inflater;
      //beta_distance_reduce*=inflater;
      //constraint_dev*=2*config->rho_distance_ratio;
      //beta_distance_reduce*=2*(1.-config->rho_distance_ratio);
    }
    for(int j=0;j<variables;++j){
      //ofs_debug<<"Beta "<<j<<" is "<<beta[j]<<endl;
    }
    this->map_distance = beta_distance_reduce + theta_distance_reduce + constraint_dev;
    if(config->verbose) cerr<<"UPDATE_MAP_DISTANCE: Deviances: beta:"<<beta_distance_reduce<<" theta:"<<theta_distance_reduce<<" constraint:"<<constraint_dev<<endl;
    cerr<<"UPDATE_MAP_DISTANCE: Deviances: beta:"<<beta_distance_reduce<<" theta:"<<theta_distance_reduce<<" constraint:"<<constraint_dev<<endl;
    //if (iter_rho_epsilon==0) this->map_distance = 1e5;
    this->dist_func = rho/sqrt(this->map_distance+epsilon);
    ofs_debug<<"Map distance is now "<<this->dist_func<<endl;
  }

  MPI_Bcast(&this->map_distance,1,MPI_FLOAT,0,MPI_COMM_WORLD);
  //MPI_Barrier(MPI_COMM_WORLD);
  //if(config->verbose) cerr<<"BENCHMARK UPDATE_MAP_DIST: "<<(clock()-start)/CLOCKS_PER_SEC<<endl;
#endif
}

float regression_with_theta_t::get_map_distance(){
  return this->map_distance;

//  float beta_dev = 0;
//  float theta_dev = 0;
//  for(int j=0;j<variables;++j){
//    beta_dev+=(beta[j]-beta_project[j])*(beta[j]-beta_project[j]);
//  }
//  beta_dev = sqrt(beta_dev);
//  for(int i=0;i<observations;++i){
//    theta_dev+=(theta[i]-theta_project[i])*(theta[i]-theta_project[i]);
//  }
//  theta_dev = sqrt(theta_dev);
//  return beta_dev+theta_dev; 
}

void regression_with_theta_t::update_theta(){
#ifdef USE_MPI
  //double start = clock();
  bool debug = true;
  float theta_project_reduce[observations];
  MPI_Reduce(theta_project,theta_project_reduce,observations,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Bcast(theta_project_reduce,1,floatSubjectsArrayType,0,MPI_COMM_WORLD);
  MPI_Bcast(&dist_func,1,MPI_FLOAT,0,MPI_COMM_WORLD);
  if(slave_id>=0){
    float coeff = 1./(slaves+dist_func);
    for(int i=0;i<observations;++i){
      theta[i] = theta_project[i]+coeff*(y[i]-theta_project_reduce[i]);
      //cerr<<"Subject "<<i<<" theta "<<theta[i]<<" projection "<<theta_project[i]<<endl;

      if (debug)ofs_debug<<"UPDATE_THETA: reduction: "<<theta_project_reduce[i]<<" coeff: "<<coeff<<" theta["<<i<<"]:" <<theta[i]<<endl;
    }
  }
  float theta_reduce[observations];
  if (mpi_rank==0) for(int i=0;i<observations;++i) theta[i] = 0;
  MPI_Reduce(theta,theta_reduce,observations,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
  if (mpi_rank==0) for(int i=0;i<observations;++i) {
    theta[i] = theta_reduce[i];
    if (debug)ofs_debug<<"UPDATE_THETA for i: "<<theta[i]<<endl;
  }
  //if(config->verbose) cerr<<"BENCHMARK UPDATE_THETA: "<<(clock()-start)/CLOCKS_PER_SEC<<endl;
#endif

}

void regression_with_theta_t::update_beta(){
#ifdef USE_MPI
  //double start = clock();
  bool debug = true;
  if(mpi_rank==0){
    for(int j=0;j<variables;++j){
      //constrained_beta[j] = j;
      //ofs_debug<<"UPDATE_BETA for "<<j<<" constrained: "<<constrained_beta[j]<<" and final "<<beta[j]<<endl;
    }
  }
  MPI_Scatterv(constrained_beta,node_sizes,node_offsets,MPI_FLOAT,constrained_beta,variables,MPI_FLOAT,0,MPI_COMM_WORLD);
  
  if(slave_id>=0){
    if(1==2){
    //if(in_feasible_region()){
      //cerr<<"UPDATE_BETA: slave "<<slave_id<<" is feasible so beta = beta_project\n";
      for(int j=0;j<variables;++j){
        beta[j] = constrained_beta[j]!=0?beta_project[j]:0;
      }
    }else{
      //cerr<<"UPDATE_BETA: slave "<<slave_id<<" is not feasible\n";
      for(int j=0;j<variables;++j){
      //if(rho>=config->rho_max){
        //beta[j] = (1.-rho_distance_ratio) * beta_project[j]+(rho_distance_ratio*constrained_beta[j]);
      //}else{
          beta[j] = .5*(beta_project[j]+constrained_beta[j]);
        //cerr<<"UPDATE_BETA for "<<j<<" is project "<<beta_project[j]<<" constrained "<<constrained_beta[j]<<" and final "<<beta[j]<<endl;
      //}
        if (debug) ofs_debug<<"UPDATE_BETA for "<<j<<" is project "<<beta_project[j]<<" constrained "<<constrained_beta[j]<<" and final "<<beta[j]<<endl;
      }
    }
  }
  MPI_Gatherv(beta,variables,MPI_FLOAT,beta,node_sizes,node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
  if(mpi_rank==0){
    for(int j=0;j<variables;++j){
      if (debug)ofs_debug<<"UPDATE_BETA for "<<j<<": constrained="<<constrained_beta[j]<<" beta="<<beta[j]<<endl;
    }
  }
  //if(config->verbose) cerr<<"BENCHMARK UPDATE_BETA: "<<(clock()-start)/CLOCKS_PER_SEC<<endl;
#endif
}

void regression_with_theta_t::loss(){
  // not used
  float l=0;
  for(int i=0;i<observations;++i){
    l+=(y[i]-Xbeta[i])*(y[i]-Xbeta[i]);
  }
  l*=.5;
  //ofs_debug<<"Loss is "<<l<<endl;
}

void regression_with_theta_t::check_constraints(){
}

bool regression_with_theta_t::in_feasible_region(){
  float mapdist = get_map_distance();
  bool ret= (mapdist>0 && mapdist<1e-6);
  //cerr<<"IN_FEASIBLE_REGION: node "<<mpi_rank<<" returning "<<ret<<endl;
  return ret;
}

void regression_with_theta_t::parse_config_line(string & token,istringstream & iss){
  proxmap_t::parse_config_line(token,iss);
  if (token.compare("TRAIT")==0){
    iss>>config->traitfile;
  }else if (token.compare("TASKFILE")==0){
    iss>>config->taskfile;
  }else if (token.compare("TOP_K")==0){
    iss>>config->top_k;
  }else if (token.compare("MARGINAL_FILE_PREFIX")==0){
    iss>>config->marginal_file_prefix;
  }else if (token.compare("XXI_INV_FILE_PREFIX")==0){
    iss>>config->xxi_inv_file_prefix;
  }else if (token.compare("BETA_EPSILON")==0){
    iss>>config->beta_epsilon;
  }else if (token.compare("DEBUG_MPI")==0){
    iss>>config->debug_mpi;
  }
}

void testsvd2(int mpi_rank){
  bool testsvd = false;
  if (testsvd && mpi_rank==0){
    gsl_matrix * tempx = gsl_matrix_alloc(3,4);
    gsl_matrix * tempxx = gsl_matrix_alloc(3,3);
    FILE * fp = fopen("testmat.txt","r");
    int rc =gsl_matrix_fscanf(fp,tempx);
    if (rc!=0) cerr<<"Failed to read file\n";
    fclose(fp);
    for(int i=0;i<3;++i){
      for(int j=0;j<4;++j){
        //cerr<<i<<","<<j<<":"<<gsl_matrix_get(tempx,i,j)<<endl;
      }
    }
    gsl_blas_dgemm(CblasNoTrans,CblasTrans,1,tempx,tempx,0,tempxx);
    for(int i=0;i<3;++i){
      for(int j=0;j<3;++j){
        //cerr<<i<<","<<j<<":"<<gsl_matrix_get(tempxx,i,j)<<endl;
      }
    }
    gsl_matrix * tempv = gsl_matrix_alloc(3,3);
    gsl_vector * temps = gsl_vector_alloc(3);
    gsl_vector * work  = gsl_vector_alloc(3);
    gsl_linalg_SV_decomp(tempxx,tempv,temps,work);
    cerr<<"S:"<<endl;
    for(int i=0;i<3;++i){
      cerr<<" "<<gsl_vector_get(temps,i);
    }
    cerr<<endl;
    cerr<<"V:"<<endl;
    for(int i=0;i<3;++i){
      for(int j=0;j<3;++j){
        cerr<<" "<<gsl_matrix_get(tempv,i,j);
      }
      cerr<<endl;
    }
  }
}

void regression_with_theta_t::read_dataset(){
#ifdef USE_MPI
  // figure out the dimensions
  //
  const char * genofile = config->genofile.data();
  const char * phenofile = config->traitfile.data();
  this->observations = linecount(phenofile);
  int variables = colcount(genofile);
  ofs_debug<<"Subjects: "<<observations<<" and predictors: "<<variables<<endl;
  // taken from Eric Chi's project
  bool single_mask[] = {true};
  bool subject_mask[observations];
  for(int i=0;i<observations;++i) subject_mask[i] = true;
  bool variable_mask[variables];
  for(int i=0;i<variables;++i) variable_mask[i] = slave_id<0?true:false;
  vector<vector<int> > indices_vec;
  ifstream ifs_task(config->taskfile.data());

  // assign variable indices to each slave
  // 
  if (ifs_task.is_open()){
    string line;
    while(getline(ifs_task,line)){
      istringstream iss(line);
      int var_index;
      vector<int> indices;
      while(iss>>var_index){
        indices.push_back(var_index);
      }
      indices_vec.push_back(indices);
    }
    if (indices_vec.size()!=slaves){
      ofs_debug<<"There are "<<slaves<<" slaves, but only "<<indices_vec.size()<<
      " rows in the task file "<<config->taskfile<<
      ".  Be sure that the number of rows match the number of slaves.\n";
      MPI_Finalize();
      exit(1);
    }
    ifs_task.close();
  }else{
    ofs_debug<<"Slave task file "<<config->taskfile<<
    " not found.  Using default of equal sized chunks.\n";
    // this section divides the variables into equal sized blocks that
    // are each handled by a MPI slave
    // in the future, this can be 
    int variables_per_slave = variables/slaves;
    int i = 0;
    //bool debug = false;
    for(uint j=0;j<slaves;++j){
      vector<int> indices;
      uint total_variables = (j<slaves-1)?variables_per_slave:
      (variables-(slaves-1)*variables_per_slave);
      ofs_debug<<"Total variables for slave "<<j<<" is "<<total_variables<<endl;  
      for(uint k=0;k<total_variables;++k){
        indices.push_back(i);
        //if(debug && slave_id==j) ofs_debug<<"Slave "<<j<<" fitting index "<<i<<endl;
        ++i;
      }
      indices_vec.push_back(indices);
    }
  }

  // now populate the variable mask for input with the assigned indices
  int slave_variables = slave_id<0?variables:indices_vec[slave_id].size();
  if (slave_id>=0){
    for(uint i=0;i<indices_vec[slave_id].size();++i){
      variable_mask[indices_vec[slave_id][i]] = true;
    }
  }
  //floatVariablesArrayType = new MPI_Datatype[slaves];
  node_sizes[0] = 0;
  node_offsets[0] = 0;
  int offset = 0;
  for(uint i=0;i<slaves;++i){
    node_sizes[i+1] =  indices_vec[i].size();
    node_offsets[i+1] = offset;
    offset+=node_sizes[i+1];
    //MPI_Type_contiguous(slave_sizes[i],MPI_FLOAT,floatVariablesArrayType+i);
    //MPI_Type_commit(floatVariablesArrayType+i);
  }

  MPI_Type_contiguous(observations,MPI_FLOAT,&floatSubjectsArrayType);
  MPI_Type_commit(&floatSubjectsArrayType);
  // master and slaves share same number of observations
  load_matrix_data(config->traitfile.data(),y,observations,1,observations,1,subject_mask, single_mask,true,0);
  this->variables=slave_id>=0?slave_variables:variables;
  ofs_debug<<"Node "<<mpi_rank<<" with "<<this->variables<<" variables.\n";
  if(slave_id>=0){
    load_matrix_data(genofile,X,observations,variables,observations,slave_variables, subject_mask,variable_mask,true,0);
    // At this point we should standardize all the variables
    bool standardize = false;
    if(standardize){
      ofs_debug<<"Standardizing variables\n";
      ofs_debug.flush();
  
      for(int j=0;j<this->variables;++j){ 
        float tempvec[observations];
        float mean = 0;
        for(int i=0;i<observations;++i){
          tempvec[i] = X[i*this->variables+j];
          mean+=tempvec[i];
        }
        mean/=observations;
        float sd = 0;
        for(int i=0;i<observations;++i){
          sd+=(tempvec[i]-mean)*(tempvec[i]-mean);
        }
        if (sd>0){
          sd = sqrt(sd/(observations-1));
          for(int i=0;i<observations;++i){
            X[i*this->variables+j] = (tempvec[i]-mean)/sd;
          }
        }else{
          for(int i=0;i<observations;++i){
            X[i*this->variables+j] = 0;
          }
        }
      }
    }
    // Also, set up X transpose too, a constant matrix variable
    this->XT = new float[this->variables*observations];
    for(int j=0;j<this->variables;++j){
      for(int i=0;i<observations;++i){
        XT[j*observations+i] = X[i*this->variables+j];
      }
    }
  }


  // SANITY CHECK THAT WE READ IN THE DATASET CORRECTLY
  //
  bool debugoutput=false;
  if(debugoutput){
    ostringstream oss;
    oss<<"OUTPUT."<<mpi_rank<<".txt";
    ofstream ofs(oss.str().data());
    for(int i=0;i<observations;++i){
      ofs<<y[i]<<"\t";
      for(int j=0;j<this->variables;++j){
        ofs<<X[i*this->variables+j];
      }
      ofs<<endl;
    }
    ofs.close();
  }
#endif
}

float regression_with_theta_t::compute_marginal_beta(float * xvec){
  // dot product first;
  float xxi = 0;
  float xy = 0;
  for(int i=0;i<observations;++i){
    xxi +=xvec[i]*xvec[i];
    xy +=xvec[i]*y[i];
  }
  if(fabs(xxi)<1e-5) return 0;
  xxi = 1./xxi;
  return xxi*xy;
}

void regression_with_theta_t::init_marginal_screen(){
  ostringstream oss_marginal;
  oss_marginal<<config->marginal_file_prefix<<"."<<mpi_rank<<".txt";
  ifstream ifs_marginal(oss_marginal.str().data());
  if (ifs_marginal.is_open()){
    if(config->verbose) cerr<<"Using cached copy of marginal betas\n";
    string line;
    for(int j=0;j<variables;++j){
      getline(ifs_marginal,line);
      istringstream iss(line);
      iss>>beta[j];
    }
    ifs_marginal.close();
  }else{
    if(config->verbose) cerr<<"Creating cached copy of marginal betas\n";
    ofstream ofs_marginal(oss_marginal.str().data());
    for(int j=0;j<variables;++j){
      beta[j] = compute_marginal_beta(XT+j*observations);
      ofs_marginal<<beta[j]<<endl; 
    }
    ofs_marginal.close();
  }
  update_Xbeta(beta);
  float xbeta_norm = 0;
  for(int i=0;i<this->observations;++i){
    //theta_project[i] = theta[i] = y[i];
    xbeta_norm+=Xbeta[i]*Xbeta[i];
    Xbeta[i] = 0;
    theta_project[i] = theta[i] = Xbeta[i];
  }
  xbeta_norm=sqrt(xbeta_norm);
  cerr<<"Xbeta norm for node "<<mpi_rank<<" is "<<xbeta_norm<<endl;
  for(int j=0;j<variables;++j){
    constrained_beta[j] = beta_project[j] = beta[j] = 0;
  }
}

void regression_with_theta_t::init_xxi_inv(){
    XXI_inv = new float[observations*observations];
    ostringstream oss_xxi_inv;
    oss_xxi_inv<<config->xxi_inv_file_prefix<<"."<<mpi_rank<<".txt";
    ifstream ifs_xxi_inv(oss_xxi_inv.str().data());
    if (ifs_xxi_inv.is_open()){
      if(config->verbose) cerr<<"Using cached copy of singular values and vectors\n";
      ofs_debug<<"Using cached copy of singular values and vectors\n";
      ofs_debug.flush();
      string line;
      for(int i=0;i<observations;++i){
        getline(ifs_xxi_inv,line);
        istringstream issvec(line);
        for(int j=0;j<observations;++j){
          issvec>>XXI_inv[i*observations+j];
        }
      }
      ifs_xxi_inv.close();
    }else{
      //int rc = 0;
      //float * singular_vectors = new float[observations*observations];
      //float * singular_values = new float[observations];
      if(config->verbose) cerr<<"Cannot find cached copy of singular values and vectors. Computing\n";
      ofs_debug<<"Cannot find cached copy of singular values and vectors. Computing\n";
      ofs_debug.flush();
      gsl_matrix * tempx = gsl_matrix_alloc(observations,this->variables);
      gsl_matrix * tempxx = gsl_matrix_alloc(observations,observations);
      ofs_debug<<"Copying into gsl matrix\n";
      for(int i=0;i<observations;++i){
        for(int j=0;j<this->variables;++j){
          gsl_matrix_set(tempx,i,j,X[i*this->variables+j]);
        }
      }
      ofs_debug<<"Computing X%*%X^T\n"; 
      ofs_debug.flush();
      gsl_blas_dgemm(CblasNoTrans,CblasTrans,1,tempx,tempx,0,tempxx);
      if(mpi_rank==-1){
      for(int i=0;i<observations;++i){
        for(int j=0;j<this->variables;++j){
          if (j) cerr<<" ";
          cerr<<gsl_matrix_get(tempxx,i,j);
        }
        cerr<<endl;
      }

      }
      ofs_debug<<"Performing eigen decomp\n";
      ofs_debug.flush();
      //This function allocates a workspace for computing eigenvalues and eigenvectors of n-by-n real symmetric matrices. The size of the workspace is O(4n).
      gsl_eigen_symmv_workspace * eigen_work =  gsl_eigen_symmv_alloc (observations);
      gsl_matrix * tempv = gsl_matrix_alloc(observations,observations);
      gsl_vector * temps = gsl_vector_alloc(observations);
      //This function computes the eigenvalues and eigenvectors of the real symmetric matrix A. Additional workspace of the appropriate size must be provided in w. The diagonal and lower triangular part of A are destroyed during the computation, but the strict upper triangular part is not referenced. The eigenvalues are stored in the vector eval and are unordered. The corresponding eigenvectors are stored in the columns of the matrix evec. For example, the eigenvector in the first column corresponds to the first eigenvalue. The eigenvectors are guaranteed to be mutually orthogonal and normalised to unit magnitude.
        
      int code = gsl_eigen_symmv(tempxx, temps, tempv, eigen_work);
      if (code!=0) ofs_debug<<"Returning nonzero code in eigendecomp\n";
      gsl_matrix * tempvdi = gsl_matrix_alloc(observations,observations);
      for(int i=0;i<observations;++i){
        for(int j=0;j<observations;++j){
          //ofs_debug<<"Eigen "<<j<<","<<i<<":"<<gsl_matrix_get(tempv,j,i)<<endl;
          //ofs_debug.flush();
          float vdi = gsl_matrix_get(tempv,i,j)/(1.+gsl_vector_get(temps,j));
          gsl_matrix_set(tempvdi,i,j,vdi);
        }
      }
      if(mpi_rank==-1){
        ofstream ofs_rtest0("r_test0.txt");
        for(int i=0;i<observations;++i){
          for(int j=0;j<variables;++j){
            if (j) ofs_rtest0<<"\t";
            ofs_rtest0<<gsl_matrix_get(tempx,i,j);
          }
          ofs_rtest0<<endl;
        }
        ofstream ofs_rtest1("r_test1.txt");
        ofstream ofs_rtest2("r_test2.txt");
        ofstream ofs_rtest3("r_test3.txt");
        for(int i=0;i<observations;++i){
          for(int j=0;j<observations;++j){
            if (j) ofs_rtest1<<"\t";
            ofs_rtest1<<gsl_matrix_get(tempxx,i,j);
            if (i==j) ofs_rtest2<<gsl_vector_get(temps,i);
            if (j) ofs_rtest3<<"\t";
            ofs_rtest3<<gsl_matrix_get(tempv,i,j);
            //ofs_rtest2<<gsl_matrix_get(tempvdi,i,j);
          }
          ofs_rtest1<<endl;
          ofs_rtest2<<endl;
          ofs_rtest3<<endl;
        }
        ofs_rtest0.close();
        ofs_rtest1.close();
        ofs_rtest2.close();
        ofs_rtest3.close();
        exit(1);
      }
      //
      //Function: void gsl_eigen_symmv_free (gsl_eigen_symmv_workspace * w)
      //This function frees the memory associated with the workspace w.
      //
      gsl_matrix_free(tempxx);
      gsl_vector_free(temps);
      gsl_eigen_symmv_free(eigen_work);

      //ofs_debug<<"Performing SVD\n";
      //ofs_debug.flush();
      //cerr<<"Variables: "<<variables<<" observations: "<<observations<<endl;
      //gsl_matrix * tempv_svd = gsl_matrix_alloc(variables,variables);
      //gsl_vector * temps_svd = gsl_vector_alloc(variables);
      //gsl_vector * work_svd  = gsl_vector_alloc(variables);
      //gsl_linalg_SV_decomp(tempx,tempv_svd,temps_svd,work_svd);
      //gsl_vector_free(work_svd);
      for(int j=0;j<this->variables;++j){
        for(int i=0;i<observations;++i){
          //ofs_debug<<"SVD "<<j<<","<<i<<":"<<gsl_matrix_get(tempx,i,j)<<endl;
          //ofs_debug.flush();
//          //float vdi = gsl_matrix_get(tempx,i,j)/(1.+gsl_vector_get(temps,j));
//          //gsl_matrix_set(tempvdi,i,j,vdi);
        }
      }
      gsl_matrix_free(tempx);
      //gsl_matrix_free(tempv_svd);
      //gsl_vector_free(temps_svd);
      ofs_debug<<"Computing VDIV\n";
      ofs_debug.flush();
      gsl_matrix * tempvdiv = gsl_matrix_alloc(observations,observations);
      gsl_blas_dgemm(CblasNoTrans,CblasTrans,1,tempvdi,tempv,0,tempvdiv);
      gsl_matrix_free(tempvdi);
      gsl_matrix_free(tempv);
      ofs_debug<<"Writing to file "<<oss_xxi_inv.str()<<"\n";
      ofs_debug.flush();
      ofstream ofs_xxi_inv(oss_xxi_inv.str().data());
      for(int i=0;i<observations;++i){
        for(int j=0;j<observations;++j){
          if (j) ofs_xxi_inv<<"\t";
          XXI_inv[i*observations+j] = gsl_matrix_get(tempvdiv,i,j);
          ofs_xxi_inv<<XXI_inv[i*observations+j];
        }
        ofs_xxi_inv<<endl;
      }
      ofs_xxi_inv.close();
      gsl_matrix_free(tempvdiv);
    }
}

void regression_with_theta_t::init(string config_file){
#ifdef USE_MPI
  if(this->single_run){
    MPI::Init();
  }
  this->mpi_numtasks = MPI::COMM_WORLD.Get_size();
  this->slaves = mpi_numtasks-1;
  this->mpi_rank = MPI::COMM_WORLD.Get_rank();
  this->slave_id = mpi_rank-1;
  config->beta_epsilon = 1e-3;
  config->marginal_file_prefix = "marginal";
#endif
  proxmap_t::init(config_file);
  if (this->slave_id>=0)  config->verbose = false;
}

void regression_with_theta_t::allocate_memory(){
#ifdef USE_MPI
  ostringstream oss_debugfile;
  oss_debugfile<<"debug.rank."<<mpi_rank;
  if (config->debug_mpi){
    ofs_debug.open(oss_debugfile.str().data());
    if (ofs_debug.is_open()){
      cerr<<"Successfully opened debug file\n";
    }else{
      cerr<<"Could not open debug file\n";
    }
  }
  // initialize MPI data structures
  this->track_residual = false;
  this->node_sizes = new int[mpi_numtasks];
  this->node_offsets = new int[mpi_numtasks];
  ofs_debug<<"Initialized MPI with "<<slaves<<" slaves.  I am slave "<<slave_id<<endl;
  read_dataset();

  // ALLOCATE MEMORY FOR ESTIMATED PARAMETERS
  //
  this->last_residual = 1e10;
  this->residual = 0;
  this->active_set_size = 0;
  this->active_set = new bool[this->variables];
  this->beta = new float[this->variables];
  this->last_beta = new float[this->variables];
  this->beta_project = new float[this->variables];
  this->constrained_beta = new float[this->variables];
  this->theta = new float[observations];
  this->theta_project = new float[observations];
  this->Xbeta = new float[observations];
  // the LaGrange multiplier
  this->lambda = new float[this->observations];

  // IN THIS SECTION WE INITIALIZE (X^T%&%X+I)^-1 WITH THE SVD
  //
  if(slave_id>=0){
    // init XX inverse
    init_xxi_inv();
    // init marginal screen
    //init_marginal_screen();
  }
  proxmap_t::allocate_memory();
#endif
//  blocks = 1;
//  rho = 1;
//  X = new float[observations*variables];
//  y = new float[observations];
//  nu = new float[variables];
//  beta = new float[variables];
//  beta_project = new float[variables];
//  // the Lagrangian vector
//  lambda = new float[observations];
//  for(int i=0;i<observations;++i) {
//    theta_project[i] = theta[i] = 0;
//    Xbeta[i] = 0;
//    lambda[i] = 0;
//  }
//  for(int j=0;j<variables;++j){
//    beta_project[j] = beta[j] = 0;
//    nu[j] = 0;
//  }
//  
//  load_into_matrix(genofile,X,observations,variables);
//  for(int j=0;j<variables;++j){
//    for(int i=0;i<observations;++i){
//      XT[j*observations+i] = X[i*variables+j];
//    }
//  }
//  XXT = new float[observations*observations];
//  mmultiply(X,observations,variables,XT,observations,XXT);
//  XXTI = new float[observations*observations];
//  load_into_matrix(phenofile,y,observations,1);
//  // the parameter vector for the means Xbeta
//  float row_sums[observations];
//  float col_sums[variables];
//  for(int j=0;j<variables;++j) col_sums[j]=0;
//  for(int i=0;i<observations;++i){
//    row_sums[i] = 0;
//    for(int j=0;j<variables;++j){
//      row_sums[i]+=X[i*variables+j];
//      col_sums[j]+=X[i*variables+j];
//    }
//  }
//  float maxr=-1000,maxc=-1000;
//  for(int i=0;i<observations;++i) if (row_sums[i]>maxr)maxr = row_sums[i];
//  for(int j=0;j<variables;++j) if (col_sums[j]>maxc)maxc = col_sums[j];
//  L = 1 + maxr*maxc ;
//  ofs_debug<<"Max rowsum, colsum, and L: "<<maxr<<","<<maxc<<","<<L<<endl;
//  for(int i1=0;i1<observations;++i1){
//    for(int i2=0;i2<observations;++i2){
//      XXTI[i1*observations+i2] = (i1==i2)? XXT[i1*observations+i2]+1:XXT[i1*observations+i2];
//    }
//  }
}
float regression_with_theta_t::infer_epsilon(){
  float new_epsilon=0;
#ifdef USE_MPI
  if (mu==0) return config->epsilon_max;
  //else if(mu==config->mu_min) return config->epsilon_max;
  //float scaler = this->mu;
  if (mpi_rank==0){
    ofs_debug<<"INFER_EPSILON: Inner iterate: "<<iter_rho_epsilon<<endl;
    new_epsilon = last_epsilon;
    if(iter_rho_epsilon==0 && track_residual && new_epsilon>config->epsilon_min) new_epsilon  = last_epsilon / config->epsilon_scale_fast;
    
  }
  MPI_Bcast(&new_epsilon,1,MPI_FLOAT,0,MPI_COMM_WORLD);
#endif
  return new_epsilon;
}

float regression_with_theta_t::infer_rho(){
  float new_rho = 0;
#ifdef USE_MPI
  if(mu==0) return config->rho_min;
  //float scaler = this->mu;
  if (mpi_rank==0){
    ofs_debug<<"INFER_RHO: Inner iterate: "<<iter_rho_epsilon<<endl;
    if(iter_rho_epsilon==0){
      //cerr<<"DEBUG: residual "<<residual<<" map dist "<<map_distance<<" epsilon "<<epsilon<<endl;
      if(track_residual){
        new_rho = rho_distance_ratio * residual * sqrt(map_distance+epsilon)/map_distance;
        if(new_rho>=config->rho_max) new_rho = config->rho_max;
        if(new_rho<=config->rho_min) new_rho = config->rho_min;
        
      }else{
        new_rho=last_rho * config->rho_scale_fast;
        if(new_rho>=config->rho_max) new_rho = config->rho_max;
        if (in_feasible_region()) { // the projections are feasible
        //if (new_rho>=config->rho_max) {
          track_residual = true;
          cerr<<"INFER_RHO: enabling residual tracking\n";
        }
      }
      if (config->verbose) cerr<<"NEW RHO "<<new_rho<<" LAST RHO "<<last_rho<<endl;
    }else{
      new_rho = last_rho;
    }
     //new_rho *= new_rho>config->rho_max?config->rho_scale_slow:config->rho_scale_fast;
    //}
    //new_rho = scaler * this->slaves * sqrt(this->map_distance+epsilon);
    //if (config->verbose) cerr <<"INFER_RHO: mu: "<<this->mu<<" new rho: "<<new_rho<<endl;
  }
  MPI_Bcast(&new_rho,1,MPI_FLOAT,0,MPI_COMM_WORLD);
#endif
  return new_rho;
}

void regression_with_theta_t::initialize(){
  if (mpi_rank==0){
    ofs_debug<<"Mu iterate: "<<iter_mu<<" mu="<<mu<<" of "<<config->mu_max<<endl;
  }
  if (this->mu == 0){
    this->top_k = config->top_k;
    for(int j=0;j<this->variables;++j){
      last_beta[j] = beta[j] = 0;
      constrained_beta[j] = 0;
      beta_project[j] = 0;
    }
    for(int i=0;i<this->observations;++i){
      Xbeta[i] = 0;
      theta_project[i] = 0;
      theta[i] = y[i]/slaves;
      //theta[i] = slave_id==0?.999*y[i]:.001*y[i];
      //theta[i] = y[i]/2;
    }
    this->map_distance = 0;
  }
}

bool regression_with_theta_t::finalize_inner_iteration(){
  //int proceed = false;
#ifdef USE_MPI
  if(mpi_rank==0){
    float active_norm = 0;
    float inactive_norm = 0;
    active_set_size = 0;
    for(int j=0;j<variables;++j){
      active_set[j]=fabs(beta[j])>config->beta_epsilon;
      active_set_size+=active_set[j];
      if(constrained_beta[j]==0){
        inactive_norm+=fabs(beta[j]);
      }else{
        active_norm+=fabs(beta[j]);
      }
    }
    if (config->verbose) cerr<<"active set size "<<active_set_size<<", inactive norm: "<<inactive_norm<<" active norm: "<<active_norm<<endl;
    //proceed = (mu==0 || active_set_size>config->top_k);
  }
  print_output();
  //MPI_Bcast(&proceed,1,MPI_INT,0,MPI_COMM_WORLD);
#endif
  return true;
  //return proceed;
}

bool regression_with_theta_t::finalize_iteration(){
  int proceed = false; 
  //int active_count = 0;
  if(mpi_rank==0){
    for(int j=0;j<variables;++j){
      //if(fabs(beta[j])<config->beta_epsilon) beta[j] = 0;
      //beta[j] = constrained_beta[j];
      //active_count+=beta[j]!=0;
    }
    //if(config->verbose) cerr<<"FINALIZE_ITERATION: active_count reset to: "<<active_count<<endl;
  }
  //MPI_Scatterv(beta,node_sizes,node_offsets,MPI_FLOAT,beta,variables,MPI_FLOAT,0,MPI_COMM_WORLD);
#ifdef USE_MPI
  if(mpi_rank==0){
    //float active_norm = 0;
    //float inactive_norm = 0;
    //active_set_size = 0;
    //for(int j=0;j<variables;++j){
     
    //  active_set[j]=fabs(beta[j])>config->beta_epsilon;
    //  ofs_debug<<"FINALIZE_ITERATION: "<<active_set[j]<<","<<beta[j]<<","<<config->beta_epsilon<<endl;
    //  active_set_size+=active_set[j];
    //  if(constrained_beta[j]==0){
    //    inactive_norm+=fabs(beta[j]);
    //  }else{
    //    active_norm+=fabs(beta[j]);
    //  }
   // }
   // if (config->verbose)cerr<<"inactive norm: "<<inactive_norm<<" active norm: "<<active_norm<<endl;
   // ofs_debug<<"FINALIZE_ITERATION: active set size at mu "<<mu<<": "<<active_set_size<<endl;
    float diff_norm = 0;
    for(int j=0;j<variables;++j){
      if(constrained_beta[j]!=0){
        float dev = beta[j]-last_beta[j];
        diff_norm+=dev*dev;
      }
      last_beta[j] = beta[j];
    }
    diff_norm = sqrt(diff_norm);
    if (config->verbose) cerr<<"FINALIZE_ITERATION: Beta norm difference is "<<diff_norm<<endl;
    if (diff_norm<1e-6 && !in_feasible_region()){
      if (config->verbose) cerr<<"FINALIZE_ITERATION: Doubling rho_max as we converged to an infeasible region\n";
      config->rho_max*=2;
    }
    proceed = get_map_distance() > 1e-6  || diff_norm>1e-6;
    //proceed = get_map_distance() > 1e-6  || active_set_size>this->top_k;
  }
  MPI_Bcast(&proceed,1,MPI_INT,0,MPI_COMM_WORLD);
  if(!proceed){
    int n = observations;
    int p = variables;
    ostringstream oss;
    oss<<"param.final."<<mpi_rank;
    string filename=oss.str();
    cerr<<"FINALIZE_ITERATION: Dumping parameter contents in "<<filename<<"\n";
    ofstream ofs(filename.data());
    ofs<<"i\tTheta\tTheta project\n";
    for(int i=0;i<n;++i){
      ofs<<i<<"\t"<<theta[i]<<"\t"<<theta_project[i]<<"\n";
    }
    ofs<<"j\tBeta\tBeta project\n";
    for(int j=0;j<p;++j){
      ofs<<j<<"\t"<<beta[j]<<"\t"<<beta_project[j]<<"\n";
    }
    ofs.close();
  }
//cout<<"active size size "<<active_set_size<<" mu: "<<mu<<" proceed: "<<proceed<<endl;
#endif
  return proceed;
}
  

void regression_with_theta_t::iterate(){
  update_lambda();
  project_beta();
  project_theta();
  if(slave_id>=0){
    for(int i=0;i<observations;++i){
      //ofs_debug<<"slave "<<slave_id<<" theta["<<i<<"] projection: "<<theta_project[i]<<endl;
    }
    for(int j=0;j<variables;++j){
      //ofs_debug<<"slave "<<slave_id<<" beta["<<j<<"] projection: "<<beta_project[j]<<endl;
    }
  }
  update_map_distance();
  update_theta();
  update_beta();
  update_map_distance();
  //float d2 = get_map_distance();
  //MPI::Finalize();
  //exit(0);
  //for(int iter=0;iter<10;++iter){
  //mmultiply(X,observations,variables,beta,1,Xbeta);
    ////for(int i=0;i<5;++i){ ofs_debug<<"Xbeta["<<i<<"] is "<<Xbeta[i]<<endl;}
    //loss();
    //float obj = evaluate_obj();
    //ofs_debug<<"Current objective at iter "<<iter<<" is "<<obj<<endl;
    //check_constraints();
  //}
  //cout<<"BETA:\n";
  //for(int j=0;j<p;++j){
    //if (j<10) ofs_debug<<"BETA: "<<j<<":"<<beta[j]<<endl;
  //}
  //ofs_debug<<"Done!\n";
  ++total_iterations;
}

void regression_with_theta_t::print_output(){
  if(mpi_rank==0 ){
  //if(mpi_rank==0 && active_set_size<=config->top_k){
    cerr<<"Mu: "<<mu<<" rho: "<<rho<<" epsilon: "<<epsilon<<" total iterations: "<<this->total_iterations<<" active size: "<<active_set_size<<endl;
    cerr<<"ACTIVE?\tINDEX\tBETA\n";
    //for(int j=0;j<10;++j){
    for(int j=0;j<variables;++j){
      if (active_set[j] && constrained_beta[j]!=0){
        cerr<<"+";
        cerr<<"\t"<<j<<"\t"<<beta[j]<<endl;
      }
      if (active_set[j]){
        //cerr<<"+";
        //cerr<<"\t"<<j<<"\t"<<beta[j]<<endl;
      }else{
        //cerr<<"-";
        //cerr<<"\t"<<j<<"\t"<<beta[j]<<endl;
        //beta[j] = 0;
      }
    }
    ofs_debug<<"Done!\n";
  }
}


bool regression_with_theta_t::proceed_qn_commit(){
  int proceed = false;
#ifdef USE_MPI
  if(mpi_rank==0){
    proceed = true;
    //proceed =  (residual<last_residual) || ((residual - last_residual)/residual < .01);
    if (!proceed && config->verbose) cerr<<"QN commit failed because residual difference was "<<(residual - last_residual)/residual<<endl;
  }
  MPI_Bcast(&proceed,1,MPI_INT,0,MPI_COMM_WORLD);
#endif
  return proceed;
}

float regression_with_theta_t::evaluate_obj(){
  float obj=0;
#ifdef USE_MPI
  int bypass = 0;
  if (mpi_rank==0){
    last_residual = residual;
    residual = 0;
    for(int i=0;i<observations;++i){
      residual+=(y[i]-theta[i])*(y[i]-theta[i]);
    }
    //float penalty = 0;
    //for(int j=0;j<variables;++j){
      //penalty+=nu[j]*fabs(beta[j]);
    //}
    float proxdist = get_prox_dist_penalty(); 
    obj = .5*residual+proxdist;
    if(config->verbose){
      cerr<<"EVALUATE_OBJ: norm1 "<<residual<<" PROXDIST: "<<proxdist<<" FULL: "<<obj<<endl;
    }
    bool use_kevin_increment = false;
    if (use_kevin_increment){
      float obj_diff = last_obj-obj;
      if( obj_diff>0 && obj_diff <config->obj_epsilon*100){
        private_epsilon/=(private_epsilon>config->epsilon_min)?config->epsilon_scale_fast:config->epsilon_scale_slow;
        private_rho*=(private_rho<config->rho_max)?config->rho_scale_fast:config->rho_scale_slow;
        if(config->verbose) {
          cerr<<"EVALUATE_OBJ: Suggested increasing rho and decreasing eps\n";
          cerr<<"INFER_EPSILON: last,current: "<<last_obj<<","<<obj<<" epsilon decremented to "<<private_epsilon<<endl;
          cerr<<"INFER_RHO: Rho incremented to "<<private_rho<<endl;
          //cerr<<"INFER_RHO: Bypassing downill check\n";
        }
      }else{
        private_rho = rho;
        private_epsilon = epsilon;
      }
      //bypass = true;
      bypass = (rho!=last_rho || epsilon!=last_epsilon);
      if(config->verbose) {
        cerr<<"INFER_RHO: lastrho,current: "<<last_rho<<","<<rho<<" downhill bypass: "<<bypass<<"\n";
      }
    }
  }
  ofs_debug<<"EVALUATE_OBJ: Last obj "<<last_obj<<" current:  "<<obj<<"!\n";
  //MPI_Bcast(&bypass,1,MPI_INT,0,MPI_COMM_WORLD);
  //bypass_downhill_check = bypass;
  MPI_Bcast(&obj,1,MPI_FLOAT,0,MPI_COMM_WORLD);
#endif
  return obj;
}

void regression_with_theta_t::load_matrix_data(const char *  mat_file,float * & mat,int input_rows, int input_cols,int output_rows, int output_cols, bool * row_mask, bool * col_mask,bool file_req, float defaultVal){
#ifdef USE_MPI
  ofs_debug<<"Loading matrix data with input dim "<<input_rows<<" by "<<input_cols<<" and output dim "<<output_rows<<" by "<<output_cols<<endl;
  ofs_debug.flush();
  if(output_rows==0||output_cols==0) return;
  mat = new float[output_rows*output_cols];

  ifstream ifs(mat_file);
  if (!ifs.is_open()){
    ofs_debug<<"Cannot open file "<<mat_file<<endl;
    if (file_req){
      ofs_debug<<"File "<<mat_file<<" is required. Program will exit now.\n";
      MPI_Finalize();
      exit(1);
    }else{
      ofs_debug<<"File is optional.  Will default values to "<<defaultVal<<endl;
      for(int i=0;i<output_rows*output_cols;++i) mat[i] = defaultVal;
      return;
    }
  }
  string line;
  int output_row = 0;
  for(int i=0;i<input_rows;++i){
    getline(ifs,line);
    if (row_mask[i]){
      istringstream iss(line);
      int output_col = 0;
      for(int j=0;j<input_cols;++j){
        float val;
        iss>>val;
        if (col_mask[j]){
           if (output_row>=output_rows || output_col>=output_cols) {
             ofs_debug<<mpi_rank<<": Assigning element at "<<output_row<<" by "<<output_col<<endl;
             ofs_debug<<mpi_rank<<": Out of bounds\n";
             MPI_Finalize();
             exit(1);
           }
           mat[output_row*output_cols+output_col] = val;
           ++output_col;
        }
      }
      ++output_row;
    }
  }
  ofs_debug<<"MPI rank "<<mpi_rank<<" successfully read in "<<mat_file<<endl;
  ifs.close();
#endif
}

int regression_with_theta_t::get_qn_parameter_length(){
  int len = 0;
  if(mpi_rank==0){
    len = mpi_numtasks * this->observations;
    //len = this->variables + mpi_numtasks * this->observations;
  }
  return len;
}

void regression_with_theta_t::get_qn_current_param(float * params){
#ifdef USE_MPI
  float all_theta[mpi_numtasks * observations];
  MPI_Gather(theta,observations,MPI_FLOAT,all_theta,observations,MPI_FLOAT,0,MPI_COMM_WORLD);
  if(mpi_rank==0){
    for(int i=0;i<variables;++i){
      //params[i] = beta[i];
    }
    //int k = variables;
    int k = 0;
    for(int rank = 0;rank<mpi_numtasks;++rank){
      for(int i=0;i<observations;++i){
        params[k++] = all_theta[rank*observations+i];
      }
    }
  }
#endif
}

void regression_with_theta_t::store_qn_current_param(float * params){
#ifdef USE_MPI
  float all_theta[mpi_numtasks * observations];
  if(mpi_rank==0){
    //for(int i=0;i<variables;++i){
     // beta[i] = params[i];
    //}
    //int k = variables;
    int k = 0;
    for(int rank = 0;rank<mpi_numtasks;++rank){
      for(int i=0;i<observations;++i){
        all_theta[rank*observations+i] = params[k++] ;
      }
    }
    for(int i=0;i<observations;++i){
      //theta[i] = params[variables+i];
    }
  }
  //MPI_Scatterv(beta,node_sizes,node_offsets,MPI_FLOAT,beta,variables,MPI_FLOAT,0,MPI_COMM_WORLD);
  MPI_Scatter(all_theta,observations,MPI_FLOAT,theta,observations,MPI_FLOAT,0,MPI_COMM_WORLD);
  //cerr<<"got store2\n";
#endif
}


int main_regression_with_theta(int argc,char * argv[]){
//  if(argc<3){
//    ofs_debug<<"Usage: <genofile> <outcome file>\n";
//    return 1;
//  }
//  int arg=0;
//  const char * genofile = argv[++arg];
//  const char * phenofile = argv[++arg];
//
  return 0;
}

