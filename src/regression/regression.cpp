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
#include"regression.hpp"

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

regression_t::regression_t(bool single_run){
  this->single_run = single_run;
  this->total_iterations = 0;
  //cerr<<"Single run initialized\n";
}

regression_t::~regression_t(){
#ifdef USE_MPI
  if(this->single_run){
    MPI::Finalize();
  } 
  return;
  if(slave_id>=0){
    delete [] X;
    delete [] XT;
    delete [] all_y;
    delete [] y;
  }else{
    delete [] XX;
    delete [] XXI_inv;
  }
  delete [] snp_node_sizes;
  delete [] snp_node_offsets;
  delete [] subject_node_sizes;
  delete [] subject_node_offsets;
  delete [] active_set;
  delete [] beta;
  delete [] all_beta;
  delete [] beta_project;
  delete [] all_constrained_beta;
  delete [] constrained_beta;
  delete [] Xbeta;
  delete [] theta;
  delete [] theta_project;
  delete [] lambda;
  if(config->debug_mpi){
    ofs_debug.close();
  }
#endif
}


// should only be called by slave nodes



void regression_t::update_lambda(){
#ifdef USE_MPI
  bool stripe_method = false;
  float Xbeta_full[observations];
  if (stripe_method){
    float xbeta_theta[sub_observations];
    if(slave_id>=0){
      for(int i=0;i<sub_observations;++i){
        Xbeta[i] = 0;
        for(int j=0;j<all_variables;++j){
          //Xbeta[i]+=X_stripe[i*all_variables+j] * all_constrained_beta[j];
          Xbeta[i]+=X_stripe[i*all_variables+j] * all_beta[j];
        }
      }
      for(int i=0;i<sub_observations;++i){
        xbeta_theta[i] = Xbeta[i]-theta[i];
        //if (total_iterations==1) cerr<<"slave "<<slave_id<<" i "<<i<<" xbeta "<<Xbeta[i]<<" theta "<<theta[i]<<endl;
      }
    }
    // now assemble xbeta_theta
    //if(mpi_rank==0) cerr<<"UPDATE_LAMBDA "<<mpi_rank<<endl;
    MPI_Gatherv(xbeta_theta,sub_observations,MPI_FLOAT,xbeta_theta,subject_node_sizes,subject_node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
    if(mpi_rank==0){
      mmultiply(XXI_inv,sub_observations,sub_observations,xbeta_theta,1,lambda);
      for(int i=0;i<sub_observations;++i){
      }
    }
  }else{
    float xbeta_theta[observations];
    if(slave_id>=0){
      for(int i=0;i<observations;++i){
        Xbeta_full[i] = 0;
        for(int j=0;j<variables;++j){
          Xbeta_full[i]+=X[i*variables+j] * beta[j];
        }
 //       cerr<<"Slave "<<slave_id<<" i "<<i<<" X_betafull: "<<Xbeta_full[i]<<endl;
      }
    }else{
      for(int i=0;i<observations;++i) Xbeta_full[i] = 0;
    }
    float xbeta_reduce[observations];
    MPI_Reduce(Xbeta_full,xbeta_reduce,observations,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
    if(mpi_rank==0){
      for(int i=0;i<observations;++i){
        xbeta_theta[i] = xbeta_reduce[i]-theta[i];
//cerr<<"i, xbetareduce, theta:"<<i<<","<<xbeta_reduce[i]<<","<<theta[i]<<endl;
      }
      mmultiply(XXI_inv,observations,observations,xbeta_theta,1,lambda);
    }
  }
  MPI_Bcast(lambda,observations,MPI_FLOAT,0,MPI_COMM_WORLD);
#endif
}

void regression_t::project_theta(){
  if(mpi_rank==0){
    for(int i=0;i<sub_observations;++i){
      theta_project[i] = theta[i]+lambda[i];
      //if(i<10)cerr<<"Thetaproject "<<i<<" "<<theta_project[i]<<endl;
 
    }
  }
  //MPI_Scatterv(theta_project,subject_node_sizes,subject_node_offsets,MPI_FLOAT,theta_project,sub_observations,MPI_FLOAT,0,MPI_COMM_WORLD);
}

void regression_t::project_beta(){
  if(slave_id>=0){
    for(int j=0;j<variables;++j){
      if(!in_feasible_region() || constrained_beta[j]!=0){
        float xt_lambda = 0;
        for(int i=0;i<observations;++i){
          xt_lambda+=XT[j*observations+i] * lambda[i];
        }
        beta_project[j] = beta[j]-xt_lambda;
      }else{
        beta_project[j] = 0;
      }
//        cerr<<"PROJECT_BETA: var "<<j<<" is "<<beta_project[j]<<endl;
    }
  }
}

void regression_t::update_map_distance(){
#ifdef USE_MPI
  theta_distance = 0;
  if(mpi_rank==0){
    for(int i=0;i<observations;++i){
      float dev = theta[i]-theta_project[i];
       theta_distance+=dev*dev;
    }
  }
  beta_distance = 0;
  if(slave_id>=0){
    for(int j=0;j<variables;++j){
      float dev = (beta[j]-beta_project[j]);
      beta_distance+=(dev*dev);
    }
  }
  float beta_distance_reduce;
  MPI_Reduce(&beta_distance,&beta_distance_reduce,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Gatherv(beta,variables,MPI_FLOAT,beta,snp_node_sizes,snp_node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
  if(mpi_rank==0){
    float constraint_dev=0;
    for(int j=0;j<all_variables;++j){
      float dev = constrained_beta[j]==0?beta[j]:0;
      constraint_dev+=dev*dev;
    }
    this->map_distance = theta_distance + beta_distance_reduce + constraint_dev;
    if(config->verbose) cerr<<"UPDATE_MAP_DISTANCE: Deviances: theta: "<<theta_distance<<" beta:"<<beta_distance_reduce<<" constraint:"<<constraint_dev<<endl;
    this->dist_func = rho/sqrt(this->map_distance+epsilon);
    //cerr<<"UPDATE_MAP_DISTANCE: Deviances: theta: "<<theta_distance<<" beta:"<<beta_distance_reduce<<" constraint:"<<constraint_dev<<" map dist: "<<this->map_distance<<endl;
  }
  MPI_Bcast(&this->map_distance,1,MPI_FLOAT,0,MPI_COMM_WORLD);
#endif
}

float regression_t::get_map_distance(){
  return this->map_distance;
}

void regression_t::update_theta(){
#ifdef USE_MPI
  if(mpi_rank==0){
    float coeff = 1./(1+dist_func);
    for(int i=0;i<sub_observations;++i){
      theta[i] = theta_project[i]+coeff*(y[i]-theta_project[i]);
      //cerr<<"Subject "<<i<<" theta "<<theta[i]<<" projection "<<theta_project[i]<<endl;
    }
  }
  MPI_Scatterv(theta,subject_node_sizes,subject_node_offsets,MPI_FLOAT,theta,sub_observations,MPI_FLOAT,0,MPI_COMM_WORLD);
#endif
}

void regression_t::update_beta(){
#ifdef USE_MPI
  //double start = clock();
  MPI_Scatterv(constrained_beta,snp_node_sizes,snp_node_offsets,MPI_FLOAT,constrained_beta,variables,MPI_FLOAT,0,MPI_COMM_WORLD);
  if(slave_id>=0){
    for(int j=0;j<variables;++j){
      if(!in_feasible_region() || constrained_beta[j]!=0){
        beta[j] = .5*(beta_project[j]+constrained_beta[j]);
      }else{
        beta[j] = 0;
      }
    }
  }
  MPI_Gatherv(beta,variables,MPI_FLOAT,beta,snp_node_sizes,snp_node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
  update_constrained_beta();
#endif
}

void regression_t::update_constrained_beta(){
  // prepare to broadcast full beta vector to slaves
  //int active_indices[top_k];
  //float active_vals[top_k];
  int active_counter = 0;
  if(mpi_rank==0){
    multiset<beta_t,byValDesc> sorted_beta;
    for(int j=0;j<variables;++j){
      beta_t b(j,beta[j]);
      sorted_beta.insert(b);
    }
    int j=0;
    for(multiset<beta_t,byValDesc>::iterator it=sorted_beta.begin();it!=sorted_beta.end();it++){
      beta_t b = *it;
      if (j<top_k){
        constrained_beta[b.index] = b.val;
        //active_indices[active_counter] = b.index;
        //active_vals[active_counter] = b.val;
        ++active_counter; 
      }else{
        constrained_beta[b.index] = 0;
      }
      ++j;
    }

    for(int j=0;j<variables;++j){
      all_beta[j] = beta[j];
      all_constrained_beta[j] = constrained_beta[j];
      
      //if(j>=0) cerr<<"All beta"<<j<<" : "<<all_beta[j]<<endl;
    }
  }
//  MPI_Bcast(all_beta,all_variables,MPI_FLOAT,0,MPI_COMM_WORLD);
//  MPI_Bcast(active_indices,top_k,MPI_INT,0,MPI_COMM_WORLD);
//  MPI_Bcast(active_vals,top_k,MPI_FLOAT,0,MPI_COMM_WORLD);
//  if(slave_id>=0){
//    for(int j=0;j<variables;++j){
//      all_constrained_beta[j] = 0;
//    }
//    for(int j=0;j<top_k;++j){
//      all_constrained_beta[active_indices[j]] = active_vals[j];
//    }
//  }
}

void regression_t::loss(){
}

void regression_t::check_constraints(){
}

bool regression_t::in_feasible_region(){
  float mapdist = get_map_distance();
  bool ret= (mapdist>0 && mapdist<1e-3);
  return ret;
}

void regression_t::parse_config_line(string & token,istringstream & iss){
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

void testsvd(int mpi_rank){
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

void regression_t::standardize(float * X,int observations,int variables){
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

void regression_t::read_dataset(){
#ifdef USE_MPI
  // figure out the dimensions
  //
  const char * genofile = config->genofile.data();
  const char * phenofile = config->traitfile.data();
  this->observations = linecount(phenofile);
  this->all_variables = colcount(genofile);
  if(config->verbose) cerr<<"Subjects: "<<observations<<" and predictors: "<<all_variables<<endl;
  // taken from Eric Chi's project
  
  bool single_mask[] = {true};
  bool full_variable_mask[all_variables];
  bool variable_mask[all_variables];
  bool full_subject_mask[observations];
  bool subject_mask[observations];

  for(int i=0;i<observations;++i) full_subject_mask[i] = true;
  for(int i=0;i<observations;++i) subject_mask[i] = slave_id<0?true:false;
  for(int i=0;i<all_variables;++i) full_variable_mask[i] = true;
  for(int i=0;i<all_variables;++i) variable_mask[i] = slave_id<0?true:false;

  vector<vector<int> > snp_indices_vec;
  vector<vector<int> > observation_indices_vec;

  // assign variable indices to each slave
  // 
  // this section divides the variables into equal sized blocks that
  // are each handled by a MPI slave
  // in the future, this can be 
  int variables_per_slave = all_variables/slaves;
  int observations_per_slave = observations/slaves;
  int i = 0;
  int j = 0;
  for(uint slave=0;slave<slaves;++slave){
    vector<int> snp_indices;
    vector<int> observation_indices;
    uint chunked_variables = (slave<slaves-1)?variables_per_slave:
    (all_variables-(slaves-1)*variables_per_slave);
    uint chunked_observations = (slave<slaves-1)?observations_per_slave:
    (observations-(slaves-1)*observations_per_slave);
    if(config->verbose) cerr<<"Chunked variables for slave "<<slave
    <<" is "<<chunked_variables<<" and observations is "<<chunked_observations<<endl;  
    for(uint k=0;k<chunked_variables;++k){
      snp_indices.push_back(j);
      ++j;
    }
    for(uint k=0;k<chunked_observations;++k){
      observation_indices.push_back(i);
      ++i;
    }
    snp_indices_vec.push_back(snp_indices);
    observation_indices_vec.push_back(observation_indices);
  }

  // now populate the variable mask for input with the assigned indices
  if (slave_id>=0){
    for(uint i=0;i<snp_indices_vec[slave_id].size();++i){
      variable_mask[snp_indices_vec[slave_id][i]] = true;
    }
    
    for(uint i=0;i<observation_indices_vec[slave_id].size();++i){
      subject_mask[observation_indices_vec[slave_id][i]] = true;
      //if(slave_id==1) cerr<<" "<<observation_indices_vec[slave_id][i];
    }
    //if (slave_id==1) cerr<<endl;
  }
  snp_node_sizes[0] = 0;
  snp_node_offsets[0] = 0;
  subject_node_sizes[0] = 0;
  subject_node_offsets[0] = 0;
  int snp_offset = 0;
  int subject_offset = 0;
  for(uint i=0;i<slaves;++i){
    snp_node_sizes[i+1] =  snp_indices_vec[i].size();
    snp_node_offsets[i+1] = snp_offset;
    snp_offset+=snp_node_sizes[i+1];
    subject_node_sizes[i+1] =  observation_indices_vec[i].size();
    subject_node_offsets[i+1] = subject_offset;
    subject_offset+=subject_node_sizes[i+1];
    if(config->verbose) cerr<<"Offsets: SNP: "<<snp_offset<<" subject: "<<subject_offset<<endl;
  }

  MPI_Type_contiguous(observations,MPI_FLOAT,&floatSubjectsArrayType);
  MPI_Type_commit(&floatSubjectsArrayType);
  this->variables=slave_id>=0?snp_indices_vec[slave_id].size():all_variables;
  this->sub_observations=slave_id>=0?observation_indices_vec[slave_id].size():observations;
  if(config->verbose) cerr<<"Node "<<mpi_rank<<" with "<<variables<<" variables.\n";
  // master and slaves share same number of observations
  load_matrix_data(config->traitfile.data(),y,observations,1,sub_observations,1,subject_mask, single_mask,true,0);
  if(slave_id>=0){
    load_matrix_data(config->traitfile.data(),all_y,observations,1,observations,1,full_subject_mask, single_mask,true,0);
  }
  if(slave_id>=0){
    load_matrix_data(genofile,X,observations,all_variables,observations,variables, full_subject_mask,variable_mask,true,0);
    load_matrix_data(genofile,X_stripe,observations,all_variables,observations,all_variables, subject_mask,full_variable_mask,true,0);
    // At this point we should standardize all the variables
    //if(config->verbose) cerr<<"Standardizing variables\n";
    //if(config->verbose) cerr.flush();
    //standardize(X,observations,variables);
    //standardize(X_stripe,sub_observations,all_variables);
    // SANITY CHECK THAT WE READ IN THE DATASET CORRECTLY
    //
    bool debugoutput=false;
    if(debugoutput){
      ostringstream oss;
      oss<<"X."<<mpi_rank<<".txt";
      ofstream ofs(oss.str().data());
      for(int i=0;i<observations;++i){
        for(int j=0;j<this->variables;++j){
          if(j) ofs<<"\t";
          ofs<<X[i*this->variables+j];
        }
        ofs<<endl;
      }
      ofs.close();
      ostringstream oss2;
      oss2<<"X_STRIPE."<<mpi_rank<<".txt";
      ofstream ofs2(oss2.str().data());
      for(int i=0;i<sub_observations;++i){
        for(int j=0;j<all_variables;++j){
          if(j) ofs2<<"\t";
          ofs2<<X_stripe[i*all_variables+j];
        }
        ofs2<<endl;
      }
      ofs2.close();
    }
  }
  //MPI_Finalize();
  //exit(0);
#endif
}

float regression_t::compute_marginal_beta(float * xvec){
  // dot product first;
  float xxi = 0;
  float xy = 0;
  for(int i=0;i<observations;++i){
    xxi +=xvec[i]*xvec[i];
    xy +=xvec[i]*all_y[i];
  }
  if(fabs(xxi)<1e-5) return 0;
  xxi = 1./xxi;
  return xxi*xy;
}

void regression_t::init_marginal_screen(){
  if(slave_id>=0){
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
    //update_Xbeta(beta);
    //float xbeta_norm = 0;
    //for(int i=0;i<this->observations;++i){
      //theta_project[i] = theta[i] = y[i];
      //xbeta_norm+=Xbeta[i]*Xbeta[i];
      //Xbeta[i] = 0;
      //theta_project[i] = theta[i] = Xbeta[i];
    //}
    //xbeta_norm=sqrt(xbeta_norm);
    //cerr<<"Xbeta norm for node "<<mpi_rank<<" is "<<xbeta_norm<<endl;
  }
  for(int j=0;j<variables;++j){
    constrained_beta[j] = beta_project[j] = beta[j] = 0;
  }

}

void regression_t::compute_XX(){
#ifdef USE_MPI
  // just a stub for now
  if(mpi_rank==0){
    bool test = false;
    if(test){
      float * X2 = new float[observations*variables];
      ifstream ifs(config->genofile.data());
      cerr<<"Reading from "<<config->genofile<<" into "<<observations<<" by "<<variables<<endl;
      for(int i=0;i<observations;++i){
        string line;
        getline(ifs,line);
        istringstream iss(line);
        for(int j=0;j<variables;++j){
          iss>>X2[i*variables+j];
        } 
      }
      ifs.close();
      mmultiply(X2,observations,variables,XX);
      delete[]X2;
      for(int i=0;i<observations;++i){
        for(int j=0;j<observations;++j){
        }
        //cerr<<endl;
      }
    }
  }
  float anchor[all_variables];
  ifstream ifs_full;
  if(mpi_rank==0){
     ifs_full.open(config->genofile.data());
  }
  for(int i=0;i<observations;++i){
    if(mpi_rank==0){
      string line;
      getline(ifs_full,line);
      istringstream iss(line);
      for(int j=0;j<all_variables;++j){
        iss>>anchor[j];
      } 
    }
    //cerr<<"Broadcasting at 1st index "<<i<<endl;
    MPI_Bcast(anchor,all_variables,MPI_FLOAT,0,MPI_COMM_WORLD);
    float dot_prod[sub_observations];
    if(slave_id>=0){
      //cerr<<"Slave "<<slave_id<<" running\n";
      for(int i2=0;i2<sub_observations;++i2){
        dot_prod[i2] = 0;
        for(int j=0;j<all_variables;++j){
          dot_prod[i2]+=anchor[j]*X_stripe[i2*all_variables+j];
        }
      }
      //cerr<<"Slave "<<slave_id<<" done running\n";
    }
    //cerr<<"Gathering\n";
    MPI_Gatherv(dot_prod,sub_observations,MPI_FLOAT,dot_prod,subject_node_sizes,subject_node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
    if(mpi_rank==0){
      for(int i2=0;i2<sub_observations;++i2){
        XX[i*sub_observations+i2] = dot_prod[i2]; 
      }
    }
  }
  if(mpi_rank==0){
     ifs_full.close();
  }
#endif
}

void regression_t::init_xxi_inv(){
#ifdef USE_MPI
  int cached = false;
  ostringstream oss_xxi_inv;
  oss_xxi_inv<<config->xxi_inv_file_prefix<<"."<<mpi_rank<<".txt";
  if(mpi_rank==0){
    ifstream ifs_xxi_inv(oss_xxi_inv.str().data());
    if (ifs_xxi_inv.is_open()){
      cached = true;
      ifs_xxi_inv.close();
    }
  }    
  MPI_Bcast(&cached,1,MPI_INT,0,MPI_COMM_WORLD);
  if(cached){
    if(mpi_rank==0){
      ostringstream oss_xxi_inv;
      oss_xxi_inv<<config->xxi_inv_file_prefix<<"."<<mpi_rank<<".txt";
      ifstream ifs_xxi_inv(oss_xxi_inv.str().data());
      if(config->verbose) cerr<<"Using cached copy of singular values and vectors\n";
      string line;
      for(int i=0;i<observations;++i){
        getline(ifs_xxi_inv,line);
        istringstream issvec(line);
        for(int j=0;j<observations;++j){
          issvec>>XXI_inv[i*observations+j];
        }
      }
      ifs_xxi_inv.close();
    }
  }else{
    XX = new float[observations * observations];
    compute_XX();
    if(config->verbose) cerr<<"Cannot find cached copy of singular values and vectors. Computing\n";
    if (mpi_rank==0){
      if(config->verbose) cerr<<"Allocating GSL xx of size "<<observations<<endl;
      gsl_matrix * tempxx = gsl_matrix_alloc(observations,observations);
      if(config->verbose) cerr<<"Copying into gsl matrix\n";
      for(int i=0;i<observations;++i){
        for(int j=0;j<observations;++j){
          gsl_matrix_set(tempxx,i,j,XX[i*observations+j]);
        }
      }
      if(config->verbose) cerr<<"Performing eigen decomp\n";
      if(config->verbose) cerr.flush();
      //This function allocates a workspace for computing eigenvalues and eigenvectors of n-by-n real symmetric matrices. The size of the workspace is O(4n).
      gsl_eigen_symmv_workspace * eigen_work =  gsl_eigen_symmv_alloc (observations);
      gsl_matrix * tempv = gsl_matrix_alloc(observations,observations);
      gsl_vector * temps = gsl_vector_alloc(observations);
      //This function computes the eigenvalues and eigenvectors of the real symmetric matrix A. Additional workspace of the appropriate size must be provided in w. The diagonal and lower triangular part of A are destroyed during the computation, but the strict upper triangular part is not referenced. The eigenvalues are stored in the vector eval and are unordered. The corresponding eigenvectors are stored in the columns of the matrix evec. For example, the eigenvector in the first column corresponds to the first eigenvalue. The eigenvectors are guaranteed to be mutually orthogonal and normalised to unit magnitude.
        
      int code = gsl_eigen_symmv(tempxx, temps, tempv, eigen_work);
      if (code!=0) if(config->verbose) cerr<<"Returning nonzero code in eigendecomp\n";
      gsl_matrix * tempvdi = gsl_matrix_alloc(observations,observations);
      for(int i=0;i<observations;++i){
        for(int j=0;j<observations;++j){
          //if(config->verbose) cerr<<"Eigen "<<j<<","<<i<<":"<<gsl_matrix_get(tempv,j,i)<<endl;
          //if(config->verbose) cerr.flush();
          float vdi = gsl_matrix_get(tempv,i,j)/(1.+gsl_vector_get(temps,j));
          gsl_matrix_set(tempvdi,i,j,vdi);
        }
      }
      if(mpi_rank==-1){
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

      //if(config->verbose) cerr<<"Performing SVD\n";
      //if(config->verbose) cerr.flush();
      //cerr<<"Variables: "<<variables<<" observations: "<<observations<<endl;
      //gsl_matrix * tempv_svd = gsl_matrix_alloc(variables,variables);
      //gsl_vector * temps_svd = gsl_vector_alloc(variables);
      //gsl_vector * work_svd  = gsl_vector_alloc(variables);
      //gsl_linalg_SV_decomp(tempx,tempv_svd,temps_svd,work_svd);
      //gsl_vector_free(work_svd);
      for(int j=0;j<this->variables;++j){
        for(int i=0;i<observations;++i){
          //if(config->verbose) cerr<<"SVD "<<j<<","<<i<<":"<<gsl_matrix_get(tempx,i,j)<<endl;
          //if(config->verbose) cerr.flush();
//          //float vdi = gsl_matrix_get(tempx,i,j)/(1.+gsl_vector_get(temps,j));
//          //gsl_matrix_set(tempvdi,i,j,vdi);
        }
      }
      //gsl_matrix_free(tempv_svd);
      //gsl_vector_free(temps_svd);
      if(config->verbose) cerr<<"Computing VDIV\n";
      if(config->verbose) cerr.flush();
      gsl_matrix * tempvdiv = gsl_matrix_alloc(observations,observations);
      gsl_blas_dgemm(CblasNoTrans,CblasTrans,1,tempvdi,tempv,0,tempvdiv);
      gsl_matrix_free(tempvdi);
      gsl_matrix_free(tempv);
      if(config->verbose) cerr<<"Writing to file "<<oss_xxi_inv.str()<<"\n";
      if(config->verbose) cerr.flush();
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
  //MPI_Bcast(XXI_inv,observations*observations,MPI_FLOAT,0,MPI_COMM_WORLD);
#endif
}

void regression_t::init(string config_file){
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
  if(config->verbose) cerr<<"Configuration initialized\n";
}

void regression_t::allocate_memory(){
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
  this->snp_node_sizes = new int[mpi_numtasks];
  this->snp_node_offsets = new int[mpi_numtasks];
  this->subject_node_sizes = new int[mpi_numtasks];
  this->subject_node_offsets = new int[mpi_numtasks];
  if(config->verbose) cerr<<"Initialized MPI with "<<slaves<<" slaves.  I am slave "<<slave_id<<endl;
  read_dataset();

  if(slave_id>=0){
    // Also, set up X transpose too, a constant matrix variable
    this->XT = new float[this->variables*observations];
    for(int j=0;j<this->variables;++j){
      for(int i=0;i<observations;++i){
        XT[j*observations+i] = X[i*this->variables+j];
      }
    }
  }else{
    XXI_inv = new float[observations*observations];
  }
  // this procedure will do the SVD at the master and then broadcast to slaves
  init_xxi_inv();
  // ALLOCATE MEMORY FOR ESTIMATED PARAMETERS
  this->last_residual = 1e10;
  this->residual = 0;
  this->active_set_size = 0;
  this->active_set = new bool[this->variables];
  this->beta = new float[this->variables];
  this->all_beta = new float[this->all_variables];
  this->last_beta = new float[this->variables];
  this->beta_project = new float[this->variables];
  this->all_constrained_beta = new float[this->all_variables];
  this->constrained_beta = new float[this->variables];
  this->lambda = new float[observations];
  this->Xbeta = new float[sub_observations];
  this->theta = new float[sub_observations];
  this->theta_project = new float[sub_observations];
  init_marginal_screen();
  // the LaGrange multiplier
  proxmap_t::allocate_memory();
#endif
}

float regression_t::infer_epsilon(){
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

float regression_t::infer_rho(){
  float new_rho = 0;
#ifdef USE_MPI
  if(mu==0) return config->rho_min;
  //float scaler = this->mu;
  if (mpi_rank==0){
    ofs_debug<<"INFER_RHO: Inner iterate: "<<iter_rho_epsilon<<endl;
    float saved_rho = new_rho; 
    if(iter_rho_epsilon==0){
      //cerr<<"DEBUG: residual "<<residual<<" map dist "<<map_distance<<" epsilon "<<epsilon<<endl;
      if(track_residual){
        new_rho = rho_distance_ratio * residual * sqrt(map_distance+epsilon)/map_distance;
        //if(new_rho>=config->rho_max) new_rho = config->rho_max;
        //if(new_rho<=config->rho_min) new_rho = config->rho_min;
        
      }else{
        if(last_rho>=config->rho_max) {
          config->rho_max*=config->rho_scale_fast;
        }
        new_rho = last_rho<config->rho_max?last_rho * config->rho_scale_slow:last_rho;
        if (in_feasible_region()) { // the projections are feasible
          track_residual = true;
          cerr<<"INFER_RHO: enabling residual tracking\n";
        }
      }
      if(isinf(new_rho) || isnan(new_rho)){
        new_rho = saved_rho;
        cerr<<"INFER_RHO: Overflow with new rho, backtracking to "<<new_rho<<endl;
      }
      if (config->verbose) cerr<<"NEW RHO "<<new_rho<<" LAST RHO "<<last_rho<<" RHO_MAX "<<config->rho_max<<endl;
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

void regression_t::initialize(){
  if (mpi_rank==0){
    ofs_debug<<"Mu iterate: "<<iter_mu<<" mu="<<mu<<" of "<<config->mu_max<<endl;
  }
  if (this->mu == 0){
    this->top_k = config->top_k;
    for(int j=0;j<this->all_variables;++j){
      all_beta[j] = 0;
      all_constrained_beta[j] = 0;
    }
    for(int j=0;j<this->variables;++j){
      last_beta[j] = beta[j] = 0;
      constrained_beta[j] = 0;
      beta_project[j] = 0;
    }
    for(int i=0;i<this->sub_observations;++i){
      float init_val = y[i];
      theta[i] = theta_project[i] = init_val;
    }
    this->map_distance = 0;
  }
}

bool regression_t::finalize_inner_iteration(){
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
  //print_output();
  //MPI_Bcast(&proceed,1,MPI_INT,0,MPI_COMM_WORLD);
#endif
  return true;
  //return proceed;
}

bool regression_t::finalize_iteration(){
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
  //MPI_Scatterv(beta,snp_node_sizes,snp_node_offsets,MPI_FLOAT,beta,variables,MPI_FLOAT,0,MPI_COMM_WORLD);
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
    bool abort = false;
    if(rho<config->rho_min){ 
      cerr<<"FINALIZE_ITERATION: Rho shrunk to minimum. Aborting\n";
      abort = true;
    }
//    if (diff_norm<1e-6 && !in_feasible_region()){
//      config->rho_max*=config->rho_scale_fast;
//      if (config->verbose) cerr<<"FINALIZE_ITERATION: Increasing rho_max to "<<config->rho_max<<" as we converged to an infeasible region\n";
//      if (isinf(config->rho_max)|| isnan(config->rho_max)){
//        cerr<<"FINALIZE_ITERATION: Rho max would be infinity. Aborting\n";
//        abort = true;
//      }
//    }
    //proceed = total_iterations<30000;
    proceed = (get_map_distance() > 1e-6  || diff_norm>1e-5) && (!abort);
    //proceed = get_map_distance() > 1e-6  || active_set_size>this->top_k;
  }
  MPI_Bcast(&proceed,1,MPI_INT,0,MPI_COMM_WORLD);
  if(!proceed){
    int p = variables;
    ostringstream oss;
    oss<<"param.final."<<mpi_rank;
    string filename=oss.str();
    cerr<<"FINALIZE_ITERATION: Dumping parameter contents in "<<filename<<"\n";
    ofstream ofs(filename.data());
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
  

void regression_t::iterate(){
//  if(mpi_rank==0){
//    int rows = 3;int cols = 3;
//    float mat[] = {1,2,1,1,2,3,4,2,66};
//    float outmat[rows*cols];
//    invert(mat,outmat,rows,cols);
//    for(int i=0;i<rows;++i){
//      for(int j=0;j<cols;++j){
//        if(j) cerr<<"\t";
//        cerr<<outmat[i*cols+j];
//      }
//      cerr<<endl;
//    }
//  }
//  MPI_Finalize();
//  exit(0);
  //cerr<<"ITERATE: "<<mpi_rank<<endl;
  update_lambda();
  project_theta();
  project_beta();
  update_map_distance();
  update_theta();
  update_beta();
  update_map_distance();
  ++total_iterations;
}

void regression_t::print_output(){
  if(mpi_rank==0 ){
  //if(mpi_rank==0 && active_set_size<=config->top_k){
    cerr<<"Mu: "<<mu<<" rho: "<<rho<<" epsilon: "<<epsilon<<" total iterations: "<<this->total_iterations<<" active size: "<<active_set_size<<" mapdist: "<<this->map_distance<<endl;
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


bool regression_t::proceed_qn_commit(){
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

float regression_t::evaluate_obj(){
  float obj=0;
#ifdef USE_MPI
  //int bypass = 0;
//  cerr<<"EVALUATE_OBJ "<<mpi_rank<<" sub_obs: "<<sub_observations<<"\n";
  MPI_Gatherv(Xbeta,sub_observations,MPI_FLOAT,Xbeta,subject_node_sizes,subject_node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
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
  }
  ofs_debug<<"EVALUATE_OBJ: Last obj "<<last_obj<<" current:  "<<obj<<"!\n";
  //MPI_Bcast(&bypass,1,MPI_INT,0,MPI_COMM_WORLD);
  //bypass_downhill_check = bypass;
  MPI_Bcast(&obj,1,MPI_FLOAT,0,MPI_COMM_WORLD);
  //MPI_Finalize();
  //exit(0);
#endif
  return obj;
}

void regression_t::load_matrix_data(const char *  mat_file,float * & mat,int input_rows, int input_cols,int output_rows, int output_cols, bool * row_mask, bool * col_mask,bool file_req, float defaultVal){
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

int regression_t::get_qn_parameter_length(){
  int len = 0;
  if(mpi_rank==0){
    len = this->observations + this->variables;
  }
  return len;
}

void regression_t::get_qn_current_param(float * params){
#ifdef USE_MPI
  if(mpi_rank==0){
    int k=0;
    for(int i=0;i<observations;++i){
      params[k++] = theta[i];
    }
    for(int i=0;i<variables;++i){
      params[k++] = beta[i];
    }
  }
#endif
}

void regression_t::store_qn_current_param(float * params){
#ifdef USE_MPI
  if(mpi_rank==0){
    int k = 0;
    for(int i=0;i<observations;++i){
      theta[i] = params[k++];
    }
    for(int i=0;i<variables;++i){
       beta[i] = params[k++];
    }
  }
  MPI_Scatterv(theta,subject_node_sizes,subject_node_offsets,MPI_FLOAT,theta,sub_observations,MPI_FLOAT,0,MPI_COMM_WORLD);
  MPI_Scatterv(beta,snp_node_sizes,snp_node_offsets,MPI_FLOAT,beta,variables,MPI_FLOAT,0,MPI_COMM_WORLD);
  update_constrained_beta();
#endif
}


int main_regression(int argc,char * argv[]){
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

