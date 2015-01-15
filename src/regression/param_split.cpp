#include<assert.h>
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
#include<random_access.hpp>
#include<plink_data.hpp>
#ifdef USE_GPU
#include<ocl_wrapper.hpp>
#endif
#include"param_split.hpp"

const int SMALL_BLOCK_WIDTH = 32;

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

param_split_t::param_split_t(bool single_run){
  this->single_run = single_run;
  this->total_iterations = 0;
  //cerr<<"Single run initialized\n";
}

param_split_t::~param_split_t(){
#ifdef USE_MPI
  if(this->single_run){
    MPI::Finalize();
  } 

  if(slave_id>=0){
    delete random_access_XXI_inv;
    delete random_access_XXI;
    delete [] XXI;
    delete plink_data_X_subset;
    delete plink_data_X_subset_subjectmajor;
#ifdef USE_GPU
    if(this->run_gpu) delete ocl_wrapper;
#endif
  }

  delete [] y;
  delete [] Xbeta_full;
  delete[] means;
  delete[] precisions;
  delete [] snp_node_sizes;
  delete [] snp_node_offsets;
  delete [] subject_node_sizes;
  delete [] subject_node_offsets;
  delete [] beta;
  delete [] last_beta;
  delete [] all_beta;
  delete [] beta_project;
  delete [] constrained_beta;
  delete [] theta;
  delete [] theta_project;
  delete [] lambda;
  if(config->debug_mpi){
    ofs_debug.close();
  }
#endif
}


// should only be called by slave nodes



void param_split_t::update_lambda(){
#ifdef USE_MPI
  //bool run_cpu = false;
  //bool run_gpu = true;
  bool debug = false;
  //float Xbeta_full[observations];
  if(slave_id>=0){
    bool debug_gpu = (run_cpu && slave_id==-10);
    if(run_gpu){
#ifdef USE_GPU
      ocl_wrapper->write_to_buffer("beta",variables,beta);
      ocl_wrapper->run_kernel("update_lambda",BLOCK_WIDTH*snp_chunks,observations,1,BLOCK_WIDTH,1,1);
      ocl_wrapper->run_kernel("reduce_xbeta_chunks",BLOCK_WIDTH,observations,1,BLOCK_WIDTH,1,1);
      ocl_wrapper->read_from_buffer("Xbeta_full",observations,Xbeta_full);
      if(debug_gpu){
        //float Xbeta_chunks[observations*snp_chunks];
        float Xbeta_temp[observations];
        //ocl_wrapper->read_from_buffer("Xbeta_chunks",observations*snp_chunks,Xbeta_chunks);
        ocl_wrapper->read_from_buffer("Xbeta_full",observations,Xbeta_temp);
        cerr<<"debug gpu update_lambda GPU:";
        for(int i=0;i<observations;++i){
          //float xbeta = 0;
          for(int j=0;j<snp_chunks;++j){
            //if(i<10)cerr<<" "<<i<<","<<j<<":"<<Xbeta_chunks[i*snp_chunks+j];
            //xbeta+=Xbeta_chunks[i*snp_chunks+j];
          }
          if(i>(observations-10))cerr<<" "<<i<<":"<<Xbeta_temp[i];
          //if(i<10)cerr<<" "<<i<<":"<<xbeta;
        }
        cerr<<endl;
      }
#endif
    } // if run gpu

    if(run_cpu){
      if(debug) cerr<<"UPDATE_LAMBDA slave: "<<slave_id<<", observation:";
      //for(int i=0;i<100;++i){
      if (debug_gpu) cerr<<"debug gpu update_lambda CPU:";
      for(int i=0;i<observations;++i){
        float xb = 0;
        //cerr<<i<<":";
        // emulate what we would do on the GPU
        packedgeno_t packedbatch[SMALL_BLOCK_WIDTH];
        float subset_geno[BLOCK_WIDTH];
        for(int chunk=0;chunk<snp_chunks;++chunk){  // each chunk is 512 variables
          //if(slave_id==0)cerr<<"Working on chunk "<<chunk<<endl;
          for(int threadindex=0;threadindex<SMALL_BLOCK_WIDTH;++threadindex){
            // load 32 into this temporary array
            if(chunk*SMALL_BLOCK_WIDTH+threadindex<packedstride_subjectmajor){
              packedbatch[threadindex] = packedgeno_subjectmajor[i*
              packedstride_subjectmajor+chunk*SMALL_BLOCK_WIDTH+threadindex];
            }
          }
          for(int threadindex=0;threadindex<SMALL_BLOCK_WIDTH;++threadindex){
            // expand 32 elements into 512 genotypes
            int t = 0;
            for(int b=0;b<4;++b){
              for(int c=0;c<4;++c){
                subset_geno[threadindex*16+t] = 
                c2g(packedbatch[threadindex].geno[b],c);
                ++t;
              }
            }
          }
          for(int threadindex = 0;threadindex<BLOCK_WIDTH;++threadindex){
            int var_index = chunk*BLOCK_WIDTH+threadindex;
            if(var_index<variables){
              float g=subset_geno[threadindex]==9?0:(subset_geno[threadindex]-means[var_index])*precisions[var_index];
              //float g=(subset_geno[threadindex]);
              //float g2 = plink_data_X_subset_subjectmajor->get_raw_geno(i,var_index);
              //if(g!=g2) cerr<<"update-lambda Mismatch at SNP: "<<var_index<<" obs "<<i<<": "<<g<<","<<g2<<endl;
              xb+=g * beta[var_index];
              //if(slave_id==0)cerr<<" "<<var_index<<":"<<g;
              //xb2+=beta[var_index]*means[var_index]*precisions[var_index];
  
            }
          }
        }
        //if(slave_id==0)cerr<<endl;
        // gold standard way
        for(int j=0;j<variables;++j){
          //float g = plink_data_X_subset_subjectmajor->get_geno(i,j);
          //xb+=g * beta[j];
          //float r = plink_data_X_subset_subjectmajor->get_raw_geno(i,j);
          //if(slave_id==0)cerr<<" "<<j<<":"<<g;
          //if (isnan(g)) cerr<<","<<g<<" "<<beta[j];
        }
        //if(slave_id==0)cerr<<endl;
        Xbeta_full[i] = xb;
        if(debug_gpu && i>(observations-10)) cerr<<" "<<i<<":"<<xb;
         //cerr<<" "<<i<<","<<xb;
      }
      if(debug) cerr<<endl;
      if(debug_gpu) cerr<<endl;
    } // if run_cpu
  }else{
    for(int i=0;i<observations;++i) Xbeta_full[i] = 0;
  }
  float xbeta_reduce[observations];
  MPI_Reduce(Xbeta_full,xbeta_reduce,observations,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
  bool do_landweber = true;
  if (do_landweber){
//if(mpi_rank==0){
    float inverse_lipschitz = 2./this->landweber_constant;
    //cerr<<"LIPSCHITZ CONSTANT: "<<this->spectral_norm<<endl;
    float norm_diff = 0;
    int iter=0,maxiter = static_cast<int>(config->max_landweber);
    int converged = 0;
    float tolerance = 1e-8;
    float xxi_lambda[sub_observations];
    float new_lambda[observations];
    for(int i=0;i<observations;++i) lambda[i] = 0;
    while(!converged && iter<maxiter){
      if (slave_id>=0){
        bool testgpu2 = (run_cpu && slave_id==-10);
        if(run_gpu){
#ifdef USE_GPU
          // run GPU kernel here
          ocl_wrapper->write_to_buffer("lambda",observations,lambda);
          ocl_wrapper->run_kernel("compute_xxi_lambda",BLOCK_WIDTH,sub_observations,1,BLOCK_WIDTH,1,1);
          ocl_wrapper->read_from_buffer("XXI_lambda",sub_observations,xxi_lambda);
          if(testgpu2){
            cerr<<"debuggpu xxi_lambda GPU:";
            float test_xxi_lambda[sub_observations];
            ocl_wrapper->read_from_buffer("XXI_lambda",sub_observations,test_xxi_lambda);
            for(int i=0;i<sub_observations;++i){
              if(i>(sub_observations-10))cerr<<" "<<i<<","<<test_xxi_lambda[i];
            }
            cerr<<endl;
          }
#endif
        } // if run gpu
        if(run_cpu){
          if(testgpu2) cerr<<"debuggpu xxi_lambda CPU:";
          for(int i=0;i<sub_observations;++i){ //float xxi_vec[observations];
            //random_access_XXI->extract_vec(subject_node_offsets[mpi_rank]+i,observations,xxi_vec);
            float lam = 0;
            for(int j=0;j<observations;++j){
              lam+=XXI[i*observations+j]* lambda[j];
              //if(j<10 && lambda[j]!=0) cerr<<"i: "<<i<<" lam: "<<lam<<" XXI: "<<XXI[i*observations+j]<<" lambda: "<<lambda[j]<<endl;
            }
            xxi_lambda[i] = lam; 
            if(testgpu2 && i>(sub_observations-10)) cerr<<" "<<i<<","<<xxi_lambda[i];
          }
          if(testgpu2) cerr<<endl;
        }
      }
      MPI_Gatherv(xxi_lambda,sub_observations,MPI_FLOAT,xxi_lambda,subject_node_sizes,subject_node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
      if(mpi_rank==0){
        norm_diff = 0;
        for(int i=0;i<observations;++i){
          new_lambda[i] = lambda[i] - inverse_lipschitz *
          (xxi_lambda[i]+theta[i] - xbeta_reduce[i]);
          if(i<-10) cerr<<"XXILAMBDA: "<<xxi_lambda[i]<<" THETA "<<theta[i]<<" XBETA "<<xbeta_reduce[i]<<" NEWLAMBDA: "<<new_lambda[i]<<endl;
          norm_diff+=(new_lambda[i]-lambda[i])*(new_lambda[i]-lambda[i]);
          lambda[i] = new_lambda[i];
        }
        norm_diff=sqrt(norm_diff);
        converged = (norm_diff<tolerance);
        ++iter;
        if(config->verbose)cerr<<".";
      }
      //cerr<<"L2 norm at landweber: "<<norm_diff<<endl;
      MPI_Bcast(&converged,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&iter,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(lambda,observations,MPI_FLOAT,0,MPI_COMM_WORLD);
    }
    if(mpi_rank==0){
      if(config->verbose)cerr<<"\nLandweber iterations: "<<iter<<" norm diff: "<<norm_diff<<endl;
      for(int i=0;i<observations;++i) Xbeta_full[i] = xbeta_reduce[i];
    }
    //spectral_norm*=.9;
//}
  }else{
    if(mpi_rank==0){
      cerr<<"Doing non-landweber\n";
      float xbeta_theta[observations];
      for(int i=0;i<observations;++i){
        xbeta_theta[i] = xbeta_reduce[i]-theta[i];
        //cerr<<"i, xbetareduce, theta:"<<i<<","<<xbeta_reduce[i]<<","<<theta[i]<<endl;
      }
      //mmultiply(XXI_inv,observations,observations,xbeta_theta,1,lambda);
      //mmultiply(XXI_inv,observations,observations,xbeta_theta,1,lambda);
      if(debug) cerr<<"UPDATE_LAMBDA computing lambda observation:";
      for(int i=0;i<observations;++i){
        float xxii_vec[observations];
        for(int j=0;j<observations;++j){ xxii_vec[j] = 0; }
        random_access_XXI_inv->extract_vec(i,observations,xxii_vec);
        float lam = 0;
        for(int j=0;j<observations;++j){
          lam+=xxii_vec[j]* xbeta_theta[j];
          //if (j<10) cerr<<"i,j,xxii_vec:"<<i<<","<<j<<":"<<xxii_vec[j]<<endl;
        }
        lambda[i] = lam;
        if(debug && i%100 == 0) cerr<<" "<<i<<","<<lam;
        if(i<10) cerr<<"THETA "<<theta[i]<<" XBETA "<<xbeta_reduce[i]<<" LAMBDA: "<<lambda[i]<<endl;
      }
      if(debug) cerr<<endl;
    } // MPI rank==0
  } // Landweber condition
  MPI_Bcast(lambda,observations,MPI_FLOAT,0,MPI_COMM_WORLD);
#endif
}

void param_split_t::project_theta(){
  bool debug = false;
  if(mpi_rank==0){
    if(debug) cerr<<"PROJECT_THETA:";
    for(int i=0;i<sub_observations;++i){
      theta_project[i] = theta[i]+lambda[i];
      if(debug && i%100 == 0) cerr<<" "<<i<<","<<theta[i]<<","<<lambda[i];
      //if(i<10)cerr<<"Thetaproject "<<i<<" "<<theta_project[i]<<endl;
 
    }
    if(debug) cerr<<endl;
  }
  //MPI_Scatterv(theta_project,subject_node_sizes,subject_node_offsets,MPI_FLOAT,theta_project,sub_observations,MPI_FLOAT,0,MPI_COMM_WORLD);
}

void param_split_t::project_beta(){
  //bool debug = true;
  //bool run_cpu = false; 
  //bool run_gpu = true;
  if(slave_id>=0){
    bool debug_gpu = (run_cpu && slave_id==-10);
    float xt_lambda_arr[variables];
    if(run_gpu){

#ifdef USE_GPU
      ocl_wrapper->write_to_buffer("lambda",observations,lambda);
      ocl_wrapper->run_kernel("project_beta",BLOCK_WIDTH*subject_chunks,variables,1,BLOCK_WIDTH,1,1);
      ocl_wrapper->run_kernel("reduce_xt_lambda_chunks",BLOCK_WIDTH,variables,1,BLOCK_WIDTH,1,1);
      ocl_wrapper->read_from_buffer("Xt_lambda",variables,xt_lambda_arr);
      if(debug_gpu){
        //float Xt_lambda_chunks[variables*subject_chunks];
        float Xt_temp[variables];
        //ocl_wrapper->read_from_buffer("Xt_lambda_chunks",variables*subject_chunks,Xt_lambda_chunks);
        ocl_wrapper->read_from_buffer("Xt_lambda",variables,Xt_temp);
        cerr<<"debuggpu projectbeta GPU:";
        for(int j=0;j<variables;++j){
          //float xt_lambda = 0;
          for(int i=0;i<subject_chunks;++i){
            //xt_lambda+=Xt_lambda_chunks[j*subject_chunks+i];
          }
          if(j>(variables-10))cerr<<" "<<j<<":"<<Xt_temp[j];
          //if(j<10)cerr<<" "<<j<<":"<<xt_lambda;
        }
        cerr<<endl;
      }
#endif
    }  // if run gpu
    if(run_cpu){
      //if(debug) cerr<<"PROJECT_BETA slave "<<slave_id<<" variable:";
      //bool feasible = in_feasible_region();
      if(debug_gpu) cerr<<"debuggpu projectbeta CPU:";
      for(int j=0;j<variables;++j){
        xt_lambda_arr[j] = 0;
        // emulate what we would do on the GPU
        int SMALL_BLOCK_WIDTH = 32;
        int chunks = observations/BLOCK_WIDTH+(observations%BLOCK_WIDTH!=0);
        packedgeno_t packedbatch[SMALL_BLOCK_WIDTH];
        float subset_geno[BLOCK_WIDTH];
        for(int chunk=0;chunk<chunks;++chunk){  // each chunk is 512 variables
          //if(slave_id==0)cerr<<"Working on chunk "<<chunk<<endl;
          for(int threadindex=0;threadindex<SMALL_BLOCK_WIDTH;++threadindex){
            // load 32 into this temporary array
            packedbatch[threadindex] = packedgeno_snpmajor[j*
            packedstride_snpmajor+chunk*SMALL_BLOCK_WIDTH+threadindex];
          }
          for(int threadindex=0;threadindex<SMALL_BLOCK_WIDTH;++threadindex){
            // expand 32 elements into 512 genotypes
            int t = 0;
            for(int b=0;b<4;++b){
              for(int c=0;c<4;++c){
                subset_geno[threadindex*16+t] = 
                c2g(packedbatch[threadindex].geno[b],c);
                ++t;
              }
            }
          }
          for(int threadindex = 0;threadindex<BLOCK_WIDTH;++threadindex){
            int obs_index = chunk*BLOCK_WIDTH+threadindex;
            if(obs_index<observations){
              float g=subset_geno[threadindex]==9?0:(subset_geno[threadindex]-means[j])*precisions[j];
              //float g1=(subset_geno[threadindex]);
              //float g2 = plink_data_X_subset->get_raw_geno(j,obs_index);
              //if(g1!=g2) cerr<<"project-beta Mismatch at SNP: "<<j<<" obs "<<obs_index<<": "<<g<<","<<g2<<endl;
              xt_lambda_arr[j]+= g * lambda[obs_index];
              //if(slave_id==0 && obs_index<100)cerr<<" "<<obs_index<<":"<<g<<":"<<lambda[obs_index];
            }
          }
        }
        //if(slave_id==0)cerr<<endl;
        // gold standard
        for(int i=0;i<observations;++i){
          //float g = 0;
          //float g = plink_data_X_subset->get_geno(j,i);
          //xt_lambda+= g * lambda[i];
        }
        //if(slave_id==0)cerr<<endl;
      } // loop over variables
    } // if run cpu

    for(int j=0;j<variables;++j){
      beta_project[j] = beta[j]-xt_lambda_arr[j];
      if(debug_gpu && j>(variables-10))cerr<<" "<<j<<":"<<xt_lambda_arr[j];
//    cerr<<"PROJECT_BETA: var "<<j<<" is "<<beta_project[j]<<endl;
      //if(debug && j % 1000 == 0) cerr<<" "<<j<<","<<beta[j]<<","<<beta_project[j] ;
    } // loop over variables
    //if(debug) cerr<<endl;
    if(debug_gpu) cerr<<endl;
  } // if is slave
}

void param_split_t::update_map_distance(){
#ifdef USE_MPI
  float theta_distance = 0;
  if(mpi_rank==0){
    for(int i=0;i<observations;++i){
      float dev = theta[i]-theta_project[i];
       theta_distance+=dev*dev;
    }
  }
  float beta_distance = 0;
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

float param_split_t::get_map_distance(){
  return this->map_distance;
}

void param_split_t::update_theta(){
#ifdef USE_MPI
  bool debug = false;
  if(debug) cerr<<"UPDATE_THETA: Node "<<mpi_rank<<" entering\n";
  if(mpi_rank==0){
    float coeff = 1./(1+dist_func);
    if(config->verbose) cerr<<"UPDATE_THETA: "<<coeff<<endl;
    if(debug) cerr<<"Getting observation:";
    for(int i=0;i<sub_observations;++i){
      //theta[i] = theta_project[i]+coeff*(0-theta_project[i]);
      theta[i] = theta_project[i]+coeff*(y[i]-theta_project[i]);
      //cerr<<"Subject "<<i<<" theta "<<theta[i]<<" projection "<<theta_project[i]<<endl;
      if(debug && i%100==0) cerr<<" "<<i<<","<<theta[i]<<","<<theta_project[i];
    }
    if(debug) cerr<<endl;
  }
  if(debug) cerr<<"UPDATE_THETA: Node "<<mpi_rank<<" exiting\n";
  MPI_Scatterv(theta,subject_node_sizes,subject_node_offsets,MPI_FLOAT,theta,sub_observations,MPI_FLOAT,0,MPI_COMM_WORLD);
#endif
}

void param_split_t::update_beta(){
#ifdef USE_MPI
  bool debug = false;
  //double start = clock();
  MPI_Scatterv(constrained_beta,snp_node_sizes,snp_node_offsets,MPI_FLOAT,constrained_beta,variables,MPI_FLOAT,0,MPI_COMM_WORLD);
  if(slave_id>=0){
    bool feasible = in_feasible_region();
    if(debug) cerr<<"UPDATE_BETA: Node "<<mpi_rank<<" entering for top "<<this->current_top_k<<" feasible: "<<feasible<<endl;
    for(int j=0;j<variables;++j){
      //if( constrained_beta[j]!=0){
      //if(!feasible || constrained_beta[j]!=0){
        beta[j] = .5*(beta_project[j]+constrained_beta[j]);
      //}else{
        //beta[j] = 0;
      //}
      if(debug && j%10==0) cerr<<" "<<j<<","<<constrained_beta[j]<<","<<beta_project[j]<<","<<beta[j];
    }
    if(debug) cerr<<endl;
  }
  MPI_Gatherv(beta,variables,MPI_FLOAT,beta,snp_node_sizes,snp_node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
  update_constrained_beta();
#endif
}

void param_split_t::update_constrained_beta(){
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
      if (j<this->current_top_k){
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
      //all_constrained_beta[j] = constrained_beta[j];
      
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



bool param_split_t::in_feasible_region(){
  float mapdist = get_map_distance();
  if(config->verbose)cerr<<"IN_FEASIBLE_REGION: mapdist: "<<mapdist<<" threshold: "<<this->current_mapdist_threshold<<endl;
  bool ret= (mapdist>0 && mapdist< this->current_mapdist_threshold);
  return ret;
}

void param_split_t::parse_config_line(string & token,istringstream & iss){
  proxmap_t::parse_config_line(token,iss);
  if (token.compare("FAM_FILE")==0){
    iss>>config->fam_file;
  }else if (token.compare("SNP_BED_FILE")==0){
    iss>>config->snp_bed_file;
  }else if (token.compare("BIM_FILE")==0){
    iss>>config->bim_file;
  }else if (token.compare("TOP_K_MIN")==0){
    iss>>config->top_k_min;
  }else if (token.compare("TOP_K_MAX")==0){
    iss>>config->top_k_max;
  }else if (token.compare("SPECTRAL_NORM")==0){
    iss>>config->spectral_norm;
  }else if (token.compare("MAX_LANDWEBER")==0){
    iss>>config->max_landweber;
  }else if (token.compare("XXI_FILE")==0){
    iss>>config->xxi_file;
  }else if (token.compare("XXI_INV_FILE")==0){
    iss>>config->xxi_inv_file;
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

void param_split_t::read_dataset(){
#ifdef USE_MPI
  // figure out the dimensions
  //
  const char * snpbedfile = config->snp_bed_file.data();
  //const char * subjectbedfile = config->subject_bed_file.data();
  const char * bimfile = config->bim_file.data();
  const char * famfile = config->fam_file.data();

  this->observations = linecount(famfile);
  this->all_variables = linecount(bimfile);
  if(config->verbose) cerr<<"Subjects: "<<observations<<" and predictors: "<<all_variables<<endl;
  // taken from Eric Chi's project
  
  //bool single_mask[] = {true};
  //bool full_variable_mask[all_variables];
  bool variable_mask[all_variables];
  bool full_subject_mask[observations];
  bool subject_mask[observations];

  for(int i=0;i<observations;++i) full_subject_mask[i] = true;
  for(int i=0;i<observations;++i) subject_mask[i] = slave_id<0?true:false;
  //for(int i=0;i<all_variables;++i) full_variable_mask[i] = true;
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
    subject_node_sizes[i+1] =  observation_indices_vec[i].size();
    subject_node_offsets[i+1] = subject_offset;
    if(config->verbose)cerr<<"Offset for rank "<<i+1<<" : SNP: "<<snp_offset<<" subject: "<<subject_offset<<endl;
    snp_offset+=snp_node_sizes[i+1];
    subject_offset+=subject_node_sizes[i+1];
  }

  MPI_Type_contiguous(observations,MPI_FLOAT,&floatSubjectsArrayType);
  MPI_Type_commit(&floatSubjectsArrayType);
  this->variables=slave_id>=0?snp_indices_vec[slave_id].size():all_variables;
  this->sub_observations=slave_id>=0?observation_indices_vec[slave_id].size():observations;
  if(config->verbose) cerr<<"Node "<<mpi_rank<<" with "<<variables<<" variables.\n";
  this->y = new float[sub_observations];
  parse_fam_file(famfile,subject_mask,sub_observations,this->y);
  //load_matrix_data(config->traitfile.data(),y,observations,1,sub_observations,1,subject_mask, single_mask,true,0);
  //load_random_access_data(random_access_pheno,y,1,observations,sub_observations,1,subject_mask, single_mask);
  this->means = new float[variables];
  this->precisions = new float[variables];
  if(slave_id>=0){
    //load_random_access_data(random_access_pheno,all_y,1,observations,observations,1,full_subject_mask, single_mask);
    //load_matrix_data(config->traitfile.data(),all_y,observations,1,observations,1,full_subject_mask, single_mask,true,0);
    //parse_fam_file(famfile,full_subject_mask,observations,this->all_y);
    parse_bim_file(bimfile,variable_mask,variables,means,precisions);
  }
  if(slave_id>=0){
    //cerr<<"Slave "<<slave_id<<" loading genotypes\n";
    plink_data_X_subset = new plink_data_t(all_variables,variable_mask,observations,full_subject_mask); 
    plink_data_X_subset->load_data(snpbedfile);
    plink_data_X_subset->set_mean_precision(plink_data_t::ROWS,means,precisions);
    plink_data_X_subset->transpose(plink_data_X_subset_subjectmajor);
    if(config->verbose) cerr<<"Transposed!\n";
    this->packedgeno_snpmajor = plink_data_X_subset->get_packed_geno(this->packedstride_snpmajor);
    this->packedgeno_subjectmajor = plink_data_X_subset_subjectmajor->get_packed_geno(this->packedstride_subjectmajor);
    //load_random_access_data(random_access_geno,X,all_variables,observations,observations,variables, full_subject_mask,variable_mask);
    //cerr<<"Slave "<<slave_id<<" loading genotypes stripe\n";
    //load_random_access_data(random_access_geno,X_stripe,all_variables,observations,observations,all_variables, subject_mask,full_variable_mask);
    //if (config->verbose) cerr<<"Slave "<<slave_id<<" done loading genotypes\n";
    //load_matrix_data(genofile,X,observations,all_variables,observations,variables, full_subject_mask,variable_mask,true,0);
    //load_matrix_data(genofile,X_stripe,observations,all_variables,observations,all_variables, subject_mask,full_variable_mask,true,0);
    // At this point we should standardize all the variables
    //if(config->verbose) cerr<<"Standardizing variables\n";
    //if(config->verbose) cerr.flush();
    //standardize(X,observations,variables);
    //standardize(X_stripe,sub_observations,all_variables);
    // SANITY CHECK THAT WE READ IN THE DATASET CORRECTLY
    //
    bool debugoutput=false;
    if(debugoutput){
//      ostringstream oss;
//      oss<<"X."<<mpi_rank<<".txt";
//      ofstream ofs(oss.str().data());
//      for(int i=0;i<observations;++i){
//        for(int j=0;j<this->variables;++j){
//          if(j) ofs<<"\t";
//          ofs<<X[i*this->variables+j];
//        }
//        ofs<<endl;
//      }
//      ofs.close();
//      ostringstream oss2;
//      oss2<<"X_STRIPE."<<mpi_rank<<".txt";
//      ofstream ofs2(oss2.str().data());
//      for(int i=0;i<sub_observations;++i){
//        for(int j=0;j<all_variables;++j){
//          if(j) ofs2<<"\t";
//          ofs2<<X_stripe[i*all_variables+j];
//        }
//        ofs2<<endl;
//      }
//      ofs2.close();
    }
  }
  if(config->verbose)cerr<<"Read input done\n";
#endif
}

void param_split_t::parse_fam_file(const char * infile, bool * mask,int len, float * newy){
  int maskcount = 0;
  if(config->verbose)cerr<<"Allocating Y of len "<<len<<endl;
  //newy = new float[len];
  for(int i=0;i<observations;++i){
    maskcount+=mask[i];
  }
  //cerr<<"Node "<<mpi_rank<<" FAM file Mask count is "<<maskcount<<" and len is "<<len<<endl;
  assert(maskcount==len);
  ifstream ifs(infile);
  if(!ifs.is_open()){
    cerr<<"Cannot open fam file "<<infile<<endl;
    exit(1);
  }
  int j = 0;
  for(int i=0;i<observations;++i){
    string line;
    getline(ifs,line);
    istringstream iss(line);
    string token;
    for(int k=0;k<6;++k) iss>>token;
    float outcome;
    iss>>outcome;
    if(mask[i]){
      newy[j] = outcome;
      ++j;
    }
    
  }
  ifs.close();
  
  
}
void param_split_t::parse_bim_file(const char * infile, bool * mask,int len, float * means, float * precisions){
  int maskcount = 0;
  for(int i=0;i<all_variables;++i){
    maskcount+=mask[i];
  }
  if(config->verbose)cerr<<"Node "<<mpi_rank<<" BIM file Mask count is "<<maskcount<<" and len is "<<len<<endl;
  assert(maskcount==len);
  ifstream ifs(infile);
  if(!ifs.is_open()){
    cerr<<"Cannot open bim file "<<infile<<endl;
    exit(1);
  }
  int j = 0;
  for(int i=0;i<all_variables;++i){
    string line;
    getline(ifs,line);
    istringstream iss(line);
    string token;
    for(int k=0;k<6;++k) iss>>token;
    float mean,precision;
    iss>>mean>>precision;
    if(mask[i]) {
      if (isnan(mean) || isnan(precision)){
        cerr<<"Something wrong with the input at line: "<<line<<endl;
        exit(1);
      }
      means[j] = mean;
      precisions[j] = precision;
      //if(slave_id==1) cerr<<j<<": "<<mean<<endl;
      ++j;
    }
  }
  //exit(1);
  //cerr<<"J is "<<j<<endl;
  ifs.close();
}

//float param_split_t::compute_marginal_beta(float * xvec){
//  // dot product first;
//  float xxi = 0;
//  float xy = 0;
//  for(int i=0;i<observations;++i){
//    xxi +=xvec[i]*xvec[i];
//    xy +=xvec[i]*all_y[i];
//  }
//  if(fabs(xxi)<1e-5) return 0;
//  xxi = 1./xxi;
//  return xxi*xy;
//}

//void param_split_t::init_marginal_screen(){
//  if(slave_id>=0){
//    ostringstream oss_marginal;
//    oss_marginal<<config->marginal_file_prefix<<"."<<mpi_rank<<".txt";
//    ifstream ifs_marginal(oss_marginal.str().data());
//    if (ifs_marginal.is_open()){
//      if(config->verbose) cerr<<"Using cached copy of marginal betas\n";
//      string line;
//      for(int j=0;j<variables;++j){
//        getline(ifs_marginal,line);
//        istringstream iss(line);
//        iss>>beta[j];
//      }
//      ifs_marginal.close();
//    }else{
//      if(config->verbose) cerr<<"Creating cached copy of marginal betas\n";
//      ofstream ofs_marginal(oss_marginal.str().data());
//      for(int j=0;j<variables;++j){
//        beta[j] = compute_marginal_beta(XT+j*observations);
//        ofs_marginal<<beta[j]<<endl;
//      }
//      ofs_marginal.close();
//    }
//    //update_Xbeta(beta);
//    //float xbeta_norm = 0;
//    //for(int i=0;i<this->observations;++i){
//      //theta_project[i] = theta[i] = y[i];
//      //xbeta_norm+=Xbeta[i]*Xbeta[i];
//      //Xbeta[i] = 0;
//      //theta_project[i] = theta[i] = Xbeta[i];
//    //}
//    //xbeta_norm=sqrt(xbeta_norm);
//    //cerr<<"Xbeta norm for node "<<mpi_rank<<" is "<<xbeta_norm<<endl;
//  }
//  for(int j=0;j<variables;++j){
//    constrained_beta[j] = beta_project[j] = beta[j] = 0;
//  }
//}

//void param_split_t::compute_XX(){
//#ifdef USE_MPI
//  int cached = false;
//  ostringstream oss_xx;
//  oss_xx<<config->xx_file_prefix<<"."<<mpi_rank<<".bin";
//  string xx_file = oss_xx.str();
//  if(mpi_rank==0){
//    cerr<<"Initializing XX and opening "<<xx_file<<"\n";
//    ifstream ifs_xx(xx_file.data());
//    if (ifs_xx.is_open()){
//      cached = true;
//      ifs_xx.close();
//    }
//  }    
//  MPI_Bcast(&cached,1,MPI_INT,0,MPI_COMM_WORLD);
//  if(cached){
//    if(mpi_rank==0){
//      if(config->verbose) cerr<<"Using cached copy of X %*% t(X)\n";
//      random_access_t access(xx_file.data(),observations,observations);
//      for(int i=0;i<observations;++i){
//        for(int j=0;j<observations;++j){
//          XX[i*observations+j] = access.extract_val(i,j);
//        }
//      }
//      if(config->verbose) cerr<<"Done reading X %*% t(X)\n";
//    }
//  }else{
//
//    if(mpi_rank==0){
//      if(config->verbose) cerr<<"Cannot find cached copy ("<<xx_file <<") of XX. Computing\n";
//      bool test = false;
//      if(test){
//        float * X2 = new float[observations*variables];
//        ifstream ifs(config->genofile.data());
//        cerr<<"Reading from "<<config->genofile<<" into "<<observations<<" by "<<variables<<endl;
//        for(int i=0;i<observations;++i){
//          string line;
//          getline(ifs,line);
//          istringstream iss(line);
//          for(int j=0;j<variables;++j){
//            iss>>X2[i*variables+j];
//          } 
//        }
//        ifs.close();
//        mmultiply(X2,observations,variables,XX);
//        delete[]X2;
//        for(int i=0;i<observations;++i){
//          for(int j=0;j<observations;++j){
//          }
//          //cerr<<endl;
//        }
//      }
//    }
//    cerr<<"Random access on "<<config->genofile.data()<<endl;
//    random_access_t access(config->genofile.data(),all_variables,observations);
//    float anchor[all_variables];
//    //ifstream ifs_full;
//    //if(mpi_rank==0){
//       //ifs_full.open(config->genofile.data());
//    //}
//    if(config->verbose) cerr<<"Dot products for observation";
//    for(int i=0;i<observations;++i){
//      if(config->verbose) cerr<<" "<<i;
//      if(mpi_rank==0){
//        cerr<<".";
//        for(int j=0;j<all_variables;++j){
//          anchor[j] = access.extract_val(j,i);
//        } 
//        cerr<<".";
//      }
//      MPI_Bcast(anchor,all_variables,MPI_FLOAT,0,MPI_COMM_WORLD);
//      float dot_prod[sub_observations];
//      if(slave_id>=0){
//        if (slave_id==0) cerr<<"0";
//        for(int i2=0;i2<sub_observations;++i2){
//          dot_prod[i2] = 0;
//          for(int j=0;j<all_variables;++j){
//            dot_prod[i2]+=anchor[j]*X_stripe[i2*all_variables+j];
//          }
//        }
//        if (slave_id==0) cerr<<"0";
//      }
//      MPI_Gatherv(dot_prod,sub_observations,MPI_FLOAT,dot_prod,subject_node_sizes,subject_node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
//      if(mpi_rank==0){
//        for(int i2=0;i2<sub_observations;++i2){
//          XX[i*sub_observations+i2] = dot_prod[i2]; 
//        }
//      }
//    }
//    if(config->verbose) cerr<<endl;
//
//    if(mpi_rank==0){
//      ofstream ofs_xx(xx_file.data());
//      for(int i=0;i<observations;++i){
//        float outvec[observations];
//        for(int j=0;j<observations;++j){
//          outvec[j] =  XX[i*observations+j];
//        }
//        random_access_t::marshall(ofs_xx,outvec,observations);
//      }
//      ofs_xx.close();
//    }
//  }
//#endif
//}

void param_split_t::init_xxi_inv(){
#ifdef USE_MPI
  int cached  = 0;
  //ostringstream oss_xxi_inv;
  //oss_xxi_inv<<config->xxi_inv_file_prefix<<"."<<mpi_rank<<".bin";
  string xxi_file = config->xxi_file;
  string xxi_inv_file = config->xxi_inv_file;
  //string xxi_file = oss_xxi_inv.str();
  //cerr<<"Initializing XXI_INV on node "<<mpi_rank<<" and opening "<<xxi_file<<"\n";
  if(mpi_rank==0){
    ifstream ifs_xxi_inv(xxi_inv_file.data());
    if (ifs_xxi_inv.is_open()){
      ++cached;
      ifs_xxi_inv.close();
    }
    ifstream ifs_xxi(xxi_file.data());
    if (ifs_xxi.is_open()){
      ++cached;
      ifs_xxi.close();
    }
  }    
  MPI_Bcast(&cached,1,MPI_INT,0,MPI_COMM_WORLD);
  if(cached==2){
    if (slave_id>=0){
      if(config->verbose) cerr<<"Using cached copies of singular values and vectors\n";
      if(config->verbose)cerr<<"Using XXI INV file "<<xxi_inv_file.data()<<endl;
      random_access_XXI_inv = new random_access_t(xxi_inv_file.data(),observations,observations);
      if(config->verbose)cerr<<"Using XXI file "<<xxi_file.data()<<endl;
      random_access_XXI = new random_access_t(xxi_file.data(),observations,observations);
      for(int i=0;i<sub_observations;++i){
        //cerr<<"Loading row "<<i<<" of size "<<observations<<" into XXI\n";
        random_access_XXI->extract_vec(subject_node_offsets[mpi_rank]+i,observations,XXI+i*observations);
        if(config->verbose && i%1000==0) cerr<<"Subject "<<i<<" loaded\n";
      }
      if(config->verbose)cerr<<"XXI initialized\n";
    }
  }else{
    if(config->verbose) cerr<<"Cannot find cached copy ("<<xxi_file <<") of singular values and vectors. Please pre-compute this. Exiting.\n";
    //XX = new float[observations * observations];
    MPI_Finalize();
    exit(0);
//    compute_XX();
//    if (mpi_rank==0){
//      if(config->verbose) cerr<<"Allocating GSL xx of size "<<observations<<endl;
//      gsl_matrix * tempxx = gsl_matrix_alloc(observations,observations);
//      for(int i=0;i<observations;++i){
//        for(int j=0;j<observations;++j){
//          gsl_matrix_set(tempxx,i,j,XX[i*observations+j]);
//        }
//      }
//      if(config->verbose) cerr<<"Performing eigen decomp\n";
//      if(config->verbose) cerr.flush();
//      //This function allocates a workspace for computing eigenvalues and eigenvectors of n-by-n real symmetric matrices. The size of the workspace is O(4n).
//      gsl_eigen_symmv_workspace * eigen_work =  gsl_eigen_symmv_alloc (observations);
//      gsl_matrix * tempv = gsl_matrix_alloc(observations,observations);
//      gsl_vector * temps = gsl_vector_alloc(observations);
//      //This function computes the eigenvalues and eigenvectors of the real symmetric matrix A. Additional workspace of the appropriate size must be provided in w. The diagonal and lower triangular part of A are destroyed during the computation, but the strict upper triangular part is not referenced. The eigenvalues are stored in the vector eval and are unordered. The corresponding eigenvectors are stored in the columns of the matrix evec. For example, the eigenvector in the first column corresponds to the first eigenvalue. The eigenvectors are guaranteed to be mutually orthogonal and normalised to unit magnitude.
//        
//      int code = gsl_eigen_symmv(tempxx, temps, tempv, eigen_work);
//      if (code!=0) if(config->verbose) cerr<<"Returning nonzero code in eigendecomp\n";
//      gsl_matrix * tempvdi = gsl_matrix_alloc(observations,observations);
//      for(int i=0;i<observations;++i){
//        for(int j=0;j<observations;++j){
//          float vdi = gsl_matrix_get(tempv,i,j)/(1.+gsl_vector_get(temps,j));
//          gsl_matrix_set(tempvdi,i,j,vdi);
//        }
//      }
//      if(mpi_rank==-1){
//        ofstream ofs_rtest1("r_test1.txt");
//        ofstream ofs_rtest2("r_test2.txt");
//        ofstream ofs_rtest3("r_test3.txt");
//        for(int i=0;i<observations;++i){
//          for(int j=0;j<observations;++j){
//            if (j) ofs_rtest1<<"\t";
//            ofs_rtest1<<gsl_matrix_get(tempxx,i,j);
//            if (i==j) ofs_rtest2<<gsl_vector_get(temps,i);
//            if (j) ofs_rtest3<<"\t";
//            ofs_rtest3<<gsl_matrix_get(tempv,i,j);
//          }
//          ofs_rtest1<<endl;
//          ofs_rtest2<<endl;
//          ofs_rtest3<<endl;
//        }
//        ofs_rtest1.close();
//        ofs_rtest2.close();
//        ofs_rtest3.close();
//        exit(1);
//      }
//      //
//      //Function: void gsl_eigen_symmv_free (gsl_eigen_symmv_workspace * w)
//      //This function frees the memory associated with the workspace w.
//      //
//      gsl_matrix_free(tempxx);
//      gsl_vector_free(temps);
//      gsl_eigen_symmv_free(eigen_work);
//
//      if(config->verbose) cerr<<"Computing VDIV\n";
//      if(config->verbose) cerr.flush();
//      gsl_matrix * tempvdiv = gsl_matrix_alloc(observations,observations);
//      gsl_blas_dgemm(CblasNoTrans,CblasTrans,1,tempvdi,tempv,0,tempvdiv);
//      gsl_matrix_free(tempvdi);
//      gsl_matrix_free(tempv);
//      if(config->verbose) cerr<<"Writing to binary file "<<xxi_file<<"\n";
//      if(config->verbose) cerr.flush();
//      ofstream ofs_xxi_inv(xxi_file.data());
//      for(int i=0;i<observations;++i){
//        float outvec[observations];
//        for(int j=0;j<observations;++j){
//          outvec[j] =  XXI_inv[i*observations+j] = gsl_matrix_get(tempvdiv,i,j);
//        }
//        random_access_t::marshall(ofs_xxi_inv,outvec,observations);
//      }
//      ofs_xxi_inv.close();
//      gsl_matrix_free(tempvdiv);
//    } // if mpi_rank==0
  } // if cached/not cached
#endif
}

void param_split_t::init_gpu(){
#ifdef USE_GPU
  // init GPU
  int subject_chunk_clusters = subject_chunks/BLOCK_WIDTH+(subject_chunks%BLOCK_WIDTH!=0);
  int snp_chunk_clusters = snp_chunks/BLOCK_WIDTH+(snp_chunks%BLOCK_WIDTH!=0);
  int platform_id = 0;
  int device_id = slave_id;
  vector<string> sources;
  sources.push_back("cl_constants.h");
  sources.push_back("packedgeno.c");
  sources.push_back("param_split.c");
  vector<string> paths;
  for(uint j=0;j<sources.size();++j){
    ostringstream oss;
    oss<<config->kernel_base<<"/"<<sources[j];
    paths.push_back(oss.str());
  }
  bool debug_ocl = false;
  ocl_wrapper = new ocl_wrapper_t(debug_ocl);
  ocl_wrapper->init(paths,platform_id,device_id);
  // create kernels
  ocl_wrapper->create_kernel("update_lambda");
  ocl_wrapper->create_kernel("reduce_xbeta_chunks");
  ocl_wrapper->create_kernel("project_beta");
  ocl_wrapper->create_kernel("reduce_xt_lambda_chunks");
  ocl_wrapper->create_kernel("compute_xxi_lambda");
  // create buffers
  ocl_wrapper->create_buffer<packedgeno_t>("packedgeno_snpmajor",CL_MEM_READ_ONLY,variables*packedstride_snpmajor);
  ocl_wrapper->create_buffer<packedgeno_t>("packedgeno_subjectmajor",CL_MEM_READ_ONLY,observations*packedstride_subjectmajor);
  ocl_wrapper->create_buffer<float>("means",CL_MEM_READ_ONLY,variables);
  ocl_wrapper->create_buffer<float>("precisions",CL_MEM_READ_ONLY,variables);
  ocl_wrapper->create_buffer<float>("beta",CL_MEM_READ_ONLY,variables);
  ocl_wrapper->create_buffer<float>("beta_project",CL_MEM_READ_WRITE,variables);
  ocl_wrapper->create_buffer<float>("Xbeta_full",CL_MEM_READ_WRITE,observations);
  ocl_wrapper->create_buffer<float>("Xbeta_chunks",CL_MEM_READ_WRITE,observations * snp_chunks);
  ocl_wrapper->create_buffer<float>("lambda",CL_MEM_READ_ONLY,observations);
  ocl_wrapper->create_buffer<float>("Xt_lambda_chunks",CL_MEM_READ_WRITE,variables * subject_chunks);
  ocl_wrapper->create_buffer<float>("Xt_lambda",CL_MEM_READ_WRITE,variables);
  ocl_wrapper->create_buffer<float>("XXI",CL_MEM_READ_ONLY,sub_observations*observations);
  ocl_wrapper->create_buffer<float>("XXI_lambda",CL_MEM_READ_WRITE,sub_observations);
  // initialize buffers
  ocl_wrapper->write_to_buffer("packedgeno_snpmajor",variables*packedstride_snpmajor,packedgeno_snpmajor);
  ocl_wrapper->write_to_buffer("packedgeno_subjectmajor",observations*packedstride_subjectmajor,packedgeno_subjectmajor);
  ocl_wrapper->write_to_buffer("means",variables,means);
  ocl_wrapper->write_to_buffer("precisions",variables,precisions);
  ocl_wrapper->write_to_buffer("XXI",sub_observations*observations,XXI);
  // add Kernel arguments
  ocl_wrapper->add_kernel_arg("update_lambda",observations);
  ocl_wrapper->add_kernel_arg("update_lambda",variables);
  ocl_wrapper->add_kernel_arg("update_lambda",snp_chunks);
  ocl_wrapper->add_kernel_arg("update_lambda",packedstride_subjectmajor);
  ocl_wrapper->add_kernel_arg("update_lambda",*(ocl_wrapper->get_buffer("packedgeno_subjectmajor")));
  ocl_wrapper->add_kernel_arg("update_lambda",*(ocl_wrapper->get_buffer("Xbeta_chunks")));
  ocl_wrapper->add_kernel_arg("update_lambda",*(ocl_wrapper->get_buffer("beta")));
  ocl_wrapper->add_kernel_arg("update_lambda",*(ocl_wrapper->get_buffer("means")));
  ocl_wrapper->add_kernel_arg("update_lambda",*(ocl_wrapper->get_buffer("precisions")));
  ocl_wrapper->add_kernel_arg("update_lambda",cl::__local(sizeof(packedgeno_t) * SMALL_BLOCK_WIDTH));
  ocl_wrapper->add_kernel_arg("update_lambda",cl::__local(sizeof(float) * BLOCK_WIDTH));
  ocl_wrapper->add_kernel_arg("reduce_xbeta_chunks",variables);
  ocl_wrapper->add_kernel_arg("reduce_xbeta_chunks",snp_chunks);
  ocl_wrapper->add_kernel_arg("reduce_xbeta_chunks",snp_chunk_clusters);
  ocl_wrapper->add_kernel_arg("reduce_xbeta_chunks",*(ocl_wrapper->get_buffer("Xbeta_chunks")));
  ocl_wrapper->add_kernel_arg("reduce_xbeta_chunks",*(ocl_wrapper->get_buffer("Xbeta_full")));
  ocl_wrapper->add_kernel_arg("reduce_xbeta_chunks",cl::__local(sizeof(float) * BLOCK_WIDTH));
  ocl_wrapper->add_kernel_arg("project_beta",observations);
  ocl_wrapper->add_kernel_arg("project_beta",variables);
  ocl_wrapper->add_kernel_arg("project_beta",subject_chunks);
  ocl_wrapper->add_kernel_arg("project_beta",packedstride_snpmajor);
  ocl_wrapper->add_kernel_arg("project_beta",*(ocl_wrapper->get_buffer("packedgeno_snpmajor")));
  ocl_wrapper->add_kernel_arg("project_beta",*(ocl_wrapper->get_buffer("Xt_lambda_chunks")));
  ocl_wrapper->add_kernel_arg("project_beta",*(ocl_wrapper->get_buffer("lambda")));
  ocl_wrapper->add_kernel_arg("project_beta",*(ocl_wrapper->get_buffer("means")));
  ocl_wrapper->add_kernel_arg("project_beta",*(ocl_wrapper->get_buffer("precisions")));
  ocl_wrapper->add_kernel_arg("project_beta",cl::__local(sizeof(packedgeno_t) * SMALL_BLOCK_WIDTH));
  ocl_wrapper->add_kernel_arg("project_beta",cl::__local(sizeof(float) * BLOCK_WIDTH));
  ocl_wrapper->add_kernel_arg("reduce_xt_lambda_chunks",observations);
  ocl_wrapper->add_kernel_arg("reduce_xt_lambda_chunks",subject_chunks);
  ocl_wrapper->add_kernel_arg("reduce_xt_lambda_chunks",subject_chunk_clusters);
  ocl_wrapper->add_kernel_arg("reduce_xt_lambda_chunks",*(ocl_wrapper->get_buffer("Xt_lambda_chunks")));
  ocl_wrapper->add_kernel_arg("reduce_xt_lambda_chunks",*(ocl_wrapper->get_buffer("Xt_lambda")));
  ocl_wrapper->add_kernel_arg("reduce_xt_lambda_chunks",cl::__local(sizeof(float) * BLOCK_WIDTH));
  ocl_wrapper->add_kernel_arg("compute_xxi_lambda",observations);
  ocl_wrapper->add_kernel_arg("compute_xxi_lambda",subject_chunks);
  ocl_wrapper->add_kernel_arg("compute_xxi_lambda",*(ocl_wrapper->get_buffer("XXI")));
  ocl_wrapper->add_kernel_arg("compute_xxi_lambda",*(ocl_wrapper->get_buffer("lambda")));
  ocl_wrapper->add_kernel_arg("compute_xxi_lambda",*(ocl_wrapper->get_buffer("XXI_lambda")));
  ocl_wrapper->add_kernel_arg("compute_xxi_lambda",cl::__local(sizeof(float) * BLOCK_WIDTH));
#endif
}

void param_split_t::init(string config_file){
#ifdef USE_MPI
  if(this->single_run){
    MPI::Init();
  }
  this->mpi_numtasks = MPI::COMM_WORLD.Get_size();
  this->slaves = mpi_numtasks-1;
  this->mpi_rank = MPI::COMM_WORLD.Get_rank();
  this->slave_id = mpi_rank-1;
  config->beta_epsilon = 1e-3;
  //config->marginal_file_prefix = "marginal";
  //config->xxi_inv_file_prefix = "xxi_inv";
  //config->xx_file_prefix = "xx";
#endif
  proxmap_t::init(config_file);
  //if (this->slave_id>=0)  config->verbose = false;
  if(config->verbose) cerr<<"Configuration initialized\n";
}

void param_split_t::allocate_memory(){
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

  // Now read the dataset
  read_dataset();

  if(slave_id>=0){
    XXI = new float[sub_observations * observations];
    
    // Also, set up X transpose too, a constant matrix variable
//    this->XT = new float[this->variables*observations];
//    for(int j=0;j<this->variables;++j){
//      for(int i=0;i<observations;++i){
//        XT[j*observations+i] = X[i*this->variables+j];
//      }
//    }
  }else{
//    XXI_inv = new float[observations*observations];
  }
  // this procedure will do the SVD at the master and then broadcast to slaves
  init_xxi_inv();
  // ALLOCATE MEMORY FOR ESTIMATED PARAMETERS
  this->last_residual = 1e10;
  this->residual = 0;
  //this->active_set_size = 0;
  //this->active_set = new bool[this->variables];
  this->beta = new float[this->variables];
  this->all_beta = new float[this->all_variables];
  this->last_beta = new float[this->variables];
  this->beta_project = new float[this->variables];
  //this->all_constrained_beta = new float[this->all_variables];
  this->constrained_beta = new float[this->variables];
  this->lambda = new float[observations];
  this->Xbeta_full = new float[observations];
  // the LaGrange multiplier
  this->theta = new float[sub_observations];
  this->theta_project = new float[sub_observations];
  //init_marginal_screen();
  this->BLOCK_WIDTH = plink_data_t::PLINK_BLOCK_WIDTH;

  this->subject_chunks = observations/BLOCK_WIDTH+(observations%BLOCK_WIDTH!=0);
  this->snp_chunks = variables/BLOCK_WIDTH+(variables%BLOCK_WIDTH!=0);
  // initializations
  this->landweber_constant = config->spectral_norm;
  this->last_mapdist = 1e10;
  this->current_mapdist_threshold = config->mapdist_threshold;
  for(int j=0;j<this->all_variables;++j){
    all_beta[j] = 0;
  }
  for(int j=0;j<this->variables;++j){
    last_beta[j] = beta[j] = 0;
    constrained_beta[j] = 0;
    beta_project[j] = 0;
  }
  for(int i=0;i<this->observations;++i){
    lambda[i] = 0;
  }
 for(int i=0;i<this->sub_observations;++i){
    float init_val = 0;
    theta[i] = theta_project[i] = init_val;
  }
  this->map_distance = 0;
  this->current_top_k = config->top_k_max;
  this->last_BIC = 1e10;
  proxmap_t::allocate_memory();
  if(this->run_gpu && slave_id>=0){
#ifdef USE_GPU
    init_gpu();
#endif
  }
#endif
}

float param_split_t::infer_epsilon(){
  float new_epsilon=last_epsilon;
#ifdef USE_MPI
  //if (mu==0) return config->epsilon_max;
  if (mpi_rank==0){
    ofs_debug<<"INFER_EPSILON: Inner iterate: "<<iter_rho_epsilon<<endl;
    //if(iter_rho_epsilon==0){
      new_epsilon = this->map_distance;
      if(new_epsilon==0) new_epsilon = config->epsilon_min;
    //}
  }
  MPI_Bcast(&new_epsilon,1,MPI_FLOAT,0,MPI_COMM_WORLD);
#endif
  return new_epsilon;
}

float param_split_t::infer_rho(){
  float new_rho = last_rho;
#ifdef USE_MPI
  if(mu==0) return config->rho_min;
  if (mpi_rank==0){
    ofs_debug<<"INFER_RHO: Inner iterate: "<<iter_rho_epsilon<<endl;
    if(iter_rho_epsilon==0){
      //cerr<<"DEBUG: residual "<<residual<<" map dist "<<map_distance<<" epsilon "<<epsilon<<endl;
      if(track_residual){
        new_rho = rho_distance_ratio * residual * sqrt(map_distance+epsilon)/map_distance;
        
      }else{
        bool mapdist_stalled = fabs(last_mapdist - map_distance)/last_mapdist<config->mapdist_epsilon;
        if(mapdist_stalled) {
          new_rho = last_rho * config->rho_scale_fast;
          cerr<<"INFER_RHO: Map dist stalled, accelerating rho increment\n";
        }else{
          new_rho = last_rho + config->rho_scale_slow;
        }
        this->last_mapdist = this->map_distance;
      }
      if(isinf(new_rho) || isnan(new_rho)){
        new_rho = last_rho;
        cerr<<"INFER_RHO: Overflow with new rho, backtracking to "<<new_rho<<endl;
      }
    }else{
      new_rho = last_rho;
    }
    //}
    //new_rho = scaler * this->slaves * sqrt(this->map_distance+epsilon);
    //if (config->verbose) cerr <<"INFER_RHO: mu: "<<this->mu<<" new rho: "<<new_rho<<endl;
  }
  MPI_Bcast(&new_rho,1,MPI_FLOAT,0,MPI_COMM_WORLD);
#endif
  return new_rho;
}

void param_split_t::initialize(){
  if (mpi_rank==0){
    if(config->verbose) cerr<<"Mu iterate: "<<iter_mu<<" mu="<<mu<<" of "<<config->mu_max<<endl;
  }
  
}

bool param_split_t::finalize_inner_iteration(){
  //int proceed = false;
#ifdef USE_MPI
  if(mpi_rank==0){
//    float active_norm = 0;
//    float inactive_norm = 0;
//    //active_set_size = 0;
//    for(int j=0;j<variables;++j){
//      //active_set[j]=fabs(beta[j])>config->beta_epsilon;
//      //active_set_size+=active_set[j];
//      if(constrained_beta[j]==0){
//        inactive_norm+=fabs(beta[j]);
//      }else{
//        active_norm+=fabs(beta[j]);
//      }
//    }
//    if (config->verbose) cerr<<"active norm: "<<active_norm<<" inactive norm: "<<inactive_norm<<endl;
//    //proceed = (mu==0 || active_set_size>this->current_top_k);
  }
  //print_output();
  //MPI_Bcast(&proceed,1,MPI_INT,0,MPI_COMM_WORLD);
#endif
  return true;
  //return proceed;
}

bool param_split_t::finalize_iteration(){
  int proceed = false; 
#ifdef USE_MPI
  if(mpi_rank==0){
    float diff_norm = 0;
    for(int j=0;j<variables;++j){
      if(constrained_beta[j]!=0){
        float dev = beta[j]-last_beta[j];
        diff_norm+=dev*dev;
      }
      last_beta[j] = beta[j];
    }
    diff_norm = sqrt(diff_norm);
    bool abort = false;
    bool top_k_finalized = false;
    if(rho > config->rho_max || (this->last_mapdist>0 && this->map_distance> last_mapdist)){ 
      cerr<<"FINALIZE_ITERATION: Failed to meet constraint. Last distance: "<<last_mapdist<<" current: "<<map_distance<<".\n";
      top_k_finalized = true;
    }
    if(current_BIC > last_BIC){
      cerr<<"FINALIZE_ITERATION: BIC grew from "<<last_BIC<<" to "<<current_BIC<<". Aborting search\n";
      top_k_finalized = true;
      abort = true;
    }
    if(in_feasible_region() && diff_norm<config->beta_epsilon){
      cerr<<"FINALIZE_ITERATION: Beta norm difference is "<<diff_norm<<" threshold: "<<config->beta_epsilon<<endl;
      top_k_finalized = true;
    }
    if (top_k_finalized){
      int p = variables;
      ostringstream oss;
      oss<<"betas.k."<<current_top_k<<".txt";
      string filename=oss.str();
      cerr<<"FINALIZE_ITERATION: Dumping parameter contents into "<<filename<<"\n";
      ofstream ofs(filename.data());
      ofs<<"BIC\t"<<current_BIC<<endl;
      ofs<<"INDEX\tBETA\n";
      for(int j=0;j<p;++j){
        if(constrained_beta[j]!=0){
          ofs<<j<<"\t"<<beta[j]<<"\n";
        }
      }
      ofs.close();
      last_BIC = current_BIC;
      --this->current_top_k;
    }
    proceed = (!abort) && (!top_k_finalized || this->current_top_k >= config->top_k_min);
  }
  MPI_Bcast(&proceed,1,MPI_INT,0,MPI_COMM_WORLD);
#endif
  return proceed;
}
  

void param_split_t::iterate(){
  //cerr<<"ITERATE: "<<mpi_rank<<endl;
  //if(mpi_rank==0) cerr<<"updatelambda\n";
  update_lambda();
  //if(mpi_rank==0) cerr<<"project theta\n";
  project_theta();
  //if(mpi_rank==0) cerr<<"project beta\n";
  project_beta();
  //if(mpi_rank==0) cerr<<"update map 1\n";
  update_map_distance();
  //if(mpi_rank==0) cerr<<"update theta\n";
  update_theta();
  //if(mpi_rank==0) cerr<<"update beta\n";
  update_beta();
  //update_map_distance();
  ++total_iterations;
}

void param_split_t::print_output(){
  if(mpi_rank==0 ){
  //if(mpi_rank==0 && active_set_size<=this->current_top_k){
    cerr<<"Mu: "<<mu<<" rho: "<<rho<<" epsilon: "<<epsilon<<" total iterations: "<<this->total_iterations<<" mapdist: "<<this->map_distance<<endl;
    cerr<<"INDEX\tBETA(of "<<this->current_top_k<<")\n";
    //for(int j=0;j<10;++j){
    for(int j=0;j<variables;++j){
      if (constrained_beta[j]!=0){
        cerr<<j<<"\t"<<beta[j]<<endl;
      }
      //if (active_set[j]){
        //cerr<<"+";
        //cerr<<"\t"<<j<<"\t"<<beta[j]<<endl;
      //}else{
        //cerr<<"-";
        //cerr<<"\t"<<j<<"\t"<<beta[j]<<endl;
        //beta[j] = 0;
      //}
    }
    ofs_debug<<"Done!\n";
  }
}


bool param_split_t::proceed_qn_commit(){
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

float param_split_t::evaluate_obj(){
  float obj=0;
#ifdef USE_MPI
  //int bypass = 0;
//  cerr<<"EVALUATE_OBJ "<<mpi_rank<<" sub_obs: "<<sub_observations<<"\n";
  //MPI_Gatherv(Xbeta,sub_observations,MPI_FLOAT,Xbeta,subject_node_sizes,subject_node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
  if (mpi_rank==0){
    last_residual = residual;
    residual = 0;
    for(int i=0;i<observations;++i){
      //residual+=(y[i]-theta[i])*(y[i]-theta[i]);
      if(i<-10) cerr<<"RESIDUAL: "<<Xbeta_full[i]<<","<<theta[i]<<endl;
      residual+=(Xbeta_full[i]-theta[i])*(Xbeta_full[i]-theta[i]);
    }
    //float penalty = 0;
    //for(int j=0;j<variables;++j){
      //penalty+=nu[j]*fabs(beta[j]);
    //}
    float proxdist = get_prox_dist_penalty(); 
    obj = .5*residual+proxdist;
    current_BIC = log(observations) * this->current_top_k + observations * log(residual/observations);
    cerr<<"EVALUATE_OBJ: norm1 "<<residual<<" PROXDIST: "<<proxdist<<" FULL: "<<obj<<" BIC: "<<current_BIC<<endl;
  }
  //MPI_Bcast(&bypass,1,MPI_INT,0,MPI_COMM_WORLD);
  //bypass_downhill_check = bypass;
  MPI_Bcast(&obj,1,MPI_FLOAT,0,MPI_COMM_WORLD);
  //MPI_Finalize();
  //exit(0);
#endif
  return obj;
}

// this function will do transpose a random access file into the target data matrix, subject to masks

//void param_split_t::load_random_access_data(random_access_t *   random_access, float * & mat, int in_variables, int in_observations,int out_observations, int out_variables, bool * observations_mask, bool * variables_mask){
//  
//#ifdef USE_MPI
//  if(out_observations==0||out_variables==0) return;
//  bool read_vec = out_observations==1?false:true;
//  mat = new float[out_observations*out_variables];
//  int out_variable = 0;
//  for(int i=0;i<in_variables;++i){
//    if (variables_mask[i]){
//      float row_buff[in_observations];
//      if(read_vec) random_access->extract_vec(i,in_observations,row_buff);
//      //cerr<<"Out variable "<<out_variable<<" done\n";
//      int out_observation = 0;
//      for(int j=0;j<in_observations;++j){
//        if (observations_mask[j]){
//          //if(slave_id==0) cerr<<" at "<<i<<","<<j<<endl;
//          float val = read_vec?row_buff[j]:random_access->extract_val(i,j);
//          mat[out_observation*out_variables+out_variable] = val;
//          ++out_observation;
//        }
//      }
//      ++out_variable;
//    }
//  }
//#endif
//}

//void param_split_t::load_matrix_data(const char *  mat_file,float * & mat,int input_rows, int input_cols,int output_rows, int output_cols, bool * row_mask, bool * col_mask,bool file_req, float defaultVal){
//#ifdef USE_MPI
//  ofs_debug<<"Loading matrix data with input dim "<<input_rows<<" by "<<input_cols<<" and output dim "<<output_rows<<" by "<<output_cols<<endl;
//  ofs_debug.flush();
//  if(output_rows==0||output_cols==0) return;
//  mat = new float[output_rows*output_cols];
//
//  ifstream ifs(mat_file);
//  if (!ifs.is_open()){
//    ofs_debug<<"Cannot open file "<<mat_file<<endl;
//    if (file_req){
//      ofs_debug<<"File "<<mat_file<<" is required. Program will exit now.\n";
//      MPI_Finalize();
//      exit(1);
//    }else{
//      ofs_debug<<"File is optional.  Will default values to "<<defaultVal<<endl;
//      for(int i=0;i<output_rows*output_cols;++i) mat[i] = defaultVal;
//      return;
//    }
//  }
//  string line;
//  int output_row = 0;
//  for(int i=0;i<input_rows;++i){
//    getline(ifs,line);
//    if (row_mask[i]){
//      istringstream iss(line);
//      int output_col = 0;
//      for(int j=0;j<input_cols;++j){
//        float val;
//        iss>>val;
//        if (col_mask[j]){
//           if (output_row>=output_rows || output_col>=output_cols) {
//             ofs_debug<<mpi_rank<<": Assigning element at "<<output_row<<" by "<<output_col<<endl;
//             ofs_debug<<mpi_rank<<": Out of bounds\n";
//             MPI_Finalize();
//             exit(1);
//           }
//           mat[output_row*output_cols+output_col] = val;
//           ++output_col;
//        }
//      }
//      ++output_row;
//    }
//  }
//  ofs_debug<<"MPI rank "<<mpi_rank<<" successfully read in "<<mat_file<<endl;
//  ifs.close();
//#endif
//}

int param_split_t::get_qn_parameter_length(){
  int len = 0;
  if(mpi_rank==0){
    len = this->observations + this->variables;
  }
  return len;
}

void param_split_t::get_qn_current_param(float * params){
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

void param_split_t::store_qn_current_param(float * params){
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
  //update_constrained_beta();
#endif
}


int main_param_split(int argc,char * argv[]){
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

