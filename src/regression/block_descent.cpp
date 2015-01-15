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
#include"block_descent.hpp"

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

block_descent_t::block_descent_t(bool single_run){
  this->single_run = single_run;
  this->total_iterations = 0;
  //cerr<<"Single run initialized\n";
}

block_descent_t::~block_descent_t(){
#ifdef USE_MPI
  if(this->single_run){
    MPI::Finalize();
  } 

  if(slave_id>=0){
    delete plink_data_X_subset;
    delete plink_data_X_subset_subjectmajor;
#ifdef USE_GPU
    if(this->run_gpu) delete ocl_wrapper;
#endif
  }else{
    delete random_access_geno;
  }

  delete[] mask_p;
  //delete[] temp_p;
  delete[]active_indices;
  delete[]inactive_indices;
  //delete [] cg_x;
  //delete [] cg_b;
  //delete [] bd_all_y;
  //delete [] bdinv_v;
  delete [] residuals;
  delete [] y;
  delete [] Xbeta_full;
  delete[] means;
  delete[] precisions;
  delete [] snp_node_sizes;
  delete [] snp_node_offsets;
  delete [] subject_node_sizes;
  delete [] subject_node_offsets;
  delete [] gradient;
  delete [] beta_increment;
  delete [] beta;
  delete [] last_beta;
  delete [] constrained_beta;
  if(config->debug_mpi){
    ofs_debug.close();
  }
#endif
}

void block_descent_t::update_Xbeta(){
#ifdef USE_MPI
  bool *  & mask = mask_p;
  for(int j=0;j<variables;++j) mask[j]=1;
  update_Xbeta(mask);
  //delete[]mask;
#endif
}

void block_descent_t::update_Xbeta(bool * mask){
#ifdef USE_MPI
  compute_x_times_vector(beta,mask,Xbeta_full, false);
#endif
}

// in_vec is of dimension variables
// out_vec is of dimension observations

void block_descent_t::compute_x_times_vector(float * invec,bool * mask,float * outvec,bool debug){
#ifdef USE_MPI
  bool debug_gpu = (run_gpu && run_cpu && slave_id==0);
  float temp_vec[observations];
  if(slave_id>=0){
    if(run_gpu){
  #ifdef USE_GPU
      ocl_wrapper->write_to_buffer("beta",variables,beta);
      ocl_wrapper->run_kernel("update_xbeta",BLOCK_WIDTH*snp_chunks,observations,1,BLOCK_WIDTH,1,1);
      ocl_wrapper->run_kernel("reduce_xbeta_chunks",BLOCK_WIDTH,observations,1,BLOCK_WIDTH,1,1);
      ocl_wrapper->read_from_buffer("Xbeta_full",observations,Xbeta_full);
      if(debug_gpu){
        float Xbeta_temp[observations];
        ocl_wrapper->read_from_buffer("Xbeta_full",observations,Xbeta_temp);
        cerr<<"debug gpu update_Xbeta GPU:";
        for(int i=0;i<observations;++i){
          if(i>(observations-10))cerr<<" "<<i<<":"<<Xbeta_temp[i];
        }
        cerr<<endl;
      }
#endif
    } // if run gpu
    if(run_cpu){
      if(debug) cerr<<"compute x times vec slave: "<<slave_id<<", observation:";
      //for(int i=0;i<100;++i){
      if(debug_gpu) cerr<<"debug gpu x times vec CPU:";
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
              if(mask[var_index]){
                float g=subset_geno[threadindex]==9?0:(subset_geno[threadindex]-means[var_index])*precisions[var_index];
                if(debug)cerr<<" "<<var_index<<":"<<g<<","<<invec[var_index];
                xb+= g * invec[var_index];
              }else{
                if(debug)cerr<<"Slave "<<slave_id<<" at "<<i<<", skipping var index: "<<var_index<<endl;
              }
            }
          }
        }
        temp_vec[i] = xb;
        
        if(debug)cerr<<" "<<i<<":"<<temp_vec[i];
        if(debug_gpu && i>(observations-10))cerr<<" "<<i<<":"<<temp_vec[i];
      }
      if (debug_gpu) cerr<<endl;
      if(debug) cerr<<endl;
    } // if run_cpu
  }else{
    for(int i=0;i<observations;++i) temp_vec[i] = 0;
  }
  MPI_Reduce(temp_vec,outvec,observations,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Bcast(outvec,observations,MPI_FLOAT,0,MPI_COMM_WORLD);
#endif
}

void block_descent_t::update_beta_block_descent(){
#ifdef USE_MPI
  float  Xz[observations];
  float  cg_x0[variables];
  float  cg_x[variables];
  float  bd_all_y0[variables];
  float  bd_all_y[variables];
  float  XtXz[variables];
  int max_bdinv = 50;
  float tolerance = 1e-5;
  for(int j=0;j<variables;++j){
    cg_x0[j] = cg_x[j] = 0;
    bd_all_y0[j] = bd_all_y[j] = 0;
  }
  float last_norm = 1e10;
  for(int bdinv_iter=0;bdinv_iter<max_bdinv;++bdinv_iter){
    // handle active set first
    compute_x_times_vector(cg_x,inactive_indices,Xz,false);
    compute_xt_times_vector(Xz,active_indices,XtXz, gradient_weight);
    MPI_Gatherv(XtXz,variables,MPI_FLOAT,XtXz,snp_node_sizes,snp_node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
    if(mpi_rank==0){
      for(int j=0;j<variables;++j){
        if(active_indices[j]){
          //cerr<<"A2z "<<j<<": "<<XtXz[j]<<endl;
        }
      }
    }
    if(mpi_rank==0){
      cerr<<"Total active: "<<total_active<<endl;
      // time to construct A1
      float temp_y[total_active];
      float bd_y[total_active];
      int y_mapping[total_active];
      int k=0;
      for(int j=0;j<variables;++j){
        if(active_indices[j]){
          temp_y[k] = beta_increment[j] - XtXz[j];
          y_mapping[k] = j;
          ++k;
        }
      }
      for(int k=0;k<total_active;++k){
        //cerr<<"y "<<k<<": "<<temp_y[k]<<endl;
      }
      float A1[total_active*total_active];
      float A1_inv[total_active*total_active];
      
      //cerr<<"Active indices:";
      for(int j3=0;j3<variables;++j3){
        //if(active_indices[j3]) cerr<<" "<<j3;
      }
      //cerr<<endl;
      int row = 0;
      for(int j1=0;j1<variables;++j1){
        if(active_indices[j1]){
          int col = row;
          for(int j2=j1;j2<variables;++j2){
            if(active_indices[j2]){
              matrix_indices_t query(j1,j2);
              cor_matrix_cache_t::iterator it = cor_matrix_cache.find(query);
              float dot_prod=0;
              if(it==cor_matrix_cache.end()){
                float vec1[observations];
                float vec2[observations];
                //if(total_active==20)cerr<<"Looking up "<<j1<<endl;
                random_access_geno->extract_vec(j1,observations,vec1);
                //if(total_active==20)cerr<<"Looking up "<<j2<<endl;
                random_access_geno->extract_vec(j2,observations,vec2);
                //cerr<<"Doing dot prod:\n";
                for(int i=0;i<observations;++i){
                  //cerr<<"j1: "<<j1<<" j2: "<<j2<<" i: "<<i<<" vec1: "<<vec1[i]<<" vec2: "<<vec2[i]<<endl;
                  dot_prod+=vec1[i]*vec2[i];
                }
                cor_matrix_cache[query] = dot_prod;
              }else{
                //cerr<<" Found cache at query "<<j1<<","<<j2<<endl;
                dot_prod = it->second;
              }
              dot_prod=dot_prod*gradient_weight + (j1==j2);
              //cerr<<"A At row "<<row<<" col "<<col<<endl;
              //cerr<<"B At row "<<j1<<" col "<<j2<<endl;
              A1[row*total_active+col] = dot_prod;
              A1[col*total_active+row] = dot_prod;
              //A1[j1*total_active+j2] = dot_prod;
              //A1[j2*total_active+j1] = dot_prod;
              ++col;
            }
          }
          ++row;
        }
      }
      for(int k1=0;k1<total_active;++k1){
        for(int k2=0;k2<total_active;++k2){
          //cerr<<" "<<A1[k1*total_active+k2];
        }
        //cerr<<endl;
      }
      invert(A1,A1_inv,total_active,total_active);
      mmultiply(A1_inv,total_active,total_active,temp_y,1,bd_y);
      for(int k=0;k<total_active;++k){
        bd_all_y[y_mapping[k]] = bd_y[k];
        //cerr<<"y "<<k<<": "<<bd_all_y[y_mapping[k]]<<endl;
      }
    }
    MPI_Scatterv(bd_all_y,snp_node_sizes,snp_node_offsets,MPI_FLOAT,bd_all_y,variables,MPI_FLOAT,0,MPI_COMM_WORLD);
    float  Xy[observations];
    float  A3x[variables];
    compute_x_times_vector(bd_all_y,active_indices,Xy, false);
    compute_xt_times_vector(Xy,inactive_indices,A3x, gradient_weight);
    MPI_Gatherv(A3x,variables,MPI_FLOAT,A3x,snp_node_sizes,snp_node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
    for(int j=0;j<variables;++j){
      if(inactive_indices[j]){
        A3x[j] = beta_increment[j] - A3x[j];
        //if(j<30) cerr<<"A2y "<<j<<":"<<A3x[j]<<endl;
      }
    }
    conjugate_gradient(A3x,cg_x);
    float the_norm = 0;
    for(int j=0;j<variables;++j){
      if(active_indices[j]){
        float dev = bd_all_y[j]-bd_all_y0[j];
        the_norm+=dev*dev;
      }else{
        float dev = cg_x[j]-cg_x0[j];
        the_norm+=dev*dev;
      }
    }
    int converged = 0;
    if(mpi_rank==0){
      the_norm = sqrt(the_norm);
      cerr<<"Block descent norm: "<<the_norm<<", last norm: "<<last_norm<<endl;
      converged = the_norm< tolerance || the_norm>last_norm;
      last_norm = the_norm;
    }
    MPI_Bcast(&converged,1,MPI_INT,0,MPI_COMM_WORLD);
    if(converged) break;
    for(int j=0;j<variables;++j){
      if(active_indices[j]){
        bd_all_y0[j] = bd_all_y[j];
      }else{
        cg_x0[j] = cg_x[j];
      }
    }
  }
  for(int j=0;j<variables;++j){
    if(active_indices[j]){
      beta_increment[j] = bd_all_y[j];
    }else{
      beta_increment[j] = cg_x[j];
    }
    beta[j]-=beta_increment[j];
    //if ((mpi_rank<2 && (j<10 ||j==50 )) || (j== 1500 || (mpi_rank==2 && j==500))) cerr<<"rank "<<mpi_rank<<" j "<<j<<" beta_increment "<<beta_increment[j]<<endl; 
    //if(mpi_rank==0 && j<50) cerr<<"J "<<j<<" beta: "<<beta[j]<<endl;
  }
#endif
}

void block_descent_t::conjugate_gradient(float * b, float *x){
  //cerr<<"Slave "<<slave_id<<" variables: "<<variables<<endl;
  float Xx[observations];
  float residuals[variables];
  float conjugate_vec[variables];
  float r2_old = 0;
  float tol = 1e-10;
  int max_step=100;
  compute_x_times_vector(x,inactive_indices,Xx, slave_id==-10);
  compute_xt_times_vector(Xx,inactive_indices,residuals, -gradient_weight);
  MPI_Gatherv(residuals,variables,MPI_FLOAT,residuals,snp_node_sizes,snp_node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
  for(int j=0;j<variables;++j){
    if(inactive_indices[j]){
      residuals[j]-=x[j]*(1+dist_func);
      residuals[j]+=b[j];
      conjugate_vec[j] = residuals[j];
      r2_old+=residuals[j]*residuals[j];
      //if(mpi_rank==0 && j<40) cerr<<"CG: "<<j<<": x[j]: "<<b[j]<<" A3x: "<<residuals[j]<<endl;
    }else{
      conjugate_vec[j] = 0;
    }
  }
  //if(mpi_rank==0) cerr<<"r2_old: "<<r2_old<<endl;
  float current_norm=1e10;
  float last_norm = current_norm;
  for(int i=0;i<max_step;++i){
    float Xp[observations];
    float Ap[variables];
    compute_x_times_vector(conjugate_vec,inactive_indices,Xp, slave_id==-10);
    compute_xt_times_vector(Xp,inactive_indices,Ap, +gradient_weight);
    MPI_Gatherv(Ap,variables,MPI_FLOAT,Ap,snp_node_sizes,snp_node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
    float pAp = 0;
    for(int j=0;j<variables;++j){
      if(inactive_indices[j]){
        Ap[j]+=conjugate_vec[j]*(1+dist_func);
        //if(j<40) cerr<<"CG: "<<j<<": rank: "<<mpi_rank<<" p: "<<conjugate_vec[j]<<endl;
        pAp+=conjugate_vec[j]*Ap[j];
        //if(mpi_rank==0 && j<40) cerr<<"CG: "<<j<<": Ap: "<<Ap[j]<<endl;
      }
    }
    float alpha = r2_old / pAp;
    float r2_new = 0;
    for(int j=0;j<variables;++j){
      if(inactive_indices[j]){
        x[j] = x[j] + alpha * conjugate_vec[j];
        residuals[j] = residuals[j] - alpha * Ap[j];
        r2_new+=residuals[j]*residuals[j];
      }
    }
    int converged = 0;
    if(mpi_rank==0){
      current_norm = sqrt(r2_new);
      cerr<<"CG norm: "<<current_norm<<endl;
      converged = (current_norm< tol || current_norm> last_norm);
      last_norm = current_norm;
    }
    MPI_Bcast(&converged,1,MPI_INT,0,MPI_COMM_WORLD);
    if(converged)break;
    for(int j=0;j<variables;++j){
      if(inactive_indices[j]){
        conjugate_vec[j] = residuals[j] + (r2_new / r2_old)*conjugate_vec[j];
        //if (mpi_rank==0 && j<40) cerr<<"CG: "<<j<<" p: "<<conjugate_vec[j]<<endl;
      }
    }
    r2_old = r2_new;
  }
  //delete[]A3x;
  //delete[]Xx;
}


void block_descent_t::update_beta_landweber(){
#ifdef USE_MPI
  //bool run_cpu = false;
  //bool run_gpu = true;
  //bool debug = false;
  // Begin code for Landweber update
  float inverse_lipschitz = 2./this->spectral_norm;
  cerr<<"LIPSCHITZ CONSTANT: "<<this->spectral_norm<<endl;
  float norm_diff = 0;
  int iter=0,maxiter = static_cast<int>(config->max_landweber);
  int converged = 0;
  float tolerance = 1e-8;
  float new_beta[variables];
  while(!converged && iter<maxiter){
    update_Xbeta();
    compute_xt_times_vector(Xbeta_full,XtXbeta);
    if (slave_id>=0){
      //bool testgpu2 = (run_cpu && slave_id==-10);
      //if(run_gpu){
      //} // if run gpu
      //if(run_cpu){
      for(int j=0;j<variables;++j){ 
        new_beta[j] = beta[j] - inverse_lipschitz*(XtXbeta[j]  - XtY[j]);
        //new_beta[j] = beta[j] - inverse_lipschitz*(XtXbeta[j] + dist_func*(beta[j]-constrained_beta[j]) - XtY[j]);
        if(j<5) cerr<<"J:"<<j<<" XtXbeta:"<<XtXbeta[j]<<" XtY:"<<XtY[j]<<" beta: "<<new_beta[j]<<endl;
      }
      //}
    }
    MPI_Gatherv(new_beta,variables,MPI_FLOAT,new_beta,snp_node_sizes,snp_node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
    if(mpi_rank==0){
      norm_diff = 0;
      for(int j=0;j<variables;++j){
        norm_diff+=(new_beta[j]-beta[j])*(new_beta[j]-beta[j]);
        beta[j] = new_beta[j];
      }
      norm_diff=sqrt(norm_diff);
      converged = (norm_diff<tolerance);
      ++iter;
      if(config->verbose)cerr<<".";
    }
    cerr<<"L2 norm at landweber: "<<norm_diff<<endl;
    MPI_Scatterv(beta,snp_node_sizes,snp_node_offsets,MPI_FLOAT,beta,variables,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(&converged,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&iter,1,MPI_INT,0,MPI_COMM_WORLD);
    //MPI_Bcast(beta,variables,MPI_FLOAT,0,MPI_COMM_WORLD);
  }
  if(mpi_rank==0){
    if(config->verbose)cerr<<"\nLandweber iterations: "<<iter<<" norm diff: "<<norm_diff<<endl;
  } 
#endif
}

void block_descent_t::update_beta_CG(){
#ifdef USE_MPI
  //bool run_cpu = false;
  //bool run_gpu = true;
  bool debug = false;
  float conjugate_vec[variables];
  float residual[variables];
  float X_CV[observations]; 
  update_Xbeta();
  compute_xt_times_vector(Xbeta_full,XtXbeta);
  //if (slave_id>=0) cerr<<"Dist func: "<<dist_func<<endl;
  for(int j=0;j<variables;++j){
    residual[j] = (XtY[j]+dist_func*constrained_beta[j]) - 
                  (XtXbeta[j]+dist_func*beta[j]);
    conjugate_vec[j] = residual[j];
    //if(slave_id>=0 && j<10) cerr<<snp_node_offsets[mpi_rank]+j<<": residual: "<<residual[j]<<endl;
  }
  MPI_Gatherv(residual,variables,MPI_FLOAT,residual,snp_node_sizes,snp_node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
  MPI_Gatherv(conjugate_vec,variables,MPI_FLOAT,conjugate_vec,snp_node_sizes,snp_node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
  float XtX_CV[variables];
  float norm_diff = 0;
  int iter=0,maxiter = static_cast<int>(config->max_landweber);
  int converged = 0;
  float tolerance = 1e-8;
  float new_beta[variables];
  while(!converged && iter<maxiter){
  // Begin code to update X_CV
    if(slave_id>=0){
      bool debug_gpu = (run_cpu && slave_id==-10);
      if(run_gpu){
  #ifdef USE_GPU
        ocl_wrapper->write_to_buffer("beta",variables,beta);
        ocl_wrapper->run_kernel("update_xbeta",BLOCK_WIDTH*snp_chunks,observations,1,BLOCK_WIDTH,1,1);
        ocl_wrapper->run_kernel("reduce_xbeta_chunks",BLOCK_WIDTH,observations,1,BLOCK_WIDTH,1,1);
        ocl_wrapper->read_from_buffer("Xbeta_full",observations,Xbeta_full);
        if(debug_gpu){
          float Xbeta_temp[observations];
          ocl_wrapper->read_from_buffer("Xbeta_full",observations,Xbeta_temp);
          cerr<<"debug gpu update_lambda GPU:";
          for(int i=0;i<observations;++i){
            if(i>(observations-10))cerr<<" "<<i<<":"<<Xbeta_temp[i];
          }
          cerr<<endl;
        }
  #endif
      } // if run gpu
      if(run_cpu){
        //for(int i=0;i<100;++i){
        for(int i=0;i<observations;++i){
          float xc = 0;
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
                xc+=g * conjugate_vec[var_index];
              }
            }
          }
          X_CV[i] = xc;
        }
        if(debug) cerr<<endl;
        if(debug_gpu) cerr<<endl;
      } // if run_cpu
    }else{
      for(int i=0;i<observations;++i) X_CV[i] = 0;
    }
    float xcv_reduce[observations];
    MPI_Reduce(X_CV,xcv_reduce,observations,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
    if(mpi_rank==0){
      for(int i=0;i<observations;++i) X_CV[i] = xcv_reduce[i];
    }
// End code to update X_CV and X_beta
    compute_xt_times_vector(X_CV,XtX_CV);
    MPI_Gatherv(XtX_CV,variables,MPI_FLOAT,XtX_CV,snp_node_sizes,snp_node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
    if(mpi_rank==0){
      float cxxc = 0;
      float sq_residual = 0;
      for(int j=0;j<variables;++j){
        sq_residual += residual[j]*residual[j];
        cxxc += conjugate_vec[j] * XtX_CV[j];
        float Alpha = sq_residual / cxxc;
        if(j<10) cerr<<snp_node_offsets[mpi_rank]+j<<": XtX_CV: "<<XtX_CV[j]<<" alpha: "<<Alpha<<" sq_res: "<<sq_residual<<" cxxc: "<<cxxc<<endl;
        new_beta[j] = beta[j] + Alpha * conjugate_vec[j];
        if(j<10) cerr<<snp_node_offsets[mpi_rank]+j<<": new_beta: "<<new_beta[j]<<endl;
        residual[j] = residual[j] - Alpha * XtX_CV[j];
      }
      float sq_residual2 = 0;
      for(int j=0;j<variables;++j){
        sq_residual2 += residual[j]*residual[j];
      }
      float Beta = sq_residual2/sq_residual;
      for(int j=0;j<variables;++j){
        conjugate_vec[j] += residual[j]*Beta*conjugate_vec[j];
      }
      norm_diff = 0;
      for(int j=0;j<variables;++j){
        norm_diff+=(new_beta[j]-beta[j])*(new_beta[j]-beta[j]);
        beta[j] = new_beta[j];
      }
      norm_diff=sqrt(norm_diff);
      converged = (norm_diff<tolerance);
      ++iter;
      if(config->verbose)cerr<<".";
    }
    MPI_Bcast(&converged,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&iter,1,MPI_INT,0,MPI_COMM_WORLD);
  }
#endif
}


void block_descent_t::compute_xt_times_vector(float * in_vec, float * out_vec){
  //cerr<<"At compute_xt_times_vec 1\n";
  bool * & mask = mask_p;
  for(int j=0;j<variables;++j) mask[j] = true;
  compute_xt_times_vector(in_vec, mask, out_vec,1);
}

// out_vec is of dimension variables
// in_vec is of dimension observations
void block_descent_t::compute_xt_times_vector(float * in_vec, bool * mask,float * out_vec, float scaler){
  //bool debug = true;
  //bool run_cpu = false; 
  //bool run_gpu = true;
  if(slave_id>=0){
    //cerr<<"At compute_xt_times_vec 2 with "<<variables<<" variables.\n";
    bool debug_gpu = (run_gpu && run_cpu && slave_id==0);
    if(run_gpu ){
#ifdef USE_GPU
      ocl_wrapper->write_to_buffer("invec",observations,in_vec);
      ocl_wrapper->run_kernel("compute_xt_times_vector",BLOCK_WIDTH*subject_chunks,variables,1,BLOCK_WIDTH,1,1);
      ocl_wrapper->run_kernel("reduce_xt_vec_chunks",BLOCK_WIDTH,variables,1,BLOCK_WIDTH,1,1);
      ocl_wrapper->read_from_buffer("Xt_vec",variables,out_vec);
      if(debug_gpu){
        //float Xt_y_chunks[variables*subject_chunks];
        float Xt_temp[variables];
        //ocl_wrapper->read_from_buffer("Xt_y_chunks",variables*subject_chunks,Xt_y_chunks);
        ocl_wrapper->read_from_buffer("Xt_vec",variables,Xt_temp);
        cerr<<"debuggpu compute_xt_times_vector GPU:";
        for(int j=0;j<variables;++j){
          //float xt_y = 0;
          for(int i=0;i<subject_chunks;++i){
            //xt_y+=Xt_y_chunks[j*subject_chunks+i];
          }
          if(j>(variables-10))cerr<<" "<<j<<":"<<Xt_temp[j];
          //if(j<10)cerr<<" "<<j<<":"<<xt_y;
        }
        cerr<<endl;
      }
#endif
    }  // if run gpu
    if(run_cpu){
      //if(debug) cerr<<"PROJECT_BETA slave "<<slave_id<<" variable:";
      //bool feasible = in_feasible_region();
      if(debug_gpu) cerr<<"debuggpu compute_xt_times_vector CPU:";
      for(int j=0;j<variables;++j){
        out_vec[j] = 0;
        if(mask[j]){
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
                out_vec[j]+= g * in_vec[obs_index];
                //if(slave_id==0 && obs_index<100)cerr<<" "<<obs_index<<":"<<g<<":"<<y[obs_index];
              }
            }
          }
          //if(slave_id==0)cerr<<endl;
          // gold standard
          //for(int i=0;i<observations;++i){
            //float g = 0;
            //float g = plink_data_X_subset->get_geno(j,i);
            //xt_y+= g * y[i];
          //}
          //if(j<10) cerr<<" "<<j<<":"<<out_vec[j];
          //if(debug_gpu && j>(variables-10))cerr<<" "<<j<<":"<<out_vec[j];
          //if(slave_id==0)cerr<<endl;
        }else{ // if mask enabled
          //cerr<<"Mask disabled at "<<j<<endl;
        } // if mask disnabled
        out_vec[j]*=scaler;
      } // loop over variables
      if(debug_gpu) cerr<<endl;
    } // if run cpu
  } // if is slave
}

void block_descent_t::update_map_distance(){
#ifdef USE_MPI
  //MPI_Gatherv(beta,variables,MPI_FLOAT,beta,snp_node_sizes,snp_node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
  if(mpi_rank==0){
    float constraint_dev=0;
    for(int j=0;j<all_variables;++j){
      float dev = constrained_beta[j]==0?beta[j]:0;
      constraint_dev+=dev*dev;
    }
    this->map_distance = sqrt(constraint_dev);
    this->dist_func = rho/this->map_distance;
    if(this->dist_func>1e20) this->dist_func = 1e10;
    cerr<<"UPDATE_MAP_DISTANCE: rho: "<<rho<<" dist: "<<this->map_distance<<" dist_func: "<<dist_func<<endl;
  }
  MPI_Bcast(&this->map_distance,1,MPI_FLOAT,0,MPI_COMM_WORLD);
  MPI_Bcast(&this->dist_func,1,MPI_FLOAT,0,MPI_COMM_WORLD);
#endif
}

float block_descent_t::get_map_distance(){
  return this->map_distance;
}



void block_descent_t::update_constrained_beta(){
  if(mpi_rank==0)cerr<<"Updating constrained beta\n";
  // prepare to broadcast full beta vector to slaves
  this->total_active = 0;
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
        //active_indices[b.index] = true;
        //cerr<<"Index: "<<b.index<<" "<<b.val<<endl;
        //inactive_indices[b.index] = false;
        //++total_active;
      }else{
        constrained_beta[b.index] = 0;
        //inactive_indices[b.index] = true;
        //active_indices[b.index] = false;
      }
      ++j;
    }
    //cor_matrix_cache.clear();
  }
  MPI_Scatterv(constrained_beta,snp_node_sizes,snp_node_offsets,MPI_FLOAT,constrained_beta,variables,MPI_FLOAT,0,MPI_COMM_WORLD);
  for(int j=0;j<variables;++j){
    active_indices[j] = constrained_beta[j]!=0;
    inactive_indices[j] = !active_indices[j];
    total_active+=active_indices[j];
  }
  if(mpi_rank==0)cerr<<"Active set: "<<total_active<<endl;

  //MPI_Scatterv(active_indices,snp_node_sizes,snp_node_offsets,MPI_INT,active_indices,variables,MPI_INT,0,MPI_COMM_WORLD);
  //MPI_Scatterv(inactive_indices,snp_node_sizes,snp_node_offsets,MPI_INT,inactive_indices,variables,MPI_INT,0,MPI_COMM_WORLD);
}



bool block_descent_t::in_feasible_region(){
  float mapdist = get_map_distance();
  if(config->verbose)cerr<<"IN_FEASIBLE_REGION: mapdist: "<<mapdist<<" threshold: "<<this->current_mapdist_threshold<<endl;
  bool ret= (mapdist>=0 && mapdist< this->current_mapdist_threshold);
  return ret;
}

void block_descent_t::parse_config_line(string & token,istringstream & iss){
  proxmap_t::parse_config_line(token,iss);
  if (token.compare("FAM_FILE")==0){
    iss>>config->fam_file;
  }else if (token.compare("SNP_BED_FILE")==0){
    iss>>config->snp_bed_file;
  }else if (token.compare("BIN_GENO_FILE")==0){
    iss>>config->bin_geno_file;
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
  }else if (token.compare("BETA_EPSILON")==0){
    iss>>config->beta_epsilon;
  }else if (token.compare("DEBUG_MPI")==0){
    iss>>config->debug_mpi;
  }
}


void block_descent_t::read_dataset(){
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
  //bool subject_mask[observations];

  for(int i=0;i<observations;++i) full_subject_mask[i] = true;
  //for(int i=0;i<observations;++i) subject_mask[i] = slave_id<0?true:false;
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
      //subject_mask[observation_indices_vec[slave_id][i]] = true;
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
  this->y = new float[observations];
  this->residuals = new float[observations];
  parse_fam_file(famfile,full_subject_mask,observations,this->y);
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
  }
  if(config->verbose)cerr<<"Read input done\n";
#endif
}

void block_descent_t::parse_fam_file(const char * infile, bool * mask,int len, float * newy){
  int maskcount = 0;
  if(config->verbose)cerr<<"Allocating Y of len "<<len<<endl;
  for(int i=0;i<observations;++i){
    maskcount+=mask[i];
  }
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
void block_descent_t::parse_bim_file(const char * infile, bool * mask,int len, float * means, float * precisions){
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

void block_descent_t::init_gpu(){
#ifdef USE_GPU
  // init GPU
  int subject_chunk_clusters = subject_chunks/BLOCK_WIDTH+(subject_chunks%BLOCK_WIDTH!=0);
  int snp_chunk_clusters = snp_chunks/BLOCK_WIDTH+(snp_chunks%BLOCK_WIDTH!=0);
  int platform_id = 0;
  int device_id = slave_id;
  vector<string> sources;
  sources.push_back("cl_constants.h");
  sources.push_back("packedgeno.c");
  sources.push_back("block_descent.c");
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
  ocl_wrapper->create_kernel("update_xbeta");
  ocl_wrapper->create_kernel("reduce_xbeta_chunks");
  ocl_wrapper->create_kernel("compute_xt_times_vector");
  ocl_wrapper->create_kernel("reduce_xt_vec_chunks");
  // create buffers
  ocl_wrapper->create_buffer<packedgeno_t>("packedgeno_snpmajor",CL_MEM_READ_ONLY,variables*packedstride_snpmajor);
  ocl_wrapper->create_buffer<packedgeno_t>("packedgeno_subjectmajor",CL_MEM_READ_ONLY,observations*packedstride_subjectmajor);
  ocl_wrapper->create_buffer<float>("means",CL_MEM_READ_ONLY,variables);
  ocl_wrapper->create_buffer<float>("precisions",CL_MEM_READ_ONLY,variables);
  ocl_wrapper->create_buffer<float>("beta",CL_MEM_READ_ONLY,variables);
  ocl_wrapper->create_buffer<float>("beta_project",CL_MEM_READ_WRITE,variables);
  ocl_wrapper->create_buffer<float>("invec",CL_MEM_READ_WRITE,observations);
  ocl_wrapper->create_buffer<float>("Xbeta_full",CL_MEM_READ_WRITE,observations);
  ocl_wrapper->create_buffer<float>("Xbeta_chunks",CL_MEM_READ_WRITE,observations * snp_chunks);
  ocl_wrapper->create_buffer<float>("lambda",CL_MEM_READ_ONLY,observations);
  ocl_wrapper->create_buffer<float>("Xt_vec_chunks",CL_MEM_READ_WRITE,variables * subject_chunks);
  ocl_wrapper->create_buffer<float>("Xt_vec",CL_MEM_READ_WRITE,variables);
  // initialize buffers
  ocl_wrapper->write_to_buffer("packedgeno_snpmajor",variables*packedstride_snpmajor,packedgeno_snpmajor);
  ocl_wrapper->write_to_buffer("packedgeno_subjectmajor",observations*packedstride_subjectmajor,packedgeno_subjectmajor);
  ocl_wrapper->write_to_buffer("means",variables,means);
  ocl_wrapper->write_to_buffer("precisions",variables,precisions);
  // add Kernel arguments
  ocl_wrapper->add_kernel_arg("update_xbeta",observations);
  ocl_wrapper->add_kernel_arg("update_xbeta",variables);
  ocl_wrapper->add_kernel_arg("update_xbeta",snp_chunks);
  ocl_wrapper->add_kernel_arg("update_xbeta",packedstride_subjectmajor);
  ocl_wrapper->add_kernel_arg("update_xbeta",*(ocl_wrapper->get_buffer("packedgeno_subjectmajor")));
  ocl_wrapper->add_kernel_arg("update_xbeta",*(ocl_wrapper->get_buffer("Xbeta_chunks")));
  ocl_wrapper->add_kernel_arg("update_xbeta",*(ocl_wrapper->get_buffer("beta")));
  ocl_wrapper->add_kernel_arg("update_xbeta",*(ocl_wrapper->get_buffer("means")));
  ocl_wrapper->add_kernel_arg("update_xbeta",*(ocl_wrapper->get_buffer("precisions")));
  ocl_wrapper->add_kernel_arg("update_xbeta",cl::__local(sizeof(packedgeno_t) * SMALL_BLOCK_WIDTH));
  ocl_wrapper->add_kernel_arg("update_xbeta",cl::__local(sizeof(float) * BLOCK_WIDTH));
  ocl_wrapper->add_kernel_arg("reduce_xbeta_chunks",variables);
  ocl_wrapper->add_kernel_arg("reduce_xbeta_chunks",snp_chunks);
  ocl_wrapper->add_kernel_arg("reduce_xbeta_chunks",snp_chunk_clusters);
  ocl_wrapper->add_kernel_arg("reduce_xbeta_chunks",*(ocl_wrapper->get_buffer("Xbeta_chunks")));
  ocl_wrapper->add_kernel_arg("reduce_xbeta_chunks",*(ocl_wrapper->get_buffer("Xbeta_full")));
  ocl_wrapper->add_kernel_arg("reduce_xbeta_chunks",cl::__local(sizeof(float) * BLOCK_WIDTH));
  ocl_wrapper->add_kernel_arg("compute_xt_times_vector",observations);
  ocl_wrapper->add_kernel_arg("compute_xt_times_vector",variables);
  ocl_wrapper->add_kernel_arg("compute_xt_times_vector",subject_chunks);
  ocl_wrapper->add_kernel_arg("compute_xt_times_vector",packedstride_snpmajor);
  ocl_wrapper->add_kernel_arg("compute_xt_times_vector",*(ocl_wrapper->get_buffer("packedgeno_snpmajor")));
  ocl_wrapper->add_kernel_arg("compute_xt_times_vector",*(ocl_wrapper->get_buffer("Xt_vec_chunks")));
  ocl_wrapper->add_kernel_arg("compute_xt_times_vector",*(ocl_wrapper->get_buffer("invec")));
  ocl_wrapper->add_kernel_arg("compute_xt_times_vector",*(ocl_wrapper->get_buffer("means")));
  ocl_wrapper->add_kernel_arg("compute_xt_times_vector",*(ocl_wrapper->get_buffer("precisions")));
  ocl_wrapper->add_kernel_arg("compute_xt_times_vector",cl::__local(sizeof(packedgeno_t) * SMALL_BLOCK_WIDTH));
  ocl_wrapper->add_kernel_arg("compute_xt_times_vector",cl::__local(sizeof(float) * BLOCK_WIDTH));
  ocl_wrapper->add_kernel_arg("reduce_xt_vec_chunks",observations);
  ocl_wrapper->add_kernel_arg("reduce_xt_vec_chunks",subject_chunks);
  ocl_wrapper->add_kernel_arg("reduce_xt_vec_chunks",subject_chunk_clusters);
  ocl_wrapper->add_kernel_arg("reduce_xt_vec_chunks",*(ocl_wrapper->get_buffer("Xt_vec_chunks")));
  ocl_wrapper->add_kernel_arg("reduce_xt_vec_chunks",*(ocl_wrapper->get_buffer("Xt_vec")));
  ocl_wrapper->add_kernel_arg("reduce_xt_vec_chunks",cl::__local(sizeof(float) * BLOCK_WIDTH));
#endif
}

void block_descent_t::init(string config_file){
#ifdef USE_MPI
  if(this->single_run){
    MPI::Init();
  }
  this->mpi_numtasks = MPI::COMM_WORLD.Get_size();
  this->slaves = mpi_numtasks-1;
  this->mpi_rank = MPI::COMM_WORLD.Get_rank();
  this->slave_id = mpi_rank-1;
  config->beta_epsilon = 1e-3;
#endif
  proxmap_t::init(config_file);
  //if (this->slave_id>=0)  config->verbose = false;
  if(config->verbose) cerr<<"Configuration initialized\n";
}

void block_descent_t::allocate_memory(){
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

  // ALLOCATE MEMORY FOR ESTIMATED PARAMETERS
  active_indices = new bool[variables];
  inactive_indices = new bool[variables];
  this->last_residual = 1e10;
  this->residual = 0;
  //this->temp_n = new float[observations];
  //this->temp_p = new float[variables];
  //this->cg_x = new float[this->variables];
  //this->cg_b = new float[this->variables];
  //this->bdinv_v = new float[this->variables];
  //this->bd_all_y = new float[this->variables];
  //this->mask_n = new int[this->observations];
  this->mask_p = new bool[this->variables];
  this->beta = new float[this->variables];
  this->gradient = new float[this->variables];
  this->beta_increment = new float[this->variables];
  this->XtXbeta = new float[this->variables];
  this->XtY = new float[this->variables];
  this->last_beta = new float[this->variables];
  this->constrained_beta = new float[this->variables];
  this->Xbeta_full = new float[observations];
  for(int i=0;i<observations;++i) Xbeta_full[i] = 0;
  this->BLOCK_WIDTH = plink_data_t::PLINK_BLOCK_WIDTH;

  this->subject_chunks = observations/BLOCK_WIDTH+(observations%BLOCK_WIDTH!=0);
  this->snp_chunks = variables/BLOCK_WIDTH+(variables%BLOCK_WIDTH!=0);
  // initializations
  this->spectral_norm = config->spectral_norm;
  this->last_mapdist = 1e10;
  this->current_mapdist_threshold = config->mapdist_threshold;
  this->map_distance = 0;
  this->current_top_k = config->top_k_max;
  this->last_BIC = 1e10;
  proxmap_t::allocate_memory();
  if(mpi_rank==0){
    random_access_geno = new random_access_t(config->bin_geno_file.data(),variables,observations);
  }
  if(this->run_gpu && slave_id>=0){
#ifdef USE_GPU
    init_gpu();
#endif
  }
  compute_xt_times_vector(y,XtY);
  //MPI_Finalize();exit(1);
#endif
}

float block_descent_t::infer_epsilon(){
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

float block_descent_t::infer_rho(){
  // just return one value for this algorithm
  return this->spectral_norm;
  float new_rho = last_rho;
#ifdef USE_MPI
  if(mu==0) return config->rho_min;
  if (mpi_rank==0){
    ofs_debug<<"INFER_RHO: Inner iterate: "<<iter_rho_epsilon<<endl;
    if(iter_rho_epsilon==0){
      if (in_feasible_region()  ) { // the projections are feasible
        //track_residual = true;
        //cerr<<"INFER_RHO: enabling residual tracking\n";
        //if(mapdist_stalled) this->current_mapdist_threshold = this->map_distance;
        //cerr<<"INFER_RHO: Map distance threshold revised to "<<this->current_mapdist_threshold<<"\n";
      }
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
    }
    //}
    //new_rho = scaler * this->slaves * sqrt(this->map_distance+epsilon);
    //if (config->verbose) cerr <<"INFER_RHO: mu: "<<this->mu<<" new rho: "<<new_rho<<endl;
  }
  MPI_Bcast(&new_rho,1,MPI_FLOAT,0,MPI_COMM_WORLD);
#endif
  return new_rho;
}

float block_descent_t::compute_marginal_beta(float * xvec){
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

void block_descent_t::initialize(){
  if (mpi_rank==0){
    if(config->verbose) cerr<<"Mu iterate: "<<iter_mu<<" mu="<<mu<<" of "<<config->mu_max<<endl;
  }
  if(this->iter_rho_epsilon==0){
    if(mu==0){
      //this->rho = config->rho_min;
      //for(int j=0;j<this->variables;++j) beta[j]=0;
      //update_beta_landweber();
      float temp_norm = 0;
      if(slave_id>=0){
        for(int j=0;j<this->variables;++j){
          float genovec[observations];
          for(int i=0;i<observations;++i){
            genovec[i] =  plink_data_X_subset->get_geno(j,i);
            //cerr<<"Geno is "<<genovec[i]<<endl;
            temp_norm+=genovec[i]*genovec[i];
          }
          //beta[j] = 0;
          beta[j] = compute_marginal_beta(genovec);
          //if(slave_id==0 && j<30) cerr<<"Beta MLE at "<<j<<": "<<beta[j]<<endl;
        }
      }
      MPI_Reduce(&temp_norm,&frobenius_norm,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
      
      if(mpi_rank==0){
        frobenius_norm=sqrt(frobenius_norm);
        cerr<<"Frobenius norm: "<<frobenius_norm<<", spectral norm: "<<config->spectral_norm<<endl;
        this->rho = this->spectral_norm;
        //this->rho = this->frobenius_norm;
      }
      MPI_Bcast(&this->rho,1,MPI_FLOAT,0,MPI_COMM_WORLD);
      MPI_Gatherv(beta,variables,MPI_FLOAT,beta,snp_node_sizes,snp_node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
      for(int j=0;j<this->variables;++j) last_beta[j] = beta[j]; 
    }
    update_constrained_beta();
    update_map_distance();
    evaluate_obj();
    //MPI_Finalize();
    //exit(0);
  }
}

bool block_descent_t::finalize_inner_iteration(){
  return true;
}

bool block_descent_t::finalize_iteration(){
  int proceed = false; 
#ifdef USE_MPI
  if(mpi_rank==0){
    //cerr<<"In finalize iteration\n";
    float diff_norm = 0;
    for(int j=0;j<variables;++j){
      if(constrained_beta[j]!=0){
        float dev = beta[j]-last_beta[j];
        diff_norm+=dev*dev;
      }
      last_beta[j] = beta[j];
    }
    diff_norm = sqrt(diff_norm);
    bool top_k_finalized = false;
    bool abort = false;
    if(rho > config->rho_max || (this->last_mapdist>0 && this->map_distance> last_mapdist)){ 
      cerr<<"FINALIZE_ITERATION: Failed to meet constraint. Last distance: "<<last_mapdist<<" current: "<<map_distance<<".\n";
      top_k_finalized = true;
    }
    if(current_BIC > last_BIC){
      cerr<<"FINALIZE_ITERATION: Warning: BIC grew from "<<last_BIC<<" to "<<current_BIC<<".\n";
      //top_k_finalized = true;
      //abort = true;
    }
    cerr<<"FINALIZE_ITERATION: feasible "<<in_feasible_region()<<" diff_norm "<<diff_norm<<".\n";
    if(in_feasible_region()  && diff_norm<config->beta_epsilon){
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
  

void block_descent_t::iterate(){
  update_beta_block_descent();
  //update_beta_landweber();
  //update_beta_CG();
  update_constrained_beta();
  update_map_distance();
  ++total_iterations;
  //MPI_Finalize();
  //exit(0);
}

void block_descent_t::print_output(){
  if(mpi_rank==0 ){
  //if(mpi_rank==0 && active_set_size<=this->current_top_k){
    cerr<<"Mu: "<<mu<<" rho: "<<rho<<" epsilon: "<<epsilon<<" total iterations: "<<this->total_iterations<<" mapdist: "<<this->map_distance<<endl;
    cerr<<"INDEX\tBETA(of "<<this->current_top_k<<")\n";
    //for(int j=0;j<10;++j){
    for(int j=0;j<variables;++j){
      if (constrained_beta[j]!=0){
        cerr<<j<<"\t"<<beta[j]<<endl;
      }
    }
    ofs_debug<<"Done!\n";
  }
}


bool block_descent_t::proceed_qn_commit(){
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

float block_descent_t::evaluate_obj(){
  float obj=0;
#ifdef USE_MPI
  if(in_feasible_region())update_Xbeta(active_indices);
  else update_Xbeta();
  if (mpi_rank==0){
    last_residual = residual;
    residual = 0;
    for(int i=0;i<observations;++i){
      residuals[i]=(y[i]-Xbeta_full[i]);
      //residuals[i]+=(Xbeta_full[i]-y[i])*(Xbeta_full[i]-y[i]);
      //if(i<10) cerr<<"RESIDUAL: "<<residuals[i]<<"\t"<<Xbeta_full[i]<<"\t"<<y[i]<<endl;
      residual+=residuals[i]*residuals[i];
    }
    float loss = sqrt(residual+1.);
    gradient_weight = 1./loss;
    //cerr<<"Gradient weight: "<<gradient_weight<<" and rho is "<<rho<<endl;
    //float proxdist = get_prox_dist_penalty(); 
    obj = loss + rho * this->map_distance;
    current_BIC = log(observations) * this->current_top_k + observations * log(residual/observations);
    cerr<<"EVALUATE_OBJ: norm1 "<<residual<<" DIST: "<<map_distance<<" FULL: "<<obj<<" BIC: "<<current_BIC<<endl;
  }
  MPI_Bcast(&gradient_weight,1,MPI_FLOAT,0,MPI_COMM_WORLD);
  MPI_Bcast(&obj,1,MPI_FLOAT,0,MPI_COMM_WORLD);
  MPI_Bcast(&residual,1,MPI_FLOAT,0,MPI_COMM_WORLD);
  MPI_Bcast(residuals,observations,MPI_FLOAT,0,MPI_COMM_WORLD);
  compute_xt_times_vector(residuals,gradient);
  MPI_Gatherv(gradient,variables,MPI_FLOAT,gradient,snp_node_sizes,snp_node_offsets,MPI_FLOAT,0,MPI_COMM_WORLD);
  for(int j=0;j<variables;++j){
    gradient[j]*=-gradient_weight; 
    //if(j<10) cerr<<"GRADIENT: "<<gradient[j]<<endl;
    beta_increment[j] = gradient[j] + dist_func * (beta[j]-constrained_beta[j]);
  }
#endif
  return obj;
}

int block_descent_t::get_qn_parameter_length(){
  int len = 0;
  if(mpi_rank==0){
    len = this->variables;
  }
  return len;
}

void block_descent_t::get_qn_current_param(float * params){
#ifdef USE_MPI
  if(mpi_rank==0){
    int k=0;
    for(int i=0;i<observations;++i){
      //params[k++] = theta[i];
    }
    for(int i=0;i<variables;++i){
      params[k++] = beta[i];
    }
  }
#endif
}

void block_descent_t::store_qn_current_param(float * params){
#ifdef USE_MPI
  if(mpi_rank==0){
    int k = 0;
    for(int i=0;i<observations;++i){
      //theta[i] = params[k++];
    }
    for(int i=0;i<variables;++i){
       beta[i] = params[k++];
    }
  }
  MPI_Scatterv(beta,snp_node_sizes,snp_node_offsets,MPI_FLOAT,beta,variables,MPI_FLOAT,0,MPI_COMM_WORLD);
#endif
}


int main_block_descent(int argc,char * argv[]){
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

