#include"../cl_constants.h"
#include"../proxmap.hpp"
#include"../cl_templates.hpp"
#include"cluster.hpp"

void cluster_t::init_opencl(){
  if(run_gpu){
  // initialize the GPU if necessary
#ifdef USE_GPU
    debug_opencl = false;
    proxmap_t::init_opencl();
    cerr<<"Initializing OpenCL for cluster sub class\n";
    cerr<<"P is "<<p<<", Workgroup width is "<<variable_blocks<<endl;
    // CREATE KERNELS
    createKernel("init_U",kernel_init_U);
    createKernel("update_U",kernel_update_U);
    createKernel("update_map_distance",kernel_update_map_distance);
    createKernel("init_v_project_coeff",kernel_init_v_project_coeff);
    createKernel("store_U_project",kernel_store_U_project);
    createKernel("store_U_project_prev",kernel_store_U_project_prev);
    createKernel("iterate_projection",kernel_iterate_projection);
    createKernel("evaluate_obj",kernel_evaluate_obj);
    createKernel("get_U_norm_diff",kernel_get_U_norm_diff);
    cerr<<"Kernels created\n";
    // CREATE BUFFERS
    createBuffer<float>(CL_MEM_READ_WRITE,n*p,"buffer_U",buffer_U);
    createBuffer<float>(CL_MEM_READ_WRITE,n*p,"buffer_U_prev",buffer_U_prev);
    createBuffer<float>(CL_MEM_READ_WRITE,n*p,"buffer_U_project",buffer_U_project);
    createBuffer<float>(CL_MEM_READ_WRITE,n*p,"buffer_U_project_orig",buffer_U_project_orig);
    createBuffer<float>(CL_MEM_READ_WRITE,n*p,"buffer_U_project_prev",buffer_U_project_prev);
    createBuffer<float>(CL_MEM_READ_WRITE,triangle_dim,"buffer_V_project_coeff",buffer_V_project_coeff);
    createBuffer<float>(CL_MEM_READ_ONLY,n*p,"buffer_rawdata",buffer_rawdata);
    createBuffer<float>(CL_MEM_READ_ONLY,triangle_dim,"buffer_weights",buffer_weights);
    createBuffer<int>(CL_MEM_READ_ONLY,n,"buffer_offsets",buffer_offsets);
    createBuffer<float>(CL_MEM_READ_WRITE,variable_blocks,"buffer_variable_block_norms1",buffer_variable_block_norms1);
    createBuffer<float>(CL_MEM_READ_WRITE,variable_blocks,"buffer_variable_block_norms2",buffer_variable_block_norms2);
    createBuffer<float>(CL_MEM_READ_WRITE,n*variable_blocks,"buffer_subject_variable_block_norms",buffer_subject_variable_block_norms);
    createBuffer<float>(CL_MEM_READ_ONLY,1,"buffer_unweighted_lambda",buffer_unweighted_lambda);
    createBuffer<float>(CL_MEM_READ_ONLY,1,"buffer_dist_func",buffer_dist_func);
    createBuffer<float>(CL_MEM_READ_ONLY,1,"buffer_rho",buffer_rho);
    createBuffer<float>(CL_MEM_READ_WRITE,n,"buffer_n_norms",buffer_n_norms);
    createBuffer<float>(CL_MEM_READ_WRITE,triangle_dim,"buffer_n2_norms",buffer_n2_norms);
    ////createBuffer<>(CL_MEM_READ_ONLY,,"buffer_",buffer_);
    cerr<<"GPU Buffers created\n";
    // initialize anything here
    writeToBuffer(buffer_U,n*p,U,"buffer_U");
    writeToBuffer(buffer_U_prev,n*p,U_prev,"buffer_U_prev");
    writeToBuffer(buffer_U_project,n*p,U_project,"buffer_U_project");
    writeToBuffer(buffer_U_project_orig,n*p,U_project_orig,"buffer_U_project_orig");
    writeToBuffer(buffer_rawdata,n*p,rawdata,"buffer_rawdata");
    writeToBuffer(buffer_offsets,n,offsets,"buffer_offsets");
    cerr<<"GPU Buffers initialized\n";
    // SET KERNEL ARGUMENTS HERE
    int arg;
    //int kernelWorkGroupSize;
    arg = 0;
    setArg(kernel_init_U,arg,p,"kernel_init_U");
    setArg(kernel_init_U,arg,*buffer_rawdata,"kernel_init_U");
    setArg(kernel_init_U,arg,*buffer_U,"kernel_init_U");
    setArg(kernel_init_U,arg,*buffer_U_project,"kernel_init_U");
    setArg(kernel_init_U,arg,*buffer_U_project_orig,"kernel_init_U");
    arg = 0;
    setArg(kernel_update_U,arg,p,"kernel_update_U");
    setArg(kernel_update_U,arg,*buffer_dist_func,"kernel_update_U");
    setArg(kernel_update_U,arg,*buffer_rho,"kernel_update_U");
    setArg(kernel_update_U,arg,*buffer_U,"kernel_update_U");
    setArg(kernel_update_U,arg,*buffer_U_prev,"kernel_update_U");
    setArg(kernel_update_U,arg,*buffer_rawdata,"kernel_update_U");
    setArg(kernel_update_U,arg,*buffer_U_project,"kernel_update_U");
    arg = 0;
    setArg(kernel_update_map_distance,arg,n,"kernel_update_map_distance");
    setArg(kernel_update_map_distance,arg,p,"kernel_update_map_distance");
    setArg(kernel_update_map_distance,arg,*buffer_U,"kernel_update_map_distance");
    setArg(kernel_update_map_distance,arg,*buffer_U_project,"kernel_update_map_distance");
    setArg(kernel_update_map_distance,arg,*buffer_variable_block_norms1,"kernel_update_map_distance");
    setArg(kernel_update_map_distance,arg,*buffer_variable_block_norms2,"kernel_update_map_distance");
    setArg(kernel_update_map_distance,arg,cl::__local(sizeof(float)*BLOCK_WIDTH),"kernel_update_map_distance");
    setArg(kernel_update_map_distance,arg,cl::__local(sizeof(float)*BLOCK_WIDTH),"kernel_update_map_distance");
    arg = 0;
    setArg(kernel_init_v_project_coeff,arg,n,"kernel_init_v_project_coeff");
    setArg(kernel_init_v_project_coeff,arg,p,"kernel_init_v_project_coeff");
    setArg(kernel_init_v_project_coeff,arg,variable_blocks,"kernel_init_v_project_coeff");
    setArg(kernel_init_v_project_coeff,arg,*buffer_unweighted_lambda,"kernel_init_v_project_coeff");
    setArg(kernel_init_v_project_coeff,arg,*buffer_weights,"kernel_init_v_project_coeff");
    setArg(kernel_init_v_project_coeff,arg,*buffer_U_project_orig,"kernel_init_v_project_coeff");
    setArg(kernel_init_v_project_coeff,arg,*buffer_V_project_coeff,"kernel_init_v_project_coeff");
    setArg(kernel_init_v_project_coeff,arg,*buffer_offsets,"kernel_init_v_project_coeff");
    setArg(kernel_init_v_project_coeff,arg,cl::__local(sizeof(float)*BLOCK_WIDTH),"kernel_init_v_project_coeff");
    arg = 0; 
    setArg(kernel_store_U_project,arg,p,"kernel_store_U_project");
    setArg(kernel_store_U_project,arg,*buffer_U,"kernel_store_U_project");
    setArg(kernel_store_U_project,arg,*buffer_U_project,"kernel_store_U_project");
    setArg(kernel_store_U_project,arg,*buffer_U_project_orig,"kernel_store_U_project");
    arg = 0; 
    setArg(kernel_store_U_project_prev,arg,p,"kernel_store_U_project_prev");
    setArg(kernel_store_U_project_prev,arg,*buffer_U_project,"kernel_store_U_project_prev");
    setArg(kernel_store_U_project_prev,arg,*buffer_U_project_prev,"kernel_store_U_project_prev");
    arg = 0; 
    setArg(kernel_iterate_projection,arg,n,"kernel_iterate_projection");
    setArg(kernel_iterate_projection,arg,p,"kernel_iterate_projection");
    setArg(kernel_iterate_projection,arg,variable_blocks,"kernel_iterate_projection");
    setArg(kernel_iterate_projection,arg,*buffer_U,"kernel_iterate_projection");
    setArg(kernel_iterate_projection,arg,*buffer_U_project,"kernel_iterate_projection");
    setArg(kernel_iterate_projection,arg,*buffer_U_project_orig,"kernel_iterate_projection");
    setArg(kernel_iterate_projection,arg,*buffer_U_project_prev,"kernel_iterate_projection");
    setArg(kernel_iterate_projection,arg,*buffer_offsets,"kernel_iterate_projection");
    setArg(kernel_iterate_projection,arg,*buffer_weights,"kernel_iterate_projection");
    setArg(kernel_iterate_projection,arg,*buffer_V_project_coeff,"kernel_iterate_projection");
    setArg(kernel_iterate_projection,arg,*buffer_subject_variable_block_norms,"kernel_iterate_projection");
    setArg(kernel_iterate_projection,arg,cl::__local(sizeof(float)*BLOCK_WIDTH),"kernel_iterate_projection");
    setArg(kernel_iterate_projection,arg,cl::__local(sizeof(float)*BLOCK_WIDTH),"kernel_iterate_projection");
    setArg(kernel_iterate_projection,arg,cl::__local(sizeof(float)*BLOCK_WIDTH),"kernel_iterate_projection");
    setArg(kernel_iterate_projection,arg,cl::__local(sizeof(float)*BLOCK_WIDTH),"kernel_iterate_projection");
    arg = 0; 
    setArg(kernel_evaluate_obj,arg,n,"kernel_evaluate_obj");
    setArg(kernel_evaluate_obj,arg,p,"kernel_evaluate_obj");
    setArg(kernel_evaluate_obj,arg,variable_blocks,"kernel_evaluate_obj");
    setArg(kernel_evaluate_obj,arg,*buffer_offsets,"kernel_evaluate_obj");
    setArg(kernel_evaluate_obj,arg,*buffer_rawdata,"kernel_evaluate_obj");
    setArg(kernel_evaluate_obj,arg,*buffer_U,"kernel_evaluate_obj");
    setArg(kernel_evaluate_obj,arg,*buffer_U_prev,"kernel_evaluate_obj");
    setArg(kernel_evaluate_obj,arg,*buffer_U_project,"kernel_evaluate_obj");
    setArg(kernel_evaluate_obj,arg,*buffer_weights,"kernel_evaluate_obj");
    setArg(kernel_evaluate_obj,arg,*buffer_V_project_coeff,"kernel_evaluate_obj");
    setArg(kernel_evaluate_obj,arg,*buffer_n_norms,"kernel_evaluate_obj");
    setArg(kernel_evaluate_obj,arg,*buffer_n2_norms,"kernel_evaluate_obj");
    setArg(kernel_evaluate_obj,arg,cl::__local(sizeof(float)*BLOCK_WIDTH),"kernel_evaluate_obj");
    setArg(kernel_evaluate_obj,arg,cl::__local(sizeof(float)*BLOCK_WIDTH),"kernel_evaluate_obj");
    arg = 0; 
    setArg(kernel_get_U_norm_diff,arg,n,"kernel_get_U_norm_diff");
    setArg(kernel_get_U_norm_diff,arg,p,"kernel_get_U_norm_diff");
    setArg(kernel_get_U_norm_diff,arg,variable_blocks,"kernel_get_U_norm_diff");
    setArg(kernel_get_U_norm_diff,arg,*buffer_U,"kernel_get_U_norm_diff");
    setArg(kernel_get_U_norm_diff,arg,*buffer_U_prev,"kernel_get_U_norm_diff");
    setArg(kernel_get_U_norm_diff,arg,*buffer_n_norms,"kernel_get_U_norm_diff");
    setArg(kernel_get_U_norm_diff,arg,cl::__local(sizeof(float)*BLOCK_WIDTH),"kernel_get_U_norm_diff");
    //setArg(kernel_reduce_weights2,arg,g_people,"kernel_reduce_weights2");
    //kernelWorkGroupSize = kernel_reduce_weights2->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(devices[0], &err);
    //clSafe(err,"get workgroup size kernel reduce_weights2");
    //cerr<<"reduce_weights2 kernel work group size is "<<kernelWorkGroupSize<<endl;
    cerr<<"GPU kernel arguments assigned.\n";
#endif
  }
}

void cluster_t::update_weights_gpu(){
#ifdef USE_GPU
    writeToBuffer(buffer_weights,triangle_dim,weights,"buffer_weights");
#endif
}

void cluster_t::init_v_project_coeff_gpu(){
#ifdef USE_GPU
  float unweighted_lambda = mu * dist_func / rho;
  writeToBuffer(buffer_unweighted_lambda, 1, &unweighted_lambda, "buffer_unweighted_lambda");
  runKernel("init_v_project_coeff",kernel_init_v_project_coeff,BLOCK_WIDTH*n,n,1,BLOCK_WIDTH,1,1);
  bool debug_gpu = false;
  if(debug_gpu){
    float * testv = new float[triangle_dim];
    readFromBuffer(buffer_V_project_coeff,triangle_dim,testv,"buffer_V_project_coeff");
    for(int index1=0;index1<n-1;++index1){
      for(int index2=index1+1;index2<n;++index2){
        float & scaler  = testv[offsets[index1]+(index2-index1)];
        if (scaler !=0 && scaler !=1 )
          cerr<<"GPU Init_V Index: "<<index1<<","<<index2<<": "<<scaler<<endl;
      }
    }
  }
#endif
}

void cluster_t::update_map_distance_gpu(){
#ifdef USE_GPU
  int x_dim = BLOCK_WIDTH * variable_blocks;
  runKernel("update_map_distance",kernel_update_map_distance,x_dim,1,1,BLOCK_WIDTH,1,1);
  float norm1_arr[variable_blocks];
  float norm2_arr[variable_blocks];
  readFromBuffer(buffer_variable_block_norms1,variable_blocks,norm1_arr,"buffer_variable_block_norms1");
  readFromBuffer(buffer_variable_block_norms2,variable_blocks,norm2_arr,"buffer_variable_block_norms2");
  float norm1=0,norm2=0;
  for(int i=0;i<variable_blocks;++i){
    //cerr<<"GPU Block "<<i<<" norm: "<<norm1_arr[i]<<","<<norm2_arr[i]<<endl;
    norm1+=norm1_arr[i];
    norm2+=norm2_arr[i];
  }
  if(config->verbose) cerr<<"GPU Norm1 was "<<norm1<<" and norm2 was "<<norm2<<endl;
  float norm = norm1+norm2;
  this->map_distance = norm;
  this->dist_func = sqrt(this->map_distance+epsilon);
  if(config->verbose)cerr<<"GET_MAP_DISTANCE: New map distance is "<<norm<<" with U distance="<<norm1<<", V distance="<<norm2<<" dist_func: "<<dist_func<<endl;
#endif
}

void cluster_t::get_U_gpu(){
#ifdef USE_GPU
  readFromBuffer(buffer_U,n*p,U,"buffer_U");
#endif
}

void cluster_t::initialize_gpu(){
#ifdef USE_GPU
  int x_dim = BLOCK_WIDTH * variable_blocks;
  runKernel("init_U",kernel_init_U,x_dim,n,1,BLOCK_WIDTH,1,1);
  bool debug_gpu = false;
  if(debug_gpu){
    float testArr[n*p];
    readFromBuffer(buffer_U_project,n*p,testArr,"buffer_U_project");
    for(int i=0;i<n;++i){
      for(int j=0;j<p;++j){
        if(i==(n-10) && j>(p-10)){
          cerr<<"GPU: U_project_orig "<<i<<","<<j<<": "<<testArr[i*p+j]<<endl;
        }
      }
    }
  }
#endif
}

void cluster_t::update_projection_gpu(){
#ifdef USE_GPU
      //
      // 1) Run kernel to copy U_project into U_project_prev
      // 2) Run kernel to compute sub_fnorms and store in 
      // buffer_subject_variable_block_norms.
      // 3) Fetch sub_fnorm from GPU
      //
  int x_dim = BLOCK_WIDTH * variable_blocks;
  runKernel("store_U_project_prev",kernel_store_U_project_prev,x_dim,n,1,BLOCK_WIDTH,1,1);
  bool debug_gpu1 = false;
  if(debug_gpu1){
    float testArr[n*p];
    readFromBuffer(buffer_U_project_prev,n*p,testArr,"buffer_U_project_prev");
    for(int i=0;i<n;++i){
      for(int j=0;j<p;++j){
        if(i>(n-2) && j>(p-10)){
          cerr<<"GPU: U_project_prev: "<<i<<","<<j<<": "<<testArr[i*p+j]<<endl;
        }
      }
    }
    for(int i=0;i<n;++i){
      for(int j=0;j<p;++j){
        if(i>(n-2) && j>(p-10)){
          cerr<<"CPU: U_project_prev: "<<i<<","<<j<<": "<<U_project_prev[i*p+j]<<endl;
        }
      }
    }
  }
  // for debugging only
  //writeToBuffer(buffer_V_project_coeff,triangle_dim,V_project_coeff,"buffer_V_project_coeff");
  //writeToBuffer(buffer_U_project_orig,n*p,U_project_orig,"buffer_U_project_orig");
  runKernel("iterate_projection",kernel_iterate_projection,x_dim,n,1,BLOCK_WIDTH,1,1);
  readFromBuffer(buffer_subject_variable_block_norms,n*variable_blocks,sub_fnorm,"buffer_subject_variable_block_norms");
  bool debug_gpu = false;
  if (debug_gpu){
    cerr<<"GPU iterate projection:\n";
    float test_U[n*p];
    readFromBuffer(buffer_U_project,n*p,test_U,"buffer_U_project");
        for(int i=n-5;i<n;++i){
          cerr<<i<<":";
          for(int j=0;j<p;++j){
            cerr<<" "<<test_U[i*p+j];
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
  }
#endif
}

void cluster_t::store_U_projection_gpu(){
#ifdef USE_GPU
  int x_dim = BLOCK_WIDTH * variable_blocks;
  runKernel("store_U_project",kernel_store_U_project,x_dim,n,1,BLOCK_WIDTH,1,1);
  bool debug_gpu = false;
  if(debug_gpu){
    float testArr[n*p];
    readFromBuffer(buffer_U_project,n*p,testArr,"buffer_U_project");
    for(int i=0;i<n;++i){
      for(int j=0;j<p;++j){
        if(i>(n-3) && j>(p-3)){
          cerr<<"GPU store U_project for subject,var: "<<i<<","<<j<<": "<<testArr[i*p+j]<<endl;
        }
      }
    }
  }
#endif
}

void cluster_t::evaluate_obj_gpu(){
#ifdef USE_GPU
  int x_dim = BLOCK_WIDTH * n;
  runKernel("evaluate_obj",kernel_evaluate_obj,x_dim,n,1,BLOCK_WIDTH,1,1);
  readFromBuffer(buffer_n_norms,n,norm1_arr,"buffer_n_norms");
  readFromBuffer(buffer_n2_norms,triangle_dim,norm2_arr,"buffer_n_norms2");
  bool debug_gpu =  false;
  if(debug_gpu){
    cerr<<"GPU NORM1:";
    for(int i=0;i<n;++i){
      cerr<<" "<<norm1_arr[i];
    }
    cerr<<endl;
    for(int i1=0;i1<n-1;++i1){
      cerr<<"GPU NORM2["<<i1<<"]:";
      for(int i2=i1+1;i2<n;++i2){
        if (i2>i1){
          cerr<<" "<<norm2_arr[offsets[i1]+i2-i1];
        }
      }
      cerr<<endl;
    }
  }
#endif
}


void cluster_t::finalize_iteration_gpu(){
#ifdef USE_GPU
  int x_dim = BLOCK_WIDTH * n;
  runKernel("get_U_norm_diff",kernel_get_U_norm_diff,x_dim,1,1,BLOCK_WIDTH,1,1);
  readFromBuffer(buffer_n_norms,n,norm1_arr,"buffer_n_norms");
  float gpu_U_norm_diff = 0;
  for(int i=0;i<n;++i){
    gpu_U_norm_diff+=norm1_arr[i];
  }
  U_norm_diff = sqrt(gpu_U_norm_diff);

  if(config->verbose)cerr<<"FINALIZE_ITERATION: GPU U_norm_diff: "<<U_norm_diff<<endl;
#endif
}

void cluster_t::update_u_gpu(){
#ifdef USE_GPU
  writeToBuffer(buffer_dist_func,1,&dist_func,"buffer_dist_func");
  writeToBuffer(buffer_rho,1,&rho,"buffer_rho");
  int x_dim = BLOCK_WIDTH * variable_blocks;
  runKernel("update_U",kernel_update_U,x_dim,n,1,BLOCK_WIDTH,1,1);
  bool debug_gpu = false;
  if(debug_gpu){
    float testArr[n*p];
    readFromBuffer(buffer_U,n*p,testArr,"buffer_U");
    for(int i=0;i<n;++i){
      for(int j=0;j<p;++j){
        if(i==(n-10) && j>(p-10)){
          cerr<<"update U GPU: U: "<<i<<","<<j<<": "<<testArr[i*p+j]<<endl;
        }
      }
    }
  }
#endif
}

