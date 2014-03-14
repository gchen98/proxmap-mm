#include"../cl_constants.h"
#include"../proxmap.hpp"
#include"../cl_templates.hpp"
#include"cluster.hpp"

void cluster_t::init_opencl(){
  if(run_gpu){
  // initialize the GPU if necessary
#ifdef USE_GPU
    debug_opencl = true;
    proxmap_t::init_opencl();
    cerr<<"Initializing OpenCL for cluster sub class\n";
    this->variable_blocks = (p/BLOCK_WIDTH + (p%BLOCK_WIDTH!=0));
    cerr<<"P is "<<p<<", Workgroup width is "<<variable_blocks<<endl;
    // CREATE KERNELS
    createKernel("init_U",kernel_init_U);
    createKernel("update_map_distance",kernel_update_map_distance);
    //createKernel("simple",kernel_simple);
    cerr<<"Kernels created\n";
    // CREATE BUFFERS
    createBuffer<float>(CL_MEM_READ_WRITE,n*p,"buffer_U",buffer_U);
    createBuffer<float>(CL_MEM_READ_WRITE,n*p,"buffer_U_project",buffer_U_project);
    createBuffer<float>(CL_MEM_READ_WRITE,n*p,"buffer_U_project_orig",buffer_U_project_orig);
    createBuffer<float>(CL_MEM_READ_WRITE,n*p,"buffer_U_project_prev",buffer_U_project_orig);
    createBuffer<float>(CL_MEM_READ_WRITE,triangle_dim,"buffer_V_project_coeff",buffer_V_project_coeff);
    createBuffer<float>(CL_MEM_READ_ONLY,n*p,"buffer_rawdata",buffer_rawdata);
    createBuffer<float>(CL_MEM_READ_ONLY,triangle_dim,"buffer_weights",buffer_weights);
    createBuffer<int>(CL_MEM_READ_ONLY,n,"buffer_offsets",buffer_offsets);
    createBuffer<float>(CL_MEM_READ_WRITE,variable_blocks,"buffer_variable_block_norms1",buffer_variable_block_norms1);
    createBuffer<float>(CL_MEM_READ_WRITE,variable_blocks,"buffer_variable_block_norms2",buffer_variable_block_norms2);
    ////createBuffer<>(CL_MEM_READ_ONLY,,"buffer_",buffer_);
    ////createBuffer<>(CL_MEM_READ_ONLY,,"buffer_",buffer_);
    ////createBuffer<>(CL_MEM_READ_ONLY,,"buffer_",buffer_);
    ////createBuffer<>(CL_MEM_READ_ONLY,,"buffer_",buffer_);
    cerr<<"GPU Buffers created\n";
    // initialize anything here
    ////writeToBuffer(buffer_haploid_arr,g_people,haploid_arr,"buffer_haploid_arr");
    writeToBuffer(buffer_U,n*p,U,"buffer_U");
    writeToBuffer(buffer_U_project,n*p,U_project,"buffer_U_project");
    writeToBuffer(buffer_U_project_orig,n*p,U_project_orig,"buffer_U_project_orig");
    //writeToBuffer(buffer_U_project_prev,n*p,U_project_prev,"buffer_U_project_prev");
    //writeToBuffer(buffer_V_project_coeff,triangle_dim,V_project_coeff,"buffer_V_project_coeff");
    writeToBuffer(buffer_rawdata,n*p,rawdata,"buffer_rawdata");
    writeToBuffer(buffer_weights,triangle_dim,weights,"buffer_weights");
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
    setArg(kernel_update_map_distance,arg,n,"kernel_update_map_distance");
    setArg(kernel_update_map_distance,arg,p,"kernel_update_map_distance");
    setArg(kernel_update_map_distance,arg,*buffer_U,"kernel_update_map_distance");
    setArg(kernel_update_map_distance,arg,*buffer_U_project,"kernel_update_map_distance");
    setArg(kernel_update_map_distance,arg,*buffer_variable_block_norms1,"kernel_update_map_distance");
    setArg(kernel_update_map_distance,arg,*buffer_variable_block_norms2,"kernel_update_map_distance");
    setArg(kernel_update_map_distance,arg,cl::__local(sizeof(float)*BLOCK_WIDTH),"kernel_update_map_distance");
    setArg(kernel_update_map_distance,arg,cl::__local(sizeof(float)*BLOCK_WIDTH),"kernel_update_map_distance");
    //
    //setArg(kernel_reduce_weights2,arg,g_people,"kernel_reduce_weights2");
    //setArg(kernel_store_U_project,arg,p,"kernel_store_U_project");
    //setArg(kernel_store_U_project,arg,*buffer_U_project,"kernel_store_U_project");
    //setArg(kernel_store_U_project,arg,*buffer_U_project_orig,"kernel_store_U_project");
    //setArg(kernel_reduce_weights2,arg,g_people,"kernel_reduce_weights2");
    //kernelWorkGroupSize = kernel_reduce_weights2->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(devices[0], &err);
    //clSafe(err,"get workgroup size kernel reduce_weights2");
    //cerr<<"reduce_weights2 kernel work group size is "<<kernelWorkGroupSize<<endl;
    cerr<<"GPU kernel arguments assigned.\n";
#endif
  }
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
  cerr<<"GPU Norm1 was "<<norm1<<" and norm2 was "<<norm2<<endl;
  float norm = norm1+norm2;
  this->map_distance = norm;
  this->dist_func = sqrt(this->map_distance+epsilon);
  cerr<<"GET_MAP_DISTANCE: New map distance is "<<norm<<" with U distance="<<norm1<<", V distance="<<norm2<<" dist_func: "<<dist_func<<endl;
#endif
}

void cluster_t::initialize_gpu(float mu){
#ifdef USE_GPU
  int x_dim = BLOCK_WIDTH * variable_blocks;
  runKernel("init_U",kernel_init_U,x_dim,n,1,BLOCK_WIDTH,1,1);
  bool debug_gpu = true;
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

//void MendelGPU::free_opencl(){
//  delete program;
//  delete commandQueue;
//  delete context;
//}
//
//void MendelGPU::init_window_opencl(){
//  if (run_gpu){
//    #ifdef USE_GPU
//    //writeToBuffer(buffer_active_haplotype, g_max_haplotypes, g_active_haplotype,"buffer_active_haplotype");
//    //writeToBuffer(buffer_haplotypes, 1,&g_haplotypes, "buffer_haplotypes" );
//    //writeToBuffer(buffer_markers, 1, &g_markers, "buffer_markers" );
//    //writeToBuffer(buffer_left_marker, 1, &g_left_marker, "buffer_left_marker");
//    //writeToBuffer(buffer_haplotype, g_max_window*g_max_haplotypes, g_haplotype, "buffer_haplotype");
//    //writeToBuffer(buffer_frequency, g_max_haplotypes, g_frequency, "buffer_frequency");
//    
//    #endif
//    cerr<<"Buffers sent to GPU for current window\n";
//  }
//}
//
//void MendelGPU::init_iteration_buffers_opencl(){
//#ifdef USE_GPU
//  //writeToBuffer(buffer_frequency, g_max_haplotypes, g_frequency, "buffer_frequency");
//#endif
//  cerr<<"Iteration Buffers sent to GPU\n";
//}
