__kernel void init_U(
__const int p,
__global float * rawdata,
__global float * U,
__global float * U_project,
__global float * U_project_orig
){
  int var_chunk = get_group_id(0);
  int person = get_group_id(1);
  int threadindex = get_local_id(0);
  if (var_chunk*BLOCK_WIDTH+threadindex<p){
    U_project_orig[person*p+var_chunk*BLOCK_WIDTH+threadindex] =
    rawdata[person*p+var_chunk*BLOCK_WIDTH+threadindex];
    U_project[person*p+var_chunk*BLOCK_WIDTH+threadindex] =
    rawdata[person*p+var_chunk*BLOCK_WIDTH+threadindex];
    U[person*p+var_chunk*BLOCK_WIDTH+threadindex] = 
    rawdata[person*p+var_chunk*BLOCK_WIDTH+threadindex];
  }
}

__kernel void update_map_distance(
__const int n,
__const int p,
__global float * U,
__global float * U_project,
__global float * variable_block_norms1,
__global float * variable_block_norms2,
__local float * local_norm1,
__local float * local_norm2
){
  int threadindex = get_local_id(0);
  int j = get_group_id(0)*BLOCK_WIDTH+threadindex;
  local_norm1[threadindex] = 0;
  local_norm2[threadindex] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);  
  if (j<p){
    for(int i1=0;i1<n;++i1){
      float diff = U[i1*p+j]-U_project[i1*p+j];
      local_norm1[threadindex]+=(diff*diff);
      barrier(CLK_LOCAL_MEM_FENCE);  
      for(int i2=i1+1;i2<n;++i2){
        diff = (U[i1*p+j]-U[i2*p+j]) - (U_project[i1*p+j]-U_project[i2*p+j]);
        local_norm2[threadindex]+=(diff*diff);
        barrier(CLK_LOCAL_MEM_FENCE);  
      }
    }
  }
  for(int s=BLOCK_WIDTH/2; s>0; s>>=1) {
    if (threadindex<s) {
      local_norm1[threadindex]+=local_norm1[threadindex+s];
      local_norm2[threadindex]+=local_norm2[threadindex+s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (threadindex==0){
    variable_block_norms1[get_group_id(0)] = local_norm1[0];
    variable_block_norms2[get_group_id(0)] = local_norm2[0];
  }
    //U_project[person*p+var_chunk*BLOCK_WIDTH+threadindex] = 
//var_chunk*BLOCK_WIDTH+threadindex;
    //rawdata[person*p+var_chunk*BLOCK_WIDTH+threadindex];
    //U[person*p+var_chunk*BLOCK_WIDTH+threadindex] = 
    //rawdata[person*p+var_chunk*BLOCK_WIDTH+threadindex];
}

__kernel void store_U_project(
__const int p,
__global float * U_project,
__global float * U_project_orig
){
  int var_chunk = get_group_id(0);
  int person = get_group_id(1);
  int threadindex = get_local_id(0);
  U_project_orig[person*p+var_chunk*BLOCK_WIDTH+threadindex] = 
  U_project[person*p+var_chunk*BLOCK_WIDTH+threadindex];
}
