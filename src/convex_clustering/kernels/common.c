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

__kernel void update_U(
__const int p,
__constant float * dist_func,
__constant float * rho,
__global float * U,
__global float * rawdata,
__global float * U_project
){
  int var_chunk = get_group_id(0);
  int person = get_group_id(1);
  int threadindex = get_local_id(0);
  if (var_chunk*BLOCK_WIDTH+threadindex<p){
    U[person*p+var_chunk*BLOCK_WIDTH+threadindex] = 
    dist_func[0]/(dist_func[0]+rho[0])*
    rawdata[person*p+var_chunk*BLOCK_WIDTH+threadindex] +
    rho[0]/(dist_func[0]+rho[0])*
    U_project[person*p+var_chunk*BLOCK_WIDTH+threadindex];
  }
}


__kernel void store_U_project(
__const int p,
__global float * U,
__global float * U_project,
__global float * U_project_orig
){
  int var_chunk = get_group_id(0);
  int person = get_group_id(1);
  int threadindex = get_local_id(0);
  if (var_chunk*BLOCK_WIDTH+threadindex<p){
    U_project_orig[person*p+var_chunk*BLOCK_WIDTH+threadindex] =
    U_project[person*p+var_chunk*BLOCK_WIDTH+threadindex] ;
    U_project[person*p+var_chunk*BLOCK_WIDTH+threadindex] = 
    U[person*p+var_chunk*BLOCK_WIDTH+threadindex];
  }
}

__kernel void store_U_project_prev(
__const int p,
__global float * U_project,
__global float * U_project_prev
){
  int var_chunk = get_group_id(0);
  int person = get_group_id(1);
  int threadindex = get_local_id(0);
  if (var_chunk*BLOCK_WIDTH+threadindex<p){
    U_project_prev[person*p+var_chunk*BLOCK_WIDTH+threadindex] =
    U_project[person*p+var_chunk*BLOCK_WIDTH+threadindex];
  }
}

__kernel void iterate_projection(
__const int n,
__const int p,
__const int variable_blocks,
__global float * U,
__global float * U_project,
__global float * U_project_orig,
__global float * U_project_prev,
__global int * offsets,
__global float * weights,
__global float * V_project_coeff,
__global float * subject_variable_block_norms,
__local float * local_left_summation,
__local float * local_right_summation,
__local float * local_all_summation,
__local float * local_temp_norm
){
  int person = get_group_id(1);
  int threadindex = get_local_id(0);
  int var = get_group_id(0)*BLOCK_WIDTH+threadindex;
  int left_neighbors = 0,right_neighbors = 0,total_neighbors = 0;
  local_left_summation[threadindex] = 0;
  local_right_summation[threadindex] = 0;
  local_all_summation[threadindex] = 0;
  local_temp_norm[threadindex] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int neighbor = 0;neighbor<n;++neighbor){
    if(person!=neighbor){
      int offset_index = person<neighbor?offsets[person]+neighbor-person:
      offsets[neighbor]+person-neighbor;
      float weight = weights[offset_index];
      if(weight>0){
        if(neighbor<person){
          ++left_neighbors;
          float scaler = V_project_coeff[offset_index];
          //scaler = .5;
          if (scaler>0 && var<p){
            local_left_summation[threadindex]+=scaler*
            (U_project_orig[neighbor*p+var]-U_project_orig[person*p+var]);
            barrier(CLK_LOCAL_MEM_FENCE);
          }
        }else if(person<neighbor){
          ++right_neighbors;
          float scaler = V_project_coeff[offset_index];
          //scaler = .5;
          if (scaler>0 && var<p){
            local_right_summation[threadindex]+=scaler*
            (U_project_orig[person*p+var] - U_project_orig[neighbor*p+var]);
            barrier(CLK_LOCAL_MEM_FENCE);
          }
        }
        if (var<p){
          local_all_summation[threadindex]+=U_project_prev[neighbor*p+var];
          barrier(CLK_LOCAL_MEM_FENCE);
        }
        ++total_neighbors;
      }
    } // valid neighbor check
  } // loop over neighbors
  if (var<p){
    //float u = (U[person*p+var])/ (1.+total_neighbors);
    //float u = (U[person*p+var] +local_all_summation[threadindex])/ (1.+total_neighbors);
    float u = (U[person*p+var] +local_right_summation[threadindex]- local_left_summation[threadindex]+local_all_summation[threadindex])/ (1.+total_neighbors);
    U_project[person*p+var] = u;
    local_temp_norm[threadindex]+=u*u;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for(int s=BLOCK_WIDTH/2; s>0; s>>=1) {
    if (threadindex<s) {
      local_temp_norm[threadindex]+=local_temp_norm[threadindex+s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(threadindex==0){
    subject_variable_block_norms[person*variable_blocks+get_group_id(0)] = local_temp_norm[0]; 
  }
}

__kernel void evaluate_obj(
__const int n,
__const int p,
__const int variable_blocks,
__global int * offsets,
__global float * rawdata,
__global float * U,
__global float * U_project,
__global float * weights,
__global float * V_project_coeff,
__global float * n_norms,
__global float * n2_norms,
__local float * local_norm1,
__local float * local_norm2
){
  int threadindex = get_local_id(0);
  int i1 = get_group_id(1);
  int i2 = get_group_id(0);
  if (i1>i2) return;
  if (i1==i2){
    local_norm1[threadindex] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);  
    for(int block = 0;block<variable_blocks;++block){
      int j = block*BLOCK_WIDTH+threadindex;
      if(j<p){
        float dev = rawdata[i1*p+j]-U[i1*p+j];
        //dev = rawdata[i1*p+j];
        local_norm1[threadindex]+=dev*dev;
        barrier(CLK_LOCAL_MEM_FENCE);  
      }
    }
    for(int s=BLOCK_WIDTH/2; s>0; s>>=1) {
      if (threadindex<s) {
        local_norm1[threadindex]+=local_norm1[threadindex+s];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(threadindex==0){
      n_norms[i1] = local_norm1[0];
    }
  }else{
    int offset_index = offsets[i1]+i2-i1;
    float weight = weights[offset_index];
    float n2_norm = 0;
    //weight = 1;
    if (weight>0){
      float scaler = V_project_coeff[offset_index];
      //scaler = 1;
      if (scaler>0){
        float pair_norm=0;
        local_norm2[threadindex]=0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int block = 0;block<variable_blocks;++block){
          int j = block*BLOCK_WIDTH+threadindex;
          if(j<p){
            float dev = scaler * (U_project[i1*p+j]-U_project[i2*p+j]);
            //dev = U_project[i1*p+j];
            local_norm2[threadindex]+=dev*dev;
            barrier(CLK_LOCAL_MEM_FENCE);  
          }
        }
        for(int s=BLOCK_WIDTH/2; s>0; s>>=1) {
          if (threadindex<s) {
            local_norm2[threadindex]+=local_norm2[threadindex+s];
          }
          barrier(CLK_LOCAL_MEM_FENCE);
        }
        n2_norm = weight * sqrt(local_norm2[0]);
      }
    }
    if(threadindex==0){
      n2_norms[offset_index] = n2_norm;
    }
  }
  return;
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
}

__kernel void init_v_project_coeff(
__const int n,
__const int p,
__const int variable_blocks,
__constant float * unweighted_lambda,
__global float * weights,
__global float * U_project_orig,
__global float * V_project_coeff,
__global int * offsets,
__local float * local_norm1
){
  int threadindex = get_local_id(0);
  int index1 = get_group_id(1);
  int index2 = get_group_id(0);
  if (index2<=index1) return;
  int offset_index = offsets[index1]+index2-index1;
  float weight = weights[offset_index];
  if (weight == 0){
    if (threadindex == 0){
      V_project_coeff[offset_index] = 1;
    }
  }else{
    local_norm1[threadindex] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int block = 0;block<variable_blocks;++block){
      int j = block*BLOCK_WIDTH+threadindex;
      if(j<p){
        float v = (U_project_orig[index1*p+j]-U_project_orig[index2*p+j]);
        local_norm1[threadindex]+=(v*v);
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }
    for(int s=BLOCK_WIDTH/2; s>0; s>>=1) {
      if (threadindex<s) {
        local_norm1[threadindex]+=local_norm1[threadindex+s];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (threadindex == 0){
      if (local_norm1[0] == 0){
        V_project_coeff[offset_index] = 1;
      }else{
        float l2norm_b = sqrt(local_norm1[0]);
        float lambda = unweighted_lambda[0] * weight;
        if (lambda<l2norm_b){
          V_project_coeff[offset_index] = (1-lambda/l2norm_b);
        }else{
          V_project_coeff[offset_index] = 0;
        }
      }
    }
  }
  return;
}

__kernel void get_U_norm_diff(
__const int n,
__const int p,
__const int variable_blocks,
__global float * U,
__global float * U_prev,
__global float * n_norms,
__local float * local_norm1
){
  int threadindex = get_local_id(0);
  int i = get_group_id(0);
  local_norm1[threadindex] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);  
  for(int block = 0;block<variable_blocks;++block){
    int j = block*BLOCK_WIDTH+threadindex;
    if(j<p){
      float dev = U[i*p+j]-U_prev[i*p+j];
      local_norm1[threadindex]+=dev*dev;
      barrier(CLK_LOCAL_MEM_FENCE);  
      U_prev[i*p+j] = U[i*p+j];
      //barrier(CLK_LOCAL_MEM_FENCE);  
    }
  }
  for(int s=BLOCK_WIDTH/2; s>0; s>>=1) {
    if (threadindex<s) {
      local_norm1[threadindex]+=local_norm1[threadindex+s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(threadindex==0){
    //n_norms[i] = 1;
    n_norms[i] = local_norm1[0];
  }
  return;
}
