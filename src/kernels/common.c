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
