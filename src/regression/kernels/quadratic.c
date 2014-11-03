__kernel void update_xbeta(
const unsigned int observations,
const unsigned int variables,
const unsigned int snp_chunks,
const unsigned int packedstride_subjectmajor,
__global const packedgeno_t * packedgeno_subjectmajor,
__global float * Xbeta_chunks,
__global const float * beta,
__global const float * means,
__global const float * precisions,
__local packedgeno_t * local_packedgeno,
__local float * local_floatgeno
){
  MAPPING
  int snp_chunk = get_group_id(0);
  int subject = get_group_id(1);
  int threadindex = get_local_id(0);
  local_floatgeno[threadindex] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  int snp_index = snp_chunk*BLOCK_WIDTH+threadindex;
  if (snp_index<variables){
    // LOAD ALL THE COMPRESSED GENOTYPES INTO LOCAL MEMORY
    if (threadindex<SMALL_BLOCK_WIDTH) convertgeno(subject,threadindex,snp_chunk,packedstride_subjectmajor,packedgeno_subjectmajor,local_packedgeno,local_floatgeno,mapping);
    barrier(CLK_LOCAL_MEM_FENCE);
    local_floatgeno[threadindex]= local_floatgeno[threadindex]==9?0:(local_floatgeno[threadindex]-means[snp_index])*precisions[snp_index]*beta[snp_index];
    barrier(CLK_LOCAL_MEM_FENCE);
    // REDUCE 
    for(unsigned int s=BLOCK_WIDTH/2; s>0; s>>=1) {
      if (threadindex < s) {
        local_floatgeno[threadindex] += local_floatgeno[threadindex + s];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(threadindex==0){
      Xbeta_chunks[subject*snp_chunks+snp_chunk] = local_floatgeno[0];
    }
  }
  return;
}

__kernel void reduce_xbeta_chunks(
const unsigned int variables,
const unsigned int snp_chunks,
const unsigned int chunk_clusters,
__global float * Xbeta_chunks,
__global float * Xbeta_full,
__local float * local_xbeta
){
  int subject = get_group_id(1);
  int threadindex = get_local_id(0);
  local_xbeta[threadindex] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int chunk_cluster=0;chunk_cluster<chunk_clusters;++chunk_cluster){
    int snp_chunk = chunk_cluster*BLOCK_WIDTH+threadindex;
    if(snp_chunk<snp_chunks){
      local_xbeta[threadindex] += Xbeta_chunks[subject*snp_chunks+snp_chunk];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  // REDUCE 
  for(unsigned int s=BLOCK_WIDTH/2; s>0; s>>=1) {
    if (threadindex < s) {
      local_xbeta[threadindex] += local_xbeta[threadindex + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(threadindex==0){
    Xbeta_full[subject] = local_xbeta[0];
  }
}

__kernel void compute_xt_times_vector(
const unsigned int observations,
const unsigned int variables,
const unsigned int subject_chunks,
const unsigned int packedstride_snpmajor,
__global const packedgeno_t * packedgeno_snpmajor,
__global float * Xt_vec_chunks,
__global const float * vec,
__global const float * means,
__global const float * precisions,
__local packedgeno_t * local_packedgeno,
__local float * local_floatgeno
){
  MAPPING
  int subject_chunk = get_group_id(0);
  int snp = get_group_id(1);
  float mean = means[snp];
  float precision = precisions[snp];
  int threadindex = get_local_id(0);
  local_floatgeno[threadindex] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  int subject_index = subject_chunk*BLOCK_WIDTH+threadindex;
  if (subject_index<observations){
    // LOAD ALL THE COMPRESSED GENOTYPES INTO LOCAL MEMORY
    if (threadindex<SMALL_BLOCK_WIDTH) convertgeno(snp,threadindex,subject_chunk,packedstride_snpmajor,packedgeno_snpmajor,local_packedgeno,local_floatgeno,mapping);
    barrier(CLK_LOCAL_MEM_FENCE);
    local_floatgeno[threadindex]= local_floatgeno[threadindex]==9?0:(local_floatgeno[threadindex]-mean)*precision*vec[subject_index];
    barrier(CLK_LOCAL_MEM_FENCE);
    // REDUCE 
    for(unsigned int s=BLOCK_WIDTH/2; s>0; s>>=1) {
      if (threadindex < s) {
        local_floatgeno[threadindex] += local_floatgeno[threadindex + s];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(threadindex==0){
      Xt_vec_chunks[snp*subject_chunks+subject_chunk] = local_floatgeno[0];
    }
  }
  return;
}

__kernel void reduce_xt_vec_chunks(
const unsigned int observations,
const unsigned int subject_chunks,
const unsigned int chunk_clusters,
__global float * Xt_vec_chunks,
__global float * Xt_vec,
__local float * local_xt
){
  int snp = get_group_id(1);
  int threadindex = get_local_id(0);
  local_xt[threadindex] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int chunk_cluster=0;chunk_cluster<chunk_clusters;++chunk_cluster){
    int subject_chunk = chunk_cluster*BLOCK_WIDTH+threadindex;
    if(subject_chunk<subject_chunks){
      local_xt[threadindex] += Xt_vec_chunks[snp*subject_chunks+subject_chunk];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  // REDUCE 
  for(unsigned int s=BLOCK_WIDTH/2; s>0; s>>=1) {
    if (threadindex < s) {
      local_xt[threadindex] += local_xt[threadindex + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(threadindex==0){
    Xt_vec[snp] = local_xt[0];
  }
}

