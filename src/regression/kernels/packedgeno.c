inline float c2g(char c,int shifts, __local float * mapping){
  float geno = 0;
  int val = ((int)c)>>(2*shifts) & 3;
  return mapping[val];
  switch (val){
    case 0:
      geno = 0;
      break;
    case 2:
      geno = 1;
      break;
    case 3:
      geno = 2;
      break;
    case 1:
      geno = 9;
      break;
    default:
      geno=9;
      break;
  }
  return geno;
}

inline void convertgeno(int genoindex,
int threadindex,int chunk,int packedstride,
__global const packedgeno_t * packedgeno_matrix,
__local packedgeno_t * geno,
__local float * subset_geno,
__local float * mapping
){
  geno[threadindex] = packedgeno_matrix[genoindex*packedstride+chunk*SMALL_BLOCK_WIDTH+threadindex];
  int t = 0;
  for(int b=0;b<4;++b){
    for(int c=0;c<4;++c){
      //subset_geno[threadindex * 16 + t] = 1;
      subset_geno[threadindex * 16 + t] = c2g(geno[threadindex].geno[b],c,mapping);
      ++t;
    }
  }
}

