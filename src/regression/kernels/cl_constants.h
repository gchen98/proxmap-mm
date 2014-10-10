// the following were defined for ATI firestream 9270, which 
// is actually a pathetic device, as it doesn't support block barriers
//#define SMALL_BLOCK_WIDTH 64
//#define BLOCK_WIDTH 256
// the following can be used for the nVidia Fermi
#define GRID_WIDTH 524288
#define SMALL_BLOCK_WIDTH 32
#define BLOCK_WIDTH 512
#define MISSING 120573

typedef struct{
  char geno[4];
} packedgeno_t;

#define MAPPING  __local float mapping[4]; mapping[0]=0; mapping[1]=9; mapping[2]=1; mapping[3]=2;


