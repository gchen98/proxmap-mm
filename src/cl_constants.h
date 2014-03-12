// the following were defined for ATI firestream 9270, which 
// is actually a pathetic device, as it doesn't support block barriers
//#define SMALL_BLOCK_WIDTH 64
//#define BLOCK_WIDTH 256
// the following can be used for the nVidia Fermi
#define SMALL_BLOCK_WIDTH 16
#define BLOCK_WIDTH 256
#define BLOCK_WIDTH_IMPUTE_GUIDE 256
#define LOG_HALF -.6931472

#define TEST_SUBJECT -11 
#define TEST_SUBJECT_STR "-11" 

typedef struct {
  float prob;
  int maternal;
  int paternal;
} bestpair_t; //read/write

typedef struct{
  char octet[4];
}__attribute__((aligned(4))) packedhap_t;

