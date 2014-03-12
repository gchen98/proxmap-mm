#include<iostream>
#include<fstream>
#include<sstream>
#include<cstdlib>
#include<math.h>
using namespace std;

struct config_t{
  config_t();
  float rho_min; // proximal distance parameter
  float rho_scale_fast; // proximal distance parameter
  float rho_scale_slow; // proximal distance parameter
  float rho_max;
  float epsilon_max;
  float epsilon_scale_fast; // proximal distance parameter
  float epsilon_scale_slow; // proximal distance parameter
  float epsilon_min;
  float rho_distance_ratio;

  float mu_min;  // penalty parameter
  float mu_increment;  // penalty parameter
  float mu_max;  // penalty parameter

  bool use_gpu;
  int platform_id;
  int device_id;
  string kernel_base;
  string genofile;
  // convex cluster settings
  string early_weightsfile;
  string late_weightsfile;
  int datapoints;
  int variables;
  string geno_format;
  string geno_order;
  // regression settings
  string traitfile;
  float nu;  // penalty parameter
};


class proxmap_t{
public:
  proxmap_t();
  void init(string configfile);
  void run();
  virtual ~proxmap_t();
protected:
  // utility functions
  virtual void allocate_memory(string configfile);
  virtual void parse_config_line(string & key, istringstream & iss);
  // proximal distance functions
  virtual void initialize(float mu)=0;
  virtual float get_map_distance()= 0;
  virtual void update_map_distance()= 0;
  virtual bool in_feasible_region()=0;
  virtual float evaluate_obj()= 0;
  virtual void print_output()=0;
  virtual void iterate()=0;
  virtual float infer_rho()=0;

  float get_prox_dist_penalty();
  int colcount(const char * filename);
  int linecount(const char * filename);
  void load_into_matrix(const char * filename,float * & mat,int rows, int cols);
  void mmultiply(float *  a,int a_rows, int a_cols, float *  b,int b_cols, float *  c);


  config_t * config;
  float rho,epsilon,rho_distance_ratio;
private:
};
