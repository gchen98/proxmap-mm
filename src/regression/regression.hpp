
using namespace std;

class regression_t:public proxmap_t{
public:
  regression_t();
  ~regression_t();
private:
  void allocate_memory(string configfile);
  void parse_config_line(string & key, istringstream & iss);
  void iterate();
  void finalize_iteration();
  float get_map_distance();
  void update_map_distance();
  float evaluate_obj();
  bool in_feasible_region();
  void print_output();
  void initialize(float mu);
  float infer_rho();

  int n,p;
  int blocks;
  float w;
  float maxr,maxc;
  float * nu;
  float * beta_project;
  float * beta;
  float * theta_project;
  float * theta;
  float * lambda;
  float L;
  float * X;
  float * XT;
  float * XXT;
  float * XXTI;
  float * y;
  float * Xbeta;
  
  void update_Xbeta();
  void update_XXT();
  void project_beta();
  void project_theta();
  void update_theta();
  void update_beta();
  void update_lambda();
  void loss();
  void check_constraints();
};

