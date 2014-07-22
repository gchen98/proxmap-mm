
class l0_reg_t{
public:
  l0_reg_t(int n,int p, const char * designmat, const char * trait);
  ~l0_reg_t();
  void run(int top_k);
  void output();
private:
  void read_input(int n,int p, const char * filename, float * mat);
  float * X, *Xt, *Y;
  float * x_mm;
  float * b;
  float * A;

  float * x_correction;
  float * v;
  float * x_0;
  float * x_k;
  float * x_mm_dev;
  float * x_mm_dev2;
  float * x_observations;
  float * x_observations_correction;
  float * x_temp_correction;
  float * x_temp;
  float * x_temp_dev;

  float * V;
  float * tV;
  float * d;

  int variables,observations;
  void project_k(int len,float * x,int top_k, float * y);
  float epsilon_decrement(float eps, float norm, float first_decrement, float second_decrement);
  float loss();
};

