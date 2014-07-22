#include<set>
#include<iostream>
#include<sstream>
#include<fstream>
#include<cstdlib>
#include<math.h>
#include"../proxmap.hpp"
#include"L0_reg.hpp"
using namespace std;

struct beta_t{
  float val;
  int index;
};

struct byVal{
  bool operator()(const beta_t & b1,const beta_t & b2){
    return fabs(b1.val)>fabs(b2.val);
  }
};
typedef set<beta_t,byVal> beta_set_t;

// LINEAR REGRESSION WITH AN L0 PENALTY
// This function solves L0-penalized least squares
//
//   minimize || XB - Y ||_2^2 + w || B ||_0
//   
// by exact penalization and proximal mapping. It uses an MM proximal
// mapping algorithm to solve the problem. To simplify the notation, which 
// uses various forms of X, we rewrite the problem as
//
// minimize 0.5 * x'Ax + b'x + w || x - P(x) ||_2^2
//
// where A = X'X, b = -X'Y, x = B, and P projects onto the k components of x
// with largest magnitude.
//
// Arguments:
//
// -- X is the design matrix.
// -- Y is the response vector.
//
// Coded by Kevin L. Keys (2014)
// klkeys@ucla.edu
// Ported into C++ by Gary K Chen (2014)
// gary.k.chen@usc.edu

// % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %


void l0_reg_t::project_k(int len,float * x,int top_k, float * y){
// % PROJECT ONTO K LARGEST NONZERO ELEMENTS
// %
// % This function projects a vector X onto its K elements of largest
// % magnitude. project_k can handle both integer and double input, but it
// % does not check for repeated values. Consequently, if a tie occurs,
// % project_k will rely on the indices given by descending sort and then
// % truncate at K values.
//n = length(x);
  beta_set_t beta_set;
  for(int i=0;i<len;++i){
    y[i] = 0;
    beta_t b;
    b.val = x[i];
    b.index = i;
    beta_set.insert(b); 
  }
  beta_set_t::iterator it=beta_set.begin();
  for(int k=0;k<top_k;++k){
    y[it->index] = it->val;
    it++;
  }
}

float l0_reg_t::epsilon_decrement(float eps, float norm, float first_decrement, float second_decrement){
  float y=0;
  if(log(eps) < log(norm)){
    //cerr<<"using first decrement of "<<first_decrement<<"eps: "<<eps<<" norm: "<<norm<<endl;
    y = eps / first_decrement;
  }else{
    //cerr<<"using second decrement of "<<second_decrement<<"eps: "<<eps<<" norm: "<<norm<<endl;
    y = eps / second_decrement;
  }
  return y;
}

void l0_reg_t::read_input(int n,int p, const char * filename, float * mat){
  ifstream ifs(filename);
  if(!ifs.is_open()){
    cerr<<"Cannot open "<<filename<<endl;
    exit(1);
  }
  for(int i=0;i<n;++i){
    string line;
    getline(ifs,line);
    istringstream iss(line);
    for(int j=0;j<p;++j){
      iss>>mat[i*p+j];
    }
  }
  ifs.close();
}

float l0_reg_t::loss(){
  float xA[variables];
  proxmap_t::mmultiply(x_mm,1,variables,A,variables,xA);
  float xAx[1];
  proxmap_t::mmultiply(xA,1,variables,x_mm,1,xAx);
  float bx[1];
  proxmap_t::mmultiply(b,1,variables,x_mm,1,bx);
  return .5*xAx[0]+bx[0];
}

l0_reg_t::l0_reg_t(int n,int p, const char * designmat, const char * trait){
  this->observations = n;
  this->variables = p;
  X = new float[observations*variables];
  read_input(observations,variables,designmat,X);
  Xt = new float[variables*observations];
  for(int i=0;i<observations;++i){
    for(int j=0;j<variables;++j){
      Xt[j*observations+i] = X[i*variables+j];
    }
  }
  Y = new float[observations];
  read_input(observations,1,trait,Y);

  x_mm = new float[variables];
  A = new float[variables*variables];
  b = new float[variables*1];

  x_correction = new float[variables];
  v = new float[variables];
  x_0 = new float[variables];
  x_k = new float[variables];
  x_mm_dev = new float[variables];
  x_mm_dev2 = new float[variables];
  x_observations = new float[observations];
  x_observations_correction = new float[observations];
  V = new float[variables*observations];
  tV = new float[observations * variables];
  d = new float[observations];
  x_temp_correction = new float[observations];
  x_temp = new float[observations];
  x_temp_dev = new float[observations];
}

l0_reg_t::~l0_reg_t(){
  delete[] X;
  delete[] Xt;
  delete[] Y;
  delete[] x_mm;
  delete[] A;
  delete[] b;
  delete[] x_correction;
  delete[] v;
  delete[] x_0;
  delete[] x_k;
  delete[] x_mm_dev;
  delete[] x_mm_dev2;
  delete[] x_observations;
  delete[] x_observations_correction;
  delete[] x_temp_correction; 
  delete[] x_temp;
  delete[] x_temp_dev;
  delete[] V;
  delete[] tV;
  delete[] d;
}

void l0_reg_t::run(int top_k) {
//    // global parameters
  float tolerance   = 1e-6;
  float max_iter    = 1e4;
  float epsilon_min = 1e-15;
  float rho_inc     = 1.2;
  float eps_dec     = 1.2;
  //int m = observations;
  //int n = variables;
//    // initialize all output variables to zero.
//    MM_iter     = 0;
//    MM_time     = 0;
//    MM_opt      = 0;
//    // check for required variables
//    // check for warm-start
  for(int j=0;j<variables;++j) {
    x_mm[j] = 0;
    x_correction[j] = x_mm[j];
  }

  float rho = 1;
//    // initialize other problem parameters for MM algorithm
  float epsilon       = 1;
  int epsilon_reset = 0;
  //float current_obj   = 1e10;
//    // want to map X, Y to A, b
//    A = t(X) %*% X;
  //mmultiply(float *  a,int a_rows, int a_cols, float *  b,int b_cols, float *  c);
  //cerr<<"proxmap_t::mmultiply(Xt,variables,observations,Y,1,b)\n";
  proxmap_t::mmultiply(Xt,variables,observations,Y,1,b);
  for(int j=0;j<variables;++j) b[j]*=-1;
  ifstream testA("A_mat.txt");
  if(testA.is_open()){
    testA.close();
    read_input(observations,1,"A_mat.txt",Y);
    //cerr<<"Using cached copy of A\n";
  }else{
    //cerr<<"proxmap_t::mmultiply(Xt,variables,observations,A)\n";
    proxmap_t::mmultiply(Xt,variables,observations,A);
    ofstream ofs("A_mat.txt");
    for(int j=0;j<variables;++j){
      for(int k=0;k<variables;++k){
        if(k) ofs<<"\t";
        ofs<<A[j*variables+k];
      }
      ofs<<endl;
    }
    ofs.close();
  }
  //proxmap_t::mmultiply(Xt,variables,observations,X,variables,A);
  //b = - t(X) %*% Y;

  for(int j=0;j<observations;++j){
    x_observations[j] = 0;
    x_observations_correction[j] = x_observations[j];
  } 
  read_input(variables,observations,"right_singular_vectors.txt",V);
  read_input(observations,1,"eigenvalues.txt",d);
  for(int i=0;i<variables;++i){
    for(int j=0;j<observations;++j){
      tV[j*variables+i] = V[i*observations+j];
    }
  }
  

   
  float normb = proxmap_t::norm(b,variables);
  //cerr<<"Norm B:" <<normb<<endl;
  //cerr<<"D first and last "<<d[0]<<"\t"<<d[observations-1]<<endl;
  float rho_max = (2 * d[0] / d[observations-1] + 1) * normb;
  //cerr<<"Rho max: "<<rho_max<<endl;
  // project onto k largest components, then calculate loss, objective
  project_k(variables,x_mm, top_k,x_k);
  //for(int j=0;j<variables;++j) cerr<<" "<<x_k[j];
  //cerr<<endl;
  for(int j=0;j<variables;++j) x_mm_dev[j] = x_mm[j] - x_k[j];
  float dist = proxmap_t::norm(x_mm_dev,variables);
  float next_loss = loss();
  float next_obj = next_loss + rho * sqrt(dist*dist + epsilon);
  cerr<<"\nBegin MM algorithm\n\n";
  cerr<<"Iter\tNorm\tFeasible Dist\tRho\tEpsilon\tObjective\n";

  for(int mm_iter = 0;mm_iter<max_iter;++mm_iter){
    if(mm_iter==(max_iter-1)){
      cerr<<"Warning: MM algorithm has hit max iterations "<<max_iter<<"!\n";
      break;  
    }
    for(int j=0;j<variables;++j) x_0[j] = x_mm[j];
    float current_obj = next_obj;
    float lam = sqrt(dist*dist+epsilon)/rho;
    //cerr<<"Lam is "<<lam<<endl;
    for(int j=0;j<variables;++j){
      v[j] = x_k[j] - lam*b[j];
      x_mm[j] = v[j];
      //cerr<<"X_mm: "<<b[j]<<","<<x_mm[j]<<endl;
      x_correction[j] = v[j];      
    }
    
    //cerr<<"proxmap_t::mmultiply(tV,observations,variables,x_correction,1,x_temp_correction)\n";
    proxmap_t::mmultiply(tV,observations,variables,x_correction,1,x_temp_correction);
    //for(int i=0;i<observations;++i) cerr<<"x_temp_corr: "<<x_temp_correction[i]<<endl;
    //cerr<<"proxmap_t::mmultiply(tV,observations,variables,x_mm,1,x_temp)\n";
    proxmap_t::mmultiply(tV,observations,variables,x_mm,1,x_temp);
    for(int j=0;j<observations;++j){
      x_temp[j]*=(1/(lam*d[j]+1));
    }
    for(int j=0;j<observations;++j) x_temp_dev[j] = x_temp[j]-x_temp_correction[j];
    //cerr<<"proxmap_t::mmultiply(V,variables,observations,x_temp_dev,1,x_mm)\n";
    proxmap_t::mmultiply(V,variables,observations,x_temp_dev,1,x_mm);
    for(int j=0;j<variables;++j) x_mm[j]+=v[j];

    project_k(variables,x_mm, top_k,x_k);
    //for(int j=0;j<variables;++j) cerr<<"x_mm: "<<x_mm[j]<<" x_k:" <<x_k[j]<<endl;
    for(int j=0;j<variables;++j) {
      x_mm_dev[j] = x_mm[j] - x_k[j];
      x_mm_dev2[j] = x_mm[j] - x_0[j];
      //cerr<<"dev 1 "<<x_mm_dev[j]<<" dev2 "<<x_mm_dev2[j]<<endl;
    }
    
    //cerr<<"Here\n";
    dist = proxmap_t::norm(x_mm_dev,variables);
    //cerr<<"Dist : "<<dist<<endl;
    next_loss = loss();
    next_obj = next_loss + rho * sqrt(dist*dist + epsilon);
    float the_norm = proxmap_t::norm(x_mm_dev2,variables);
    float scaled_norm = the_norm/(proxmap_t::norm(x_0,variables)+1);
    bool converged = scaled_norm/tolerance;
    bool feasible = dist<tolerance;
    cerr<<mm_iter<<"\t"<<the_norm<<"\t"<<dist<<"\t"<<rho<<"\t"<<
    epsilon<<"\t"<<next_obj<<endl;
    if (feasible && converged){
      cerr<<"\nMM algorithm has converged successfully.\n";
      for(int j=0;j<variables;++j){
        if(fabs(x_mm[j])<tolerance) x_mm[j] = 0;
      }

      project_k(variables,x_mm, top_k,x_k);
      for(int j=0;j<variables;++j) {
        x_mm_dev[j] = x_mm[j] - x_k[j];
      }    
      dist = proxmap_t::norm(x_mm_dev,variables);
      float mm_loss = loss();
      float mm_obj = mm_loss + rho * sqrt(dist*dist + epsilon);
      cerr<<"MM Results:\nIterations: "<<mm_iter<<"\n";
      cerr<<"Final rho: "<<rho<< "\n";
      cerr<<"Final Loss: "<< mm_loss<< "\n";
      cerr<<"Final Objective: "<< mm_obj<< "\n";
      cerr<<"Sparsity constraint satisfied to tolerance "<<tolerance<<"? "
      <<feasible<<"\n";
      break;    
    }
    if (feasible && epsilon_reset<1){
      epsilon = eps_dec;
      epsilon_reset = epsilon+1;
      rho_inc=1;
    }
    if (feasible && next_obj>current_obj+tolerance){
      cerr<<"\nMM  algorithm fails to descend!\n";
      cerr<<"MM Iteration: "<< mm_iter<<"\n";
      cerr<<"Current objective: "<< current_obj<< "\n";
      cerr<<"Next objective: "<< next_obj<< "\n";
      cerr<<"Difference in objectives: "<< fabs(next_obj - current_obj)<< "\n";
      return;
    }
    if(dist>10*tolerance || the_norm>10*tolerance){  
      rho = rho*rho_inc<rho_max?rho*rho_inc:rho_max;
    }
    epsilon = epsilon/eps_dec>epsilon_min?epsilon/eps_dec:epsilon_min;
  }
}

void l0_reg_t::output(){
  for(int j=0;j<variables;++j){
    if (x_mm[j]!=0){
      cout<<j<<"\t"<<x_mm[j]<<endl;
    }
  }
}

int main(int argc,char * argv[]){
  if(argc<5){
    cerr<<"Usage: <n><p><designmatrix><trait>\n";
    return 1;
  }
  int arg = 0;
  int n = atoi(argv[++arg]);
  int p = atoi(argv[++arg]);
  const char * designmat = argv[++arg];
  const char * trait = argv[++arg];
  l0_reg_t l0_reg(n,p,designmat,trait);
  cerr<<"L0 created\n";
  l0_reg.run(10);
  l0_reg.output();
  return 0;
}
