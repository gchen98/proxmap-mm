#include"../proxmap.hpp"
#include"regression.hpp"

regression_t::regression_t(){
}

regression_t::~regression_t(){
}


void regression_t::update_Xbeta(){
  mmultiply(X,n,p,beta,1,Xbeta);
}

void regression_t::update_XXT(){
  mmultiply(X,n,p,XT,n,XXT);
}

void regression_t::project_beta(){
  float xt_lambda[p];
  mmultiply(XT,p,n,lambda,1,xt_lambda);
  for(int j=0;j<p;++j){
    beta_project[j] = beta[j]-xt_lambda[j];
    //if (j<5) cerr<<"Var "<<j<<" projection is "<<beta_project[j]<<endl;
  }
}

void regression_t::project_theta(){
  for(int i=0;i<n;++i){
    theta_project[i] = theta[i]+lambda[i];
    //if (i<5) cerr<<"Subject "<<i<<" theta projection is "<<theta_project[i]<<endl;
  }
}

float regression_t::get_map_distance(){
  float beta_dev = 0;
  float theta_dev = 0;
  for(int j=0;j<p;++j){
    beta_dev+=(beta[j]-beta_project[j])*(beta[j]-beta_project[j]);
  }
  beta_dev = sqrt(beta_dev);
  for(int i=0;i<n;++i){
    theta_dev+=(theta[i]-theta_project[i])*(theta[i]-theta_project[i]);
  }
  theta_dev = sqrt(theta_dev);
  return beta_dev+theta_dev; 
}

void regression_t::update_theta(){
  float d2 = get_map_distance();
  w = rho/sqrt(d2+epsilon);
  //cerr<<"D2 is "<<d2<<" w is "<<w<<endl;
  for(int i=0;i<n;++i){
    theta[i] = theta[i] + y[i]/(blocks+w) - theta[i]/(blocks+w);
    //if (i<5) cerr<<"Subject "<<i<<" theta is "<<theta[i]<<endl;
  }
}

void regression_t::update_beta(){
  if (w==0) w = .0001;
  for(int j=0;j<p;++j){
    float shrinkage = 1.-(nu[j]/w/fabs(beta_project[j]));
    if (shrinkage<0 || beta_project[j]==0) shrinkage = 0;
    beta[j] = shrinkage*beta_project[j];
    //if(j<3) cerr<<"shrinkage "<<shrinkage<<". Beta for var "<<j<<" is "<<beta[j]<<endl;
    
  }
}

void regression_t::update_lambda(){
  cerr<<"Updating lambda\n";
  // this method implements a Landweber update
  float gradient[n];
  for(int iter=0;iter<10;++iter){
    mmultiply(XXTI,n,n,lambda,1,gradient);
    for(int i=0;i<n;++i){
      gradient[i] += theta[i] - Xbeta[i];
      lambda[i]-=gradient[i]/L;
      //if (i<3) cerr<<"Gradient/Lambda "<<i<<" is "<<gradient[i]<<"/"<<lambda[i]<<endl;
    }  
  }
}

void regression_t::loss(){
  float l=0;
  for(int i=0;i<n;++i){
    l+=(y[i]-Xbeta[i])*(y[i]-Xbeta[i]);
  }
  l*=.5;
  //cerr<<"Loss is "<<l<<endl;
}

void regression_t::check_constraints(){
}

bool regression_t::in_feasible_region(){
  float norm = 0;
  int normalizer = 0;
  cerr<<"Constraint check:\n";
  for(int i=0;i<n;++i){
    if (i<10) cerr<<i<<": theta: "<<theta[i]<<" Xbeta: "<<Xbeta[i]<<endl;
    norm+=(theta[i]-Xbeta[i])*(theta[i]-Xbeta[i]);
    ++normalizer;
  }
  norm/=1.*normalizer;
  cerr<<"Feasibility: Normalized diff: "<<norm<<endl;
  return norm<1e-4;
}

void regression_t::parse_config_line(string & token,istringstream & iss){
  proxmap_t::parse_config_line(token,iss);
  if (token.compare("TRAIT")==0){
    iss>>config->traitfile;
  }else if (token.compare("NU")==0){
    iss>>config->nu;
  }
}


void regression_t::allocate_memory(string config_file){
  proxmap_t::allocate_memory(config_file);
  // figure out the dimensions
  const char * genofile = config->genofile.data();
  const char * phenofile = config->traitfile.data();
  n = linecount(genofile);
  p = colcount(genofile);
  cerr<<"Subjects: "<<n<<" and predictors: "<<p<<endl;
  blocks = 1;
  rho = 1;
  X = new float[n*p];
  XT = new float[p*n];
  y = new float[n];
  nu = new float[p];
  beta = new float[p];
  beta_project = new float[p];
  theta = new float[n];
  theta_project = new float[n];
  Xbeta = new float[n];
  // the Lagrangian vector
  lambda = new float[n];
  for(int i=0;i<n;++i) {
    theta_project[i] = theta[i] = 0;
    Xbeta[i] = 0;
    lambda[i] = 0;
  }
  for(int j=0;j<p;++j){
    beta_project[j] = beta[j] = 0;
    nu[j] = 0;
  }
  
  load_into_matrix(genofile,X,n,p);
  for(int j=0;j<p;++j){
    for(int i=0;i<n;++i){
      XT[j*n+i] = X[i*p+j];
    }
  }
  XXT = new float[n*n];
  mmultiply(X,n,p,XT,n,XXT);
  XXTI = new float[n*n];
  load_into_matrix(phenofile,y,n,1);
  // the parameter vector for the means Xbeta
  float row_sums[n];
  float col_sums[p];
  for(int j=0;j<p;++j) col_sums[j]=0;
  for(int i=0;i<n;++i){
    row_sums[i] = 0;
    for(int j=0;j<p;++j){
      row_sums[i]+=X[i*p+j];
      col_sums[j]+=X[i*p+j];
    }
  }
  float maxr=-1000,maxc=-1000;
  for(int i=0;i<n;++i) if (row_sums[i]>maxr)maxr = row_sums[i];
  for(int j=0;j<p;++j) if (col_sums[j]>maxc)maxc = col_sums[j];
  L = 1 + maxr*maxc ;
  cerr<<"Max rowsum, colsum, and L: "<<maxr<<","<<maxc<<","<<L<<endl;
  for(int i1=0;i1<n;++i1){
    for(int i2=0;i2<n;++i2){
      XXTI[i1*n+i2] = (i1==i2)? XXT[i1*n+i2]+1:XXT[i1*n+i2];
    }
  }
}

float regression_t::infer_rho(){
  return 0;
}

void regression_t::initialize(float mu){
}
void regression_t::update_map_distance(){}

void regression_t::finalize_iteration(){
}

void regression_t::iterate(){
  //for(int iter=0;iter<10;++iter){
    update_beta();
    update_theta();
    mmultiply(X,n,p,beta,1,Xbeta);
    ////for(int i=0;i<5;++i){ cerr<<"Xbeta["<<i<<"] is "<<Xbeta[i]<<endl;}
    //loss();
    update_lambda();
    project_beta();
    project_theta();
    //float obj = evaluate_obj();
    //cerr<<"Current objective at iter "<<iter<<" is "<<obj<<endl;
    //check_constraints();
  //}
  //cout<<"BETA:\n";
  //for(int j=0;j<p;++j){
    //if (j<10) cerr<<"BETA: "<<j<<":"<<beta[j]<<endl;
  //}
  //cerr<<"Done!\n";
}

void regression_t::print_output(){
  cout<<"INDEX\tBETA\n";
  for(int j=0;j<p;++j){
    cout<<j<<":"<<beta[j]<<endl;
  }
  //cerr<<"Done!\n";
}

float regression_t::evaluate_obj(){
  float obj=0;
  float norm = 0;
  for(int i=0;i<n;++i){
    norm+=(y[i]-theta[i])*(y[i]-theta[i]);
  }
  float penalty = 0;
  for(int j=0;j<p;++j){
    penalty+=nu[j]*fabs(beta[j]);
  }
  obj = .5*norm+penalty+get_prox_dist_penalty();
  return obj;
}

int main_regression(int argc,char * argv[]){
//  if(argc<3){
//    cerr<<"Usage: <genofile> <outcome file>\n";
//    return 1;
//  }
//  int arg=0;
//  const char * genofile = argv[++arg];
//  const char * phenofile = argv[++arg];
//
  return 0;
}

