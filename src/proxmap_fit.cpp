#include"proxmap.hpp"
#include"regression/regression.hpp"
#include"regression/regression_with_theta.hpp"
#include"convex_clustering/cluster.hpp"

int main(int argc, char * argv[]){
  try{
    int arg=0;
    if (argc<3){
      cerr<<"Usage: <analysis: [cluster|regression]> <configfile>\n";
      return 1;
    }
    string analysis = argv[++arg];
    string configfile = argv[++arg];
    proxmap_t * proxmap = NULL;
    if (analysis.compare("cluster")==0){ 
      proxmap = new cluster_t();
    }else if (analysis.compare("regression")==0){ 
      proxmap = new regression_t(true);
    }else if (analysis.compare("regression_with_theta")==0){ 
      proxmap = new regression_with_theta_t(true);
    }
    if (proxmap!=NULL){
      proxmap->init(configfile);
      proxmap->allocate_memory();
      proxmap->run();
      delete proxmap;
    }
  }catch(const char * & estr){
    cerr<<"Exception caught of message: "<<estr<<endl;
    return -1;
  }
  return 0;
}

