#include"proxmap.hpp"
#include"regression/quadratic.hpp"
#include"regression/param_split.hpp"
#include"regression/param_split_with_theta.hpp"
#include"convex_clustering/cluster.hpp"
#include"genetree/genetree.hpp"

int main(int argc, char * argv[]){
  try{
    int arg=0;
    if (argc<3){
      cerr<<"Usage: <analysis: [cluster|quadratic|param_split|genetree]> <configfile>\n";
      return 1;
    }
    string analysis = argv[++arg];
    string configfile = argv[++arg];
    proxmap_t * proxmap = NULL;
    if (analysis.compare("cluster")==0){ 
      proxmap = new cluster_t();
    }else if (analysis.compare("quadratic")==0){ 
      proxmap = new quadratic_t(true);
    }else if (analysis.compare("param_split")==0){ 
      proxmap = new param_split_t(true);
    }else if (analysis.compare("param_split_with_theta")==0){ 
      //proxmap = new param_split_with_theta_t(true);
    }else if (analysis.compare("genetree")==0){ 
      proxmap = new genetree_t();
    }
    if (proxmap!=NULL){
      double start = clock();
      proxmap->init(configfile);
      proxmap->allocate_memory();
      proxmap->run();
      delete proxmap;
      cerr<<"Total time to run proxmap_fit: "<<(clock()-start)/CLOCKS_PER_SEC<<endl;
    }
  }catch(const char * & estr){
    cerr<<"Exception caught of message: "<<estr<<endl;
    return -1;
  }
  return 0;
}

