#include"../proxmap.hpp"
#include<random_access.hpp>
#include<plink_data.hpp>
#ifdef USE_GPU
#include<ocl_wrapper.hpp>
#endif
#include"block_descent.hpp"
#include"iterative_hard_threshold.hpp"


int main(int argc, char * argv[]){
  try{
    if (argc<3){
      cerr<<"Usage: <L0 regression method [iht|block_descent]> <configfile>\n";
      return 1;
    }
    int arg=0;
    string analysis = argv[++arg];
    string configfile = argv[++arg];
    proxmap_t * proxmap =  NULL;
    if (analysis.compare("block_descent")==0){
      proxmap = new block_descent_t(true);
    }else if (analysis.compare("iht")==0){
      proxmap = new iterative_hard_threshold_t(true);
    }else{
      cerr<<"Invalid method of "<<analysis<<endl;
      return 1;
    }
    if(proxmap!=NULL){
      double start = clock();
      proxmap->init(configfile);
      proxmap->allocate_memory();
      proxmap->run();
      delete proxmap;
      cerr<<"Total time to run L0: "<<(clock()-start)/CLOCKS_PER_SEC<<endl;
    }
  }catch(const char * & estr){
    cerr<<"Exception caught of message: "<<estr<<endl;
    return -1;
  }
  return 0;
}

