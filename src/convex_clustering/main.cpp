#include"../proxmap.hpp"
#include"cluster.hpp"

int main(int argc, char * argv[]){
  try{
    if (argc<2){
      cerr<<"Usage: <configfile>\n";
      return 1;
    }
    int arg=0;
    string configfile = argv[++arg];
    proxmap_t * proxmap =  new cluster_t();
    double start = clock();
    proxmap->init(configfile);
    proxmap->allocate_memory();
    proxmap->run();
    delete proxmap;
    cerr<<"Total time to run convexcluster: "<<(clock()-start)/CLOCKS_PER_SEC<<endl;
  }catch(const char * & estr){
    cerr<<"Exception caught of message: "<<estr<<endl;
    return -1;
  }
  return 0;
}

