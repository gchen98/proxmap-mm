#include<iostream>
#include<fstream>
#include<sstream>
#include<cstdlib>
#include<map>
using namespace std;

const int MAX_P=10000;

int n,p;
float * data;
int * fusions;

struct node_t;
struct node_t{
  int id;
  float weight;
  node_t * child1;
  node_t * child2;
  float val[MAX_P];
  node_t(int id){
    this->id = id;
    this->weight = 1;
    child1 = NULL;
    child2 = NULL;
  }
  void update_mean_val(int p){
    //cerr<<"Merging values of "<<child1->id<<" and "<<child2->id<<endl;
    this->weight = child1->weight + child2->weight;
    for(int j=0;j<p;++j){
      val[j] = (child1->weight * child1->val[j]+child2->weight * child2->val[j])/ this->weight;
      //cerr<<" "<<val[j];
    }
    //cerr<<endl;
  }

  void update_singleton_val(int p){
    int index = -1*this->id - 1;
    //cerr<<"Initializing values of "<<index<<endl;
    for(int j=0;j<p;++j){
      val[j] = data[index*p+j];
      //cerr<<" "<<val[j];
    }
    //cerr<<endl;
  }

};

map<int,node_t * > node_map;

void parse_data(const char * datafile){
  data = new float[n*p];
  ifstream ifs(datafile);
  if (!ifs.is_open()){
    cerr<<"Can't open "<<datafile<<endl;
    exit(1);
  }
  string line;
  for(int i=0;i<n;++i){
    getline(ifs,line);
    istringstream iss(line);
    for(int j=0;j<p;++j){
      iss>>data[i*p+j];
    }
  }
  ifs.close();
}

void parse_fusion(const char * fusionfile){
  fusions = new int[(n-1)*2];
  ifstream ifs(fusionfile);
  if (!ifs.is_open()){
    cerr<<"Can't open "<<fusionfile<<endl;
    exit(1);
  }
  string line;
  for(int i=0;i<n-1;++i){
    getline(ifs,line);
    istringstream iss(line);
    for(int j=0;j<2;++j){
      iss>> fusions[i*2+j];
      //cerr<<"ID: "<<fusions[i*2+j]<<endl;
    }
  }
  ifs.close();
}

void print_data(int index){
  ostringstream oss;
  oss<<"hclust.centers/"<<100000+index;
  const char * outfile=oss.str().data();
  ofstream ofs(outfile);
  for(int i=0;i<n;++i){
    for(int j=0;j<p;++j){
      if(j) ofs<<"\t";
      ofs<<data[i*p+j];
    }
    ofs<<endl;
  }
  ofs.close();
  
}

void update_leaves(node_t * node,float * new_val){
  if (node->id<0) {
    int index = -1*node->id-1;
    //cerr<<"Updating leaf: "<<index<<endl;
    for(int j=0;j<p;++j) {
      data[index*p+j] = new_val[j];
      //cerr<<" "<<data[index*p+j];
    }
    //cerr<<endl;
  }else{
    update_leaves(node->child1,new_val);
    update_leaves(node->child2,new_val);
  }
}

void make_nodes(){
  for(int i=0;i<n-1;++i){
    node_t * node_temp[2];
    for(int j=0;j<2;++j){
      int id = fusions[i*2+j];
      if (fusions[i*2+j]<0){
        node_t * node = new node_t(id);
        node_temp[j] = node; 
        node->update_singleton_val(p);
        node_map[id] = node;
      }else if (fusions[i*2+j]>0){
        node_temp[j] = node_map[id]; 
      }else{
        cerr<<"Fusion index should not be zero!\n";
        exit(1);
      }
    }
    node_t * merged = new node_t(i+1);
    merged->child1 = node_temp[0];
    merged->child2 = node_temp[1];
    merged->update_mean_val(p);
    node_map[i+1] = merged;
    update_leaves(merged,merged->val);
    print_data(i+1);
  }
}

int main(int argc,char * argv[]){
  if (argc<5){
    cerr<<"Usage: <n> <p> <datafile> <fusion file>\n";
    return 1;
  }
  int arg=0;
  n = atoi(argv[++arg]);
  p = atoi(argv[++arg]);
  const char * datafile=argv[++arg];
  const char * fusionfile=argv[++arg];
  parse_data(datafile);
  parse_fusion(fusionfile);
  print_data(0);
  make_nodes();
    
  delete [] data;
  delete [] fusions;
}
