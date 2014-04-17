#include<iostream>
#include<sstream>
#include<fstream>
#include<cstdlib>
#include<math.h>
#include<vector>
#include<set>
#include<string.h>

using namespace std;

float * distances;
int n,p,k;
float phi;

struct neighbor_t{
  int index;
  float dist;
  float weight;
};

struct byEpsilonDistance{
  bool operator()(const float & a,const float & b){
    return a<b;
    //return (b-a)>.0001;
  }
};

struct byEpsilonWeight{
  bool operator()(const float & a,const float & b){
    return b>a;
    //return (a-b)>.0001;
  }
};

struct byDistance{
  bool operator()(const neighbor_t & a,const neighbor_t & b){
    return a.dist<b.dist;
  }
};

struct byWeight{
  bool operator()(const neighbor_t & a,const neighbor_t & b){
    return a.weight>b.weight;
  }
};

typedef multiset<neighbor_t,byDistance> neighbor_set_t;
typedef multiset<neighbor_t,byWeight> neighbor_set2_t;
typedef set<float,byEpsilonDistance> distance_set_t;
typedef set<float,byEpsilonWeight> weight_set_t;
neighbor_set_t * neighbor_set_arr;
neighbor_set2_t * neighbor_set2_arr;
distance_set_t * distance_set_arr;
weight_set_t * weight_set_arr;

void load_distances(){
  string line;
  for(int i1=0;i1<n;++i1){
    getline(cin,line);
    istringstream iss(line);
    for(int i2=0;i2<n;++i2){
      //iss>>distances[i1*n+i2];
      float dist;
      iss>>dist;
      if (i1!=i2 && dist>0){
        neighbor_t neighbor;
        neighbor.index = i2;
        neighbor.dist = dist;
        neighbor_set_arr[i1].insert(neighbor);
      }
    }
  }
}

void store_distances(){
  distances = new float[n*n];
  for(int i=0;i<n;++i){
    for(int j=0;j<n;++j){
      distances[i*n+j]=-1;
    }
  }
  bool debug = false;
  for(int i=0;i<n;++i){
    neighbor_set_t::iterator it3 = neighbor_set_arr[i].begin();
    if (debug)cerr<<"STORE_DIST: Index "<<i<<" with neighbors (index,dist):";
    bool under_threshold = true;
    for(int j=0;j<k;++j){ 
    //while(under_threshold){
      neighbor_t neighbor = *it3;
      //under_threshold = neighbor.dist<=threshold;
      //if(under_threshold){
        if (debug)cerr<<" "<<neighbor.index<<","<<neighbor.dist;
        if(neighbor.index>i){
          distances[i*n+neighbor.index] = neighbor.dist; 
        }else{
          distances[neighbor.index*n+i] = neighbor.dist; 
          if (debug) cerr<<"REVERSE for "<<i<<" and "<<neighbor.index<<endl;
        }
        
        it3++;
      //}
    }
    if (debug)cerr<<endl;
    //neighbor_set2_t::iterator it4 = neighbor_set2_arr[i].begin();
    //if (debug)cerr<<"STORE_DIST: Index "<<i<<" with neighbors (index,weight):";
    //bool over_threshold = true;
    //while(over_threshold){
    //  neighbor_t neighbor = *it4;
    //  over_threshold = neighbor.weight>=threshold2;
    //  if(over_threshold){
    //    if (debug)cerr<<" "<<neighbor.index<<","<<neighbor.weight; 
    //    //distances[i*n+neighbor.index] = neighbor.dist; 
    //    it4++;
    //  }
   // }
   // if (debug)cerr<<endl;
  }
}

void print_weights(){
  bool debug = false;
  float min_weight = 10;
  float max_weight = 0;
  for(int i1=0;i1<n;++i1){
    for(int i2=0;i2<n;++i2){
      if (i2) cout<<" ";
      if (i2>i1 && distances[i1*n+i2]>=0){
        float kernel =  1./exp(phi*distances[i1*n+i2]);
        if (kernel>max_weight) max_weight = kernel;
        if (kernel<min_weight) min_weight = kernel;
        if (debug) cerr<<"DEBUG: Weight for "<<i1<<" to "<<i2<<" is "<<kernel<<endl;
        cout<<kernel;
      }else if (i1>i2 && distances[i2*n+i1]>=0){
        float kernel =  1./exp(phi*distances[i2*n+i1]);
        //if (debug) cerr<<"DEBUG: Weight for "<<i1<<" to "<<i2<<" is "<<kernel<<endl;
        cout<<kernel;
      }else{
        cout<<"0";
      }
    }
    cout<<endl;
  }
  cerr<<"Weight range is "<<min_weight<<" to "<<max_weight<<endl;
}

int main(int argc,char * argv[]){
  if(argc<5){
    cerr<<"Usage: <n> <p> <phi> <k> (STDIN is distance matrix)\n";
    return 1;
  }
  int arg=0;
  n = atoi(argv[++arg]);
  p = atoi(argv[++arg]);
  phi = atof(argv[++arg]);
  k = atoi(argv[++arg]);
  distances = new float[n*n];
  neighbor_set_arr = new neighbor_set_t[n];
  load_distances();
  store_distances();
  print_weights();
  return 0;
}
