#include<iostream>
#include<sstream>
#include<fstream>
#include<cstdlib>
#include<math.h>
#include<vector>
#include<set>
#include<string.h>

using namespace std;

float * data;
float * annot_data;
float * distances;
int n,p,annot_cols;
string annot_file;
float annot_weight;

void read_input(){
  string line;
  for(int i=0;i<n;++i){
    getline(cin,line);
    istringstream iss(line);
    for(int j=0;j<p;++j){
      iss>>data[i*p+j];
    }
  }
}

void read_annot(string annot_file){
  ifstream ifs(annot_file.data());
  string line;
  for(int i=0;i<n;++i){
    getline(ifs,line);
    istringstream iss(line);
    for(int j=0;j<annot_cols;++j){
      iss>>annot_data[i*annot_cols+j];
      //cerr<<"Annot at "<<i<<": "<<annot_data[i*annot_cols+j]<<endl;
    }
  }
  ifs.close();
}

void read_input_compact(){
  string line;
  int offset = static_cast<int>('0');
  for(int j=0;j<p;++j){
    getline(cin,line);
    for(int i=0;i<n;++i){
      data[i*p+j] = static_cast<int>(line[i])-offset;
      //cerr<<"Data at "<<i<<","<<j<<":"<<data[i*p+j]<<endl;
    }
  }
}

void store_distances(){
  bool debug = false;
  float data_max = 0;
  float annot_max = 0;
  float *temp_data = new float[n*n];
  float *temp_annot = new float[n*n];
  for(int i1=0;i1<n-1;++i1){
    for(int i2=i1+1;+i2<n;++i2){
      // get Euclidean distance
      float data_dist = 0;
      for(int j=0;j<p;++j){
        data_dist+=(data[i1*p+j]-data[i2*p+j])*(data[i1*p+j]-data[i2*p+j]);
      }
      temp_data[i1*n+i2] = data_dist;
      if (data_dist>data_max) data_max = data_dist;
      float annot_dist = 0;
      for(int j=0;j<annot_cols;++j){
        annot_dist+=(annot_data[i1*annot_cols+j]-annot_data[i2*annot_cols+j])*(annot_data[i1*annot_cols+j]-annot_data[i2*annot_cols+j]);
      }
      temp_annot[i1*n+i2] = annot_dist;
      if (annot_dist>annot_max) annot_max = annot_dist;
    }
  }
  for(int i1=0;i1<n-1;++i1){
    for(int i2=i1+1;+i2<n;++i2){
      // for testing
      //if(annot_dist>0) annot_dist += 1;
      float data_dist = temp_data[i1*n+i2]/data_max;
      float annot_dist = temp_annot[i1*n+i2]/annot_max;
      distances[i1*n+i2] = distances[i2*n+i1] = annot_weight*annot_dist+(1.-annot_weight)*data_dist;
      if (debug) cerr<<"DISTANCE INDEX "<<i1<<","<<i2<<" IS "<<distances[i1*n+i2]<<" with data dist: "<<data_dist<<", annot dist: "<<annot_dist<<endl;
    }
  }
  delete[]temp_annot;
  delete[]temp_data;
}
      
void print_distances(){
  for(int i=0;i<n;++i){
    for(int j=0;j<n;++j){
      if(j) cout<<"\t";
      cout<<distances[i*n+j];
    }
    cout<<endl;
  }
}



int col_count(string filename){
  ifstream ifs(filename.data());
  if(!ifs.is_open()){
    cerr<<"Cannot open annotation file "<<filename<<".  Will not use annotation\n";
    annot_weight = 0;
    return -1;
  }
  string line,token;
  getline(ifs,line);
  istringstream iss(line);
  int col=0;
  while(iss>>token) ++col;
  ifs.close();
  return  col;
}

int main(int argc,char * argv[]){
  if(argc<6){
    cerr<<"Usage: <n> <p> [format=verbose|compact] <annotation file optional> <annotion_proportion>\n";
    return 1;
  }
  int arg=0;
  n = atoi(argv[++arg]);
  p = atoi(argv[++arg]);
  string format = argv[++arg]; 
  annot_file=argv[++arg];
  annot_weight = atof(argv[++arg]);
  data = new float[n*p];
  distances = new float[n*n];
  annot_cols = col_count(annot_file);
  if(annot_cols>0){
    annot_data = new float[n*annot_cols]; 
    read_annot(annot_file);
  }
  if(format.compare("verbose")==0){
    read_input();
  }else if(format.compare("compact")==0){
    read_input_compact();
  }else{
    cerr<<"Format "<<format<<" not supported\n";
    return 1;
  }
  store_distances();
  print_distances();
  delete[]data;
  return 0;
}
