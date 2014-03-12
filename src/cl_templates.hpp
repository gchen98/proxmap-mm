#include<iostream>
#include<sstream>
#ifdef USE_GPU
#include<CL/cl.h>
#include<CL/cl.hpp>
#endif
//#include"cluster.hpp"
using namespace std;

const char * clError (cl_int rc);
void clSafe (cl_int rc, string functionname);
//bool debug_opencl = false;

inline void checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name
                 << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

#ifdef USE_GPU

//template<class T> void cluster_t::createBuffer(int rw, int dim, const char * label,cl::Buffer * & buf){
//  ostringstream oss;
//  oss<<"Creating buffer "<<label;
//  string mesg = oss.str();
//  if (debug_opencl)  cerr<<mesg<<" of dimension "<<dim<<endl;
//  buf = new cl::Buffer(*context, rw, sizeof(T) * dim, NULL, &err);
//  clSafe(err,oss.str().data());
//}
//
//template<typename T> void cluster_t::setArg(cl::Kernel * &  kernel,int & index,T arg,const char * label){
//  ostringstream oss;
//  oss<<"Setting argument "<<index<<" for kernel "<<label;
//  string mesg = oss.str();
//  if (debug_opencl) cerr<<mesg<<endl;
//  err = kernel->setArg(index++, arg);
//  clSafe(err,oss.str().data());
//}
//
//template<typename T> void cluster_t::writeToBuffer(cl::Buffer * & buffer,int dim,T hostArr,const char * label){
//  ostringstream oss;
//  T t;
//  oss<<"Writing to buffer "<<label;
//  string mesg = oss.str();
//  if (debug_opencl) cerr<<mesg<<" of dimension "<<dim<<endl;
//  err = commandQueue->enqueueWriteBuffer(*buffer, CL_TRUE, 0, sizeof(*t)*dim,hostArr,NULL,NULL);
//  clSafe(err,oss.str().data());
//}
//
//template<typename T> void cluster_t::readFromBuffer(cl::Buffer * & buffer,int dim,T hostArr,const char * label){
//  ostringstream oss;
//  T t;
//  oss<<"Reading from buffer "<<label;
//  string mesg = oss.str();
//  if (debug_opencl) cerr<<mesg<<" of dimension "<<dim<<endl;
//  err = commandQueue->enqueueReadBuffer(*buffer, CL_TRUE, 0, sizeof(*t)*dim,hostArr,NULL,NULL);
//  clSafe(err,oss.str().data());
//}
//
#endif
