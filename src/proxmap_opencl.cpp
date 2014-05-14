#include"proxmap.hpp"
#include"cl_templates.hpp"

#ifdef USE_GPU
void proxmap_t::init_opencl(){
  cerr<<"Initializing OpenCL\n";
  if(run_gpu){
  // initialize the GPU if necessary
    int platform_id = 0;
    //int platform_id = config->platform_id;
    int device_id = 0;
    //int device_id = config->device_id;
    cerr<<"Initializing GPU with platform id "<<platform_id<<
    " and device id "<<device_id<<".\n";
    vector<cl::Platform> platforms;
    err = cl::Platform::get(&platforms);
    cerr<<"Platform ID "<<platform_id<<" has name "<<
    platforms[platform_id].getInfo<CL_PLATFORM_NAME>().c_str()<<endl;
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
    (cl_context_properties)(platforms[platform_id])(),0};
    context = new cl::Context(CL_DEVICE_TYPE_GPU,cps);
    devices = context->getInfo<CL_CONTEXT_DEVICES>();
    cerr<<"There are "<<devices.size()<<" devices\n";
    cerr<<"Device ID "<<device_id<<" is a ";
    cl_device_type dtype = devices[device_id].getInfo<CL_DEVICE_TYPE>();
    switch(dtype){
      case CL_DEVICE_TYPE_GPU:
        cerr<<"GPU\n";
        break;
      case CL_DEVICE_TYPE_CPU:
        cerr<<"CPU\n";
        break;
    } 
    commandQueue = new cl::CommandQueue(*context,devices[device_id],0,&err);
    vector<string> sources;
    sources.push_back("cl_constants.h");
    sources.push_back("common.c");
    ostringstream full_source_str;
    for(uint j=0;j<sources.size();++j){
      ostringstream oss;
      oss<<config->kernel_base<<"/"<<sources[j];
      string full_path = oss.str();
      cerr<<"Opening "<<full_path<<endl;
      ifstream ifs(full_path.data());
      clSafe(ifs.is_open()?CL_SUCCESS:-1,"kernel_path not found");
      string source_str(istreambuf_iterator<char>(ifs),(istreambuf_iterator<char>()));
      full_source_str<<source_str;
    }
    string source_str = full_source_str.str();
    // create a program object from kernel source
    cerr<<"Creating source\n";
    cl::Program::Sources source(1,make_pair(source_str.c_str(),source_str.length()+1));
    cerr<<"Creating program\n";
    program = new cl::Program(*context,source);
    //const char * preproc;
//    if (geno_dim==PHASED_INPUT){
//      preproc = "-Dphased";
//    } else if(geno_dim==UNPHASED_INPUT){
//      preproc = "-Dunphased";
//    }
    err = program->build(devices);
    //err = program->build(devices,preproc);
    if(err!=CL_SUCCESS){
      cerr<<"Build failed:\n";
      string buffer;
      program->getBuildInfo(devices[0],CL_PROGRAM_BUILD_LOG,&buffer);
      cerr<<buffer<<endl;
      throw "Aborted from OpenCL build fail.";
    }
    cerr<<"GPU kernel arguments assigned.\n";
  }
}
#endif

