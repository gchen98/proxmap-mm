
#compiler
CC=g++
LINKER=g++

CUDA=$(HOME)/cuda_sdk
#ATI=$(HOME)/software/ati-stream-sdk-v2.2-lnx64
BOOST_INC_FLAGS = -I/usr/include/boost
BOOST_LIB_FLAGS = -L/usr/lib64
GSL_LIB_FLAGS = -lgsl -lgslcblas
MPI_LIB_FLAGS = -L/usr/lib64/openmpi/lib
OPENCL_INC_FLAGS = -I$(CUDA)/OpenCL/common/inc
OPENCL_LIB_FLAGS = -L$(CUDA)/OpenCL/common/lib -lOpenCL -locl_wrapper

#OPENCL_INC_FLAGS = -I$(ATI)/include
#OPENCL_LIB_FLAGS = -L$(ATI)/lib/x86_64 -lOpenCL
GCHEN_INC_FLAGS = -I$(HOME)/include
GCHEN_LIB_FLAGS = -L$(HOME)/shared_objects -lgwasutil 
CFLAGS = -Wall -g  $(BOOST_INC_FLAGS) $(GSL_INC_FLAGS) 
LINKFLAGS = -g -lm  $(BOOST_LIB_FLAGS) $(GSL_LIB_FLAGS)


ifeq ($(use_gpu),1)
	PREPROC+= -DUSE_GPU
	CFLAGS+= $(OPENCL_INC_FLAGS)
	LINKFLAGS+= $(OPENCL_LIB_FLAGS)
endif

ifeq ($(use_mpi),1)
	PREPROC+= -DUSE_MPI
	CC=mpic++
	LINKFLAGS+= $(MPI_LIB_FLAGS)
	LINKER=mpic++
endif
