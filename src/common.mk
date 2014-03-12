ATI=$(HOME)/software/ati-stream-sdk-v2.2-lnx64
BOOST_INC_FLAGS = -I/usr/include/boost
BOOST_LIB_FLAGS = -L/usr/lib64
GSL_LIB_FLAGS = -lgsl -lgslcblas
OPENCL_INC_FLAGS = -I$(ATI)/include
OPENCL_LIB_FLAGS = -L$(ATI)/lib/x86_64 -lOpenCL

PREPROC+=-DUSE_GPU
CFLAGS = -Wall -g $(BOOST_INC_FLAGS) $(GSL_INC_FLAGS) $(OPENCL_INC_FLAGS)
LINKFLAGS = -g -lm $(BOOST_LIB_FLAGS) $(GSL_LIB_FLAGS) $(OPENCL_LIB_FLAGS)

#compiler
CC=g++
LINKER=g++

