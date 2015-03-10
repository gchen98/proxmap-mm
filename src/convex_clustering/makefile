use_gpu=0

include ../common.mk

STATIC_DIR=../../examples/convex_clustering
DYNAMIC_DIR=../../bin
#ARCHIVES=/home/garyc/Downloads/gsl-1.16/.libs/libgsl.a /home/garyc/Downloads/gsl-1.16/cblas/.libs/libgslcblas.a 
ARCHIVES=/usr/lib64/libgsl.a /usr/lib64/libgslcblas.a /home/garykche/cuda_sdk/OpenCL/common/lib/liboclUtil_x86_64.a /home/garykche/shared_objects/libocl_wrapper.a
OBJS=cl_templates.o proxmap.o proxmap_opencl.o cluster.o cluster_opencl.o main.o
BINDIR=../../bin


compile: $(OBJS)
	$(LINKER)  -o $(BINDIR)/convexcluster $(LINKFLAGS) $(OBJS)

static: $(OBJS)
	$(LINKER)  -static -o $(STATIC_DIR)/convexcluster $(OBJS) $(ARCHIVES)
	$(LINKER)  -static -o $(STATIC_DIR)/knn_weights knn_weights.cpp
	$(LINKER)  -static -o $(STATIC_DIR)/distance distance.cpp


proxmap.o: ../proxmap.cpp ../proxmap.hpp
	$(CC) $(CFLAGS) $(PREPROC) -c $<

proxmap_opencl.o: ../proxmap_opencl.cpp ../cl_templates.cpp ../cl_constants.h ../proxmap.hpp
	$(CC) $(CFLAGS) $(PREPROC) -c $<

main.o: main.cpp ../proxmap.hpp
	$(CC) $(CFLAGS) $(PREPROC) -c $<

cluster.o: cluster.cpp cluster.hpp
	$(CC) $(CFLAGS) $(PREPROC) -c $<

cluster_opencl.o: cluster_opencl.cpp cluster.hpp
	$(CC) $(CFLAGS) $(PREPROC) -c $<

cl_templates.o: ../cl_templates.cpp ../cl_templates.hpp ../cl_constants.h
	$(CC) $(CFLAGS) $(PREPROC) -c $<


distance: distance.cpp 
	$(LINKER) -o $(BINDIR)/distance $(LINK_FLAGS) $(C_FLAGS) $(PREPROC) $<

knn_weights: knn_weights.cpp 
	$(LINKER) -o $(BINDIR)/knn_weights $(LINK_FLAGS) $(C_FLAGS) $(PREPROC) $<

project_hclust: project_hclust.cpp 
	$(LINKER) -o $(BINDIR)/project_hclust $(LINK_FLAGS) $(C_FLAGS) $(PREPROC) $<

clean :
	rm -fr *.o $(PROGRAM) 
