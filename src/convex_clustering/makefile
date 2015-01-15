use_gpu=0

include ../common.mk

OBJS=proxmap.o cluster.o cluster_opencl.o main.o
BINDIR=../../bin

CONVEXCLUSTER=$(BINDIR)/convexcluster

compile: $(OBJS)
	$(LINKER) -o $(CONVEXCLUSTER) $(LINKFLAGS) $(OBJS)


proxmap.o: ../proxmap.cpp ../proxmap.hpp
	$(CC) $(CFLAGS) $(PREPROC) -c $<

main.o: main.cpp ../proxmap.hpp
	$(CC) $(CFLAGS) $(PREPROC) -c $<

cluster.o: cluster.cpp cluster.hpp
	$(CC) $(CFLAGS) $(PREPROC) -c $<

cluster_opencl.o: cluster_opencl.cpp cluster.hpp
	$(CC) $(CFLAGS) $(PREPROC) -c $<

distance: distance.cpp 
	$(LINKER) -o $(BINDIR)/distance $(LINK_FLAGS) $(C_FLAGS) $(PREPROC) $<

knn_weights: knn_weights.cpp 
	$(LINKER) -o $(BINDIR)/knn_weights $(LINK_FLAGS) $(C_FLAGS) $(PREPROC) $<

project_hclust: project_hclust.cpp 
	$(LINKER) -o $(BINDIR)/project_hclust $(LINK_FLAGS) $(C_FLAGS) $(PREPROC) $<

clean :
	rm -fr *.o $(PROGRAM) 
