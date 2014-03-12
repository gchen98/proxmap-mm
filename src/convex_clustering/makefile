include ../common.mk

OBJS=cluster.o cluster_opencl.o
BINDIR=../../bin

compile: $(OBJS)

cluster.o: cluster.cpp cluster.hpp ../cl_constants.h
	$(CC) $(CFLAGS) $(PREPROC) -c $<

cluster_opencl.o: cluster_opencl.cpp cluster.hpp ../cl_constants.h
	$(CC) $(CFLAGS) $(PREPROC) -c $<

distance: distance.cpp 
	$(LINKER) -o $(BINDIR)/distance $(LINK_FLAGS) $(C_FLAGS) $(PREPROC) $<

knn_weights: knn_weights.cpp 
	$(LINKER) -o $(BINDIR)/knn_weights $(LINK_FLAGS) $(C_FLAGS) $(PREPROC) $<

clean :
	rm -fr *.o $(PROGRAM) 
