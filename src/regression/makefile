use_mpi=1
use_gpu=1

include ../common.mk

CFLAGS+=$(GCHEN_INC_FLAGS)
LINKFLAGS+=$(GCHEN_LIB_FLAGS)

OBJS=proxmap.o param_split.o quadratic.o block_descent.o proxmap_opencl.o cl_templates.o main.o
BINDIR=../../bin

L0=$(BINDIR)/L0

compile: $(OBJS)
	$(LINKER) -o $(L0) $(LINKFLAGS) $(OBJS)

proxmap.o: ../proxmap.cpp ../proxmap.hpp
	$(CC) $(CFLAGS) $(PREPROC) -c $<

proxmap_opencl.o: ../proxmap_opencl.cpp ../cl_templates.cpp ../cl_constants.h ../proxmap.hpp
	$(CC) $(CFLAGS) $(PREPROC) -c $<

cl_templates.o: ../cl_templates.cpp ../cl_templates.hpp ../cl_constants.h
	$(CC) $(CFLAGS) $(PREPROC) -c $<


main.o: main.cpp ../proxmap.hpp
	$(CC) $(CFLAGS) $(PREPROC) -c $<

block_descent.o: block_descent.cpp block_descent.hpp
	$(CC) $(CFLAGS) $(PREPROC) -c $<

quadratic.o: quadratic.cpp quadratic.hpp
	$(CC) $(CFLAGS) $(PREPROC) -c $<

param_split.o: param_split.cpp param_split.hpp
	$(CC) $(CFLAGS) $(PREPROC) -c $<

param_split_with_theta.o: param_split_with_theta.cpp param_split_with_theta.hpp
	$(CC) $(CFLAGS) $(PREPROC) -o param_split_with_theta.o -c $<
clean :
	rm -f *.o
