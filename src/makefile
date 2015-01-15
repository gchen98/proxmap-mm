use_gpu=1

include common.mk

DIRS=convex_clustering regression genetree
SUB_OBJS=convex_clustering/*.o regression/*.o genetree/*.o

OBJS=proxmap.o proxmap_opencl.o cl_templates.o

PROXMAP_FIT=../bin/proxmap
CROSS_VALIDATION=../bin/cross_validation

all::
	for dir in $(DIRS); do(cd $$dir; make $1 || exit 1) || exit 1;    done; 
all:: proxmap_fit 


proxmap_fit: $(OBJS) proxmap_fit.o
	$(LINKER) -o $(PROXMAP_FIT) $(LINKFLAGS) $(OBJS) proxmap_fit.o $(SUB_OBJS) 

cross_validation: $(OBJS) cross_validation.o
	$(LINKER) -o $(CROSS_VALIDATION) $(LINKFLAGS) $(OBJS) cross_validation.o $(SUB_OBJS)

proxmap_fit.o: proxmap_fit.cpp proxmap.hpp cl_constants.h
	$(CC) $(CFLAGS) $(PREPROC) -o proxmap_fit.o -c $<

cross_validation.o: cross_validation.cpp cross_validation.hpp proxmap.hpp cl_constants.h
	$(CC) $(CFLAGS) $(PREPROC) -o cross_validation.o -c $<

proxmap.o: proxmap.cpp proxmap.hpp cl_constants.h
	$(CC) $(CFLAGS) $(PREPROC) -o proxmap.o -c $<

proxmap_opencl.o: proxmap_opencl.cpp cl_templates.hpp cl_constants.h
	$(CC) $(CFLAGS) $(PREPROC) -o proxmap_opencl.o -c $<

cl_templates.o: cl_templates.cpp cl_templates.hpp cl_constants.h
	$(CC) $(CFLAGS) $(PREPROC) -o cl_templates.o -c $<

clean :
	for dir in $(DIRS); do(cd $$dir; make clean || exit 1) || exit 1; done
	rm -f *.o $(PROGRAM) $(OBJS)
