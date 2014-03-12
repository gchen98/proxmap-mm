include common.mk

DIRS=convex_clustering regression
SUB_OBJS=convex_clustering/*.o regression/*.o

OBJS=proxmap.o cl_templates.o
PROGRAM=../bin/proxmap

proxmap::
	for dir in $(DIRS); do(cd $$dir; make $1 || exit 1) || exit 1;    done; 

proxmap:: $(OBJS)
	$(LINKER) -o $(PROGRAM) $(LINKFLAGS) $(OBJS) $(SUB_OBJS)

proxmap.o: proxmap.cpp proxmap.hpp cl_constants.h
	$(CC) $(CFLAGS) $(PREPROC) -o proxmap.o -c $<

cl_templates.o: cl_templates.cpp cl_templates.hpp cl_constants.h
	$(CC) $(CFLAGS) $(PREPROC) -o cl_templates.o -c $<

clean :
	for dir in $(DIRS); do(cd $$dir; make clean || exit 1) || exit 1; done
	rm -f *.o $(PROGRAM) $(OBJS)
