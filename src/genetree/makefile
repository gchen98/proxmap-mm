include ../common.mk

OBJS=genetree.o 
BINDIR=../../bin

compile: $(OBJS)

genetree.o: genetree.cpp genetree.hpp 
	$(CC) $(CFLAGS) $(PREPROC) -c $<


clean :
	rm -fr *.o $(PROGRAM) 
