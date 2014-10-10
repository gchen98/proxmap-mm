include ../common.mk

OBJS=regression.o 


compile: $(OBJS)

regression.o: regression.cpp regression.hpp cl_headers.h
	$(CC) $(CFLAGS) $(PREPROC) -o regression.o -c $<

regression_with_theta.o: regression_with_theta.cpp regression_with_theta.hpp
	$(CC) $(CFLAGS) $(PREPROC) -o regression_with_theta.o -c $<
clean :
	rm -f *.o
