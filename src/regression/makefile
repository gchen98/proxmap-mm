include ../common.mk

OBJS=param_split.o quadratic.o


compile: $(OBJS)

quadratic.o: quadratic.cpp quadratic.hpp
	$(CC) $(CFLAGS) $(PREPROC) -o quadratic.o -c $<

param_split.o: param_split.cpp param_split.hpp
	$(CC) $(CFLAGS) $(PREPROC) -o param_split.o -c $<

param_split_with_theta.o: param_split_with_theta.cpp param_split_with_theta.hpp
	$(CC) $(CFLAGS) $(PREPROC) -o param_split_with_theta.o -c $<
clean :
	rm -f *.o
