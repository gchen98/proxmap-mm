include ../common.mk

OBJS=regression.o
compile: $(OBJS)

regression.o: regression.cpp regression.hpp
	$(CC) $(CFLAGS) $(PREPROC) -o regression.o -c $<
clean :
	rm -f *.o
