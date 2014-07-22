include ../common.mk

L0_reg=../../bin/l0_reg
OBJS=L0_reg.o ../proxmap.o

l0_reg: $(OBJS)
	$(LINKER) -o $(L0_reg) $(LINKFLAGS) $(OBJS)

L0_reg.o: L0_reg.cpp L0_reg.hpp
	$(CC) $(CFLAGS) $(PREPROC) -o L0_reg.o -c $<

clean :
	rm -f *.o
