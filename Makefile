SRCS:=src/main.cu src/sparse_mat.cu
OBJS:=$(SRCS:%.cu=%.o)
NVCCOPT:=-O2 -std=c++17 -lcusparse -Xcompiler='-march=native -Wall -Wextra -fopenmp'

train: $(OBJS)
	nvcc -o $@ $(NVCCOPT) $^

%.o: %.cu
	nvcc -c -o $@ $(NVCCOPT) $<

main.o: bitboard.hpp sparse_mat.hpp
sparse_mat.o: bitboard.hpp sparse_mat.hpp
