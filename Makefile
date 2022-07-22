SRCS:=src/main.cu src/sparse_mat.cu
OBJS:=$(SRCS:%.cu=%.o)
NVCCFLAGS:=-O2 -std=c++17 -Xcompiler='-march=native -Wall -Wextra -fopenmp'
NVCC_LINK_FLAGS:=-lcusparse_static -lculibos

train: $(OBJS)
	nvcc -o $@ $(NVCCFLAGS) $(NVCC_LINK_FLAGS) $^

%.o: %.cu
	nvcc -c -o $@ $(NVCCFLAGS) $<

main.o: src/bitboard.hpp src/sparse_mat.hpp
sparse_mat.o: src/bitboard.hpp src/sparse_mat.hpp

.PHONY: clean
clean: $(OBJS) train
	-rm $^
