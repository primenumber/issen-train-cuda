SRCS:=src/main.cu src/sparse_mat.cpp
NVCCOPT:=-O2 -lcusparse -Xcompiler='-march=native -Wall -Wextra'

train: $(SRCS)
	nvcc -o $@ $(NVCCOPT) $^
