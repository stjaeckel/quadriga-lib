# Fuzz testing:
# - Compile Quadriga-Lib with the "afl-g++" compiler

# Compilers
CC    = afl-g++

# External libraries
ARMA_H      = ../../external/armadillo-12.6.3/include
QDLIB_H     = /sjc/quadriga-lib/include
QDLIB_A 	= /sjc/quadriga-lib/lib/quadriga_lib.a

# Configurations
CCFLAGS     = -std=c++17 -O3 -fPIC -Wall -Wextra -Wpedantic -Wconversion

# Sources
src     	= $(wildcard src/*.cpp)
bin: 		$(src:src/%.cpp=bin/%)

all: 	dirs   bin

dirs:
	mkdir -p bin
	mkdir -p afl_out

bin/%:	src/%.cpp
	$(CC) $(CCFLAGS) -fopenmp $^ -o $@ $(QDLIB_A) -I $(ARMA_H) -I $(QDLIB_H) -lgomp

fuzz_qdant_read:   bin/afl_qdant_read   dirs
	afl-fuzz -m none -i ./qdant_samples -o afl_out -s 123 -D -M master -- $^ -i @@

