
all:
	g++ -std=c++17 -O2 -fopenmp -c qd_arrayant_interpolate.cpp -o interp.o
	g++ -std=c++17 -O2 -mavx2 -mfma -fopenmp -c qd_arrayant_interpolate_avx2.cpp -o interp_avx2.o
	g++ -std=c++17 -O2 -mavx2 -mfma -fopenmp benchmark_interp.cpp interp.o interp_avx2.o -o bench_interp

clean:
	- rm *.o
	- rm bench_interp