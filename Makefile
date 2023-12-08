wavelet: wavelet.cu utils.cpp
	nvcc -c utils.cpp
	nvcc -c wavelet.cu
	nvcc utils.o wavelet.o -o wavelet

conv: conv.cu
	nvcc $< -o $@

run: matrix
	./$< 10 1 > temp.txt

saxpy: saxpy.cu
	nvcc $< -o $@

probe: probe.cu
	nvcc $< -o $@

sum: sum.cu
	nvcc $< -o $@

matrix: matrix.cu
	nvcc $< -o $@

dotprod: dotprod.cu
	nvcc $< -o $@
