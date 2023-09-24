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
