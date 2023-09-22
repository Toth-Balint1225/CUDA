run: saxpy
	./$<

saxpy: saxpy.cu
	nvcc $< -o $@

sum: sum.cu
	nvcc $< -o $@

matrix: matrix.cu
	nvcc $< -o $@

dotprod: dotprod.cu
	nvcc $< -o $@