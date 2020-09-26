all:
	nvcc src/main.cu -o test
	nvcc src/bitonic.cu -o test_bitonic