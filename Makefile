CC=nvcc
HEADER=./inc
SRC=./src
TESTS=./tests

all: 
	$(CC) -o $(TESTS)/test_bitonic $(SRC)/bitonic_test.cu $(SRC)/bitonic.cu -I$(HEADER)