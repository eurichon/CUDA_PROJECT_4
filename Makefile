CC=nvcc
HEADER=./inc
SRC=./src
OBJ=./obj
LIB=./lib
TESTS=./tests

all: clean build_bitonic


build_bitonic:
	@echo "Building Bitonic example"
	nvcc $(SRC)/bitonic.cu $(SRC)/bitonic_test.cu -o $(TESTS)/test_bitonic -I$(HEADER) 

build_vp_tree:
	@echo "Building Object files"	
	nvcc -c -o $(OBJ)/bitonic.o $(SRC)/bitonic.cu -I$(HEADER) 
	nvcc -c -o $(OBJ)/vptree.o $(SRC)/vptree.cu -I$(HEADER)
	nvcc -c -o $(OBJ)/distance.o $(SRC)/distance.cu -I$(HEADER)

	@echo "Creating Static libs"
	ar -cvq $(LIB)/bitonic.a $(OBJ)/bitonic.o
	ar -cvq $(LIB)/vptree.a $(OBJ)/vptree.o
	ar -cvq $(LIB)/distance.a $(OBJ)/distance.o

	@echo "Link & Compile Project"
	nvcc -o $(TESTS)/vptree $(SRC)/vptree_main.cu $(LIB)/* -I$(HEADER) 

	@echo "Done"

clean:
	@echo "Removing old copies"
	$(RM) count $(LIB)/*.a $(OBJ)/*.o $(TESTS)/* 