include ../../Makefile
XCGD_INCLUDE=$(dir $(lastword $(MAKEFILE_LIST)))/include

default:
	${CXX} -std=c++17 -g -I${A2D_INCLUDE} -I${XCGD_INCLUDE} jacobian_vector_product.cpp -o jacobian_vector_product

clean:
	rm -rf jacobian_vector_product{,.dSYM}

