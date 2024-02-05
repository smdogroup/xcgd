A2D_INCLUDE=${HOME}/git/a2d/include

default:
	${CXX} -std=c++17 -g -DUSE_A2D -I${A2D_INCLUDE} test.cpp -o test
	${CXX} -std=c++17 -g -DUSE_A2D -I${A2D_INCLUDE} derivatives_verification.cpp -o derivatives_verification