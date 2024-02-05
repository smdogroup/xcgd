
default:
	${CXX} -std=c++11 -g test.cpp -o test
	${CXX} -std=c++11 -g derivatives_verification.cpp -o derivatives_verification