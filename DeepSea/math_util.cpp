#include <blaze/Math.h>

#define DEBUG 1
using blaze::DynamicVector;


DynamicVector<double> add(DynamicVector<double> A, DynamicVector<double> B){

	if(A.rows()==B.rows() && A.columns()==B.columns()){
		return A+B;
	}
	else{
#ifdef DEBUG
		std::cout << "WARN: dimention mismatch" << std::endl;;
#endif
		return A;
	}

}

DynamicVector<double> subs(DynamicVector<double> A, DynamicVector<double> B){

	if(A.rows()==B.rows() && A.columns()==B.columns()){
		return A-B;
	}
	else{
#ifdef DEBUG
		std::cout << "WARN: dimention mismatch" << std::endl;;
#endif
		return A;
	}

}

DynamicVector<double> mul(DynamicVector<double> A, DynamicVector<double> B){

	if(A.columns()==B.rows()){
		return A*B;
	}
	else{
#ifdef DEBUG
		std::cout << "WARN: dimention mismatch" << std::endl;;
#endif
		return A;
	}

}

DynamicVector<double> mul_elem(DynamicVector<double> A, DynamicVector<double> B){

	if(A.rows()==B.rows() && A.columns()==B.columns()){
		return A%B;
	}
	else{
#ifdef DEBUG
		std::cout << "WARN: dimention mismatch" << std::endl;;
#endif
		return A;
	}

}

DynamicVector<double> scaler_mul_elem(DynamicVector<double> A, double m){

	return A*m;

}
