#include <blaze/Math.h>

#define DEBUG 1
using blaze::DynamicMatrix; //Every vector is a matrix (or can be treated as)

//Broadcasting enabled
DynamicMatrix<double> add(DynamicMatrix<double> A, DynamicMatrix<double> B){

	if(A.rows()==B.rows() && A.columns()==B.columns()){
		return A+B;
	}
	else if(A.rows() > B.rows() && B.rows()==1 && A.columns()==1 && B.columns()==1){
		DynamicMatrix<double> C(A.rows(),1UL, B[0][0]));
		return A+C;
	}
	else if(A.rows() == B.rows() && A.columns() > B.columns() && B.columns()==1){
		DynamicMatrix<double> C(A.rows(), A.columns(), B);
		return A+C;
	}
	else{
#ifdef DEBUG
		std::cout << "WARN: dimention mismatch" << std::endl;;
#endif
		return A;
	}

}

//Broadcasting enabled
DynamicMatrix<double> subs(DynamicMatrix<double> A, DynamicMatrix<double> B){

	if(A.rows()==B.rows() && A.columns()==B.columns()){
		return A-B;
	}
	else if(A.rows() > B.rows() && B.rows()==1 && A.columns()==1 && B.columns()==1){
		DynamicMatrix<double> C(A.rows(),1UL, B[0][0]));
		return A-C;
	}
	else if(A.rows() == B.rows() && A.columns() > B.columns() && B.columns()==1){
		DynamicMatrix<double> C(A.rows(), A.columns(), B);
		return A-C;
	}
	else{
#ifdef DEBUG
		std::cout << "WARN: dimention mismatch" << std::endl;;
#endif
		return A;
	}

}

DynamicMatrix<double> mul(DynamicMatrix<double> A, DynamicMatrix<double> B){

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

DynamicMatrix<double> mul_elem(DynamicMatrix<double> A, DynamicMatrix<double> B){

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

DynamicMatrix<double> scaler_mul_elem(DynamicMatrix<double> A, double m){

	return A*m;

}
