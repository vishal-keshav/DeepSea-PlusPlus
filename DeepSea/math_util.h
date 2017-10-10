/* Header for mathematics operation
 * Copyright (C) bulletcross (Vishal Keshav)

 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <blaze/Math.h>
#include <iostream>

#define DEBUG 0
using blaze::DynamicMatrix; //Every vector is a matrix (or can be treated as)

//Broadcasting enabled
DynamicMatrix<double> add(DynamicMatrix<double> A, DynamicMatrix<double> B){

	if(A.rows()==B.rows() && A.columns()==B.columns()){
		return A+B;
	}
	else if(A.rows() > B.rows() && B.rows()==1 && A.columns()==1 && B.columns()==1){
		DynamicMatrix<double> C(A.rows(),1, B(0,0));
		return A+C;
	}
	else if(A.rows() == B.rows() && A.columns() > B.columns() && B.columns()==1){
		DynamicMatrix<double> C(A.rows(), A.columns());
		for(int i=0;i<C.rows(); i++){
            for(int j=0;j<C.columns();j++){
                C(i,j) = B(i,0);
            }
		}
		return A+C;
	}
	else{
#ifdef DEBUG
		std::cout << "WARN: dimention mismatch" << std::endl;
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
		DynamicMatrix<double> C(A.rows(),1, B(0,0));
		return A-C;
	}
	else if(A.rows() == B.rows() && A.columns() > B.columns() && B.columns()==1){
		DynamicMatrix<double> C(A.rows(), A.columns());
		for(int i=0;i<C.rows(); i++){
            for(int j=0;j<C.columns();j++){
                C(i,j) = B(i,0);
            }
		}
		return A-C;
	}
	else{
#ifdef DEBUG
		std::cout << "WARN: dimention mismatch" << std::endl;
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
		std::cout << "WARN: dimention mismatch" << std::endl;
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
		std::cout << "WARN: dimention mismatch" << std::endl;
#endif
		return A;
	}
}

DynamicMatrix<double> scaler_mul_elem(DynamicMatrix<double> A, double m){

	return A*m;

}

DynamicMatrix<double> apply_relu(DynamicMatrix<double> A){
	A = map(A, [](double elem) {return (elem>0?elem:0);});
	return A;
}

DynamicMatrix<double> apply_sigmoid(DynamicMatrix<double> A){
	A = map(A, [](double elem) {return (1.0/(1+exp(elem)));});
	return A;
}

DynamicMatrix<double> apply_log(DynamicMatrix<double> A){
	A = map(A, [](double elem) {return log(elem);});
	return A;
}

//Numerically stable implementation of softmax
DynamicMatrix<double> apply_softmax(DynamicMatrix<double> A){
	//softmax(vec) = exp(v_i)/sum(exp(v_i)) over all i
	//Considering each column represent one set of input
	DynamicMatrix<double> B = A;
	double max_elem, sum_elem;
	for(int i=0;i<B.columns();i++){
        max_elem = max(submatrix(B, 0, i, B.rows() ,1));
        for(int j=0;j<B.rows();j++){
            if(max_elem<B(j,i)){
                max_elem = B(j,i);
            }
        }
        for(int j=0;j<B.rows();j++){
            B(j,i) = exp(B(j,i) - max_elem);
        }
        sum_elem = 0;
        for(int j=0;j<B.rows();j++){
            sum_elem += B(j,i);
        }
        for(int j=0;j<B.rows();j++){
            B(j,i) = B(j,i)/sum_elem;
        }
	}
	return B;
}
//TODO: Implement numerically stable loss calculation function
double mean_cross_entropy_loss(DynamicMatrix<double> hot, DynamicMatrix<double> soft){
	double ret = 0.0;
#ifdef DEBUG
	if(hot.rows()!=soft.rows() || hot.columns()!=soft.columns()){
		std::cout << "WARN: dimention mismatch" << std::endl;
	}
#endif
    DynamicMatrix<double> soft_log = apply_log(soft);
	DynamicMatrix<double> loss(1, hot.columns());
	for(int i=0;i<hot.columns();i++){
		double temp = 0;
		for(int j=0;j<hot.rows();j++){
			temp = temp + hot(j,i)*soft_log(j,i);
		}
		loss(0,i) = -temp;
		ret = ret + loss(0,i);
	}
	ret = ret/hot.columns();
	return ret;
}

DynamicMatrix<double> derivative_cross_entropy_softmax(DynamicMatrix<double> hot, DynamicMatrix<double> soft){
    return subs(soft, hot);
}

DynamicMatrix<double> derivative_relu(DynamicMatrix<double> A){
	A = map(A, [](double elem) {return (elem<0?0:1);});
	return A;
}
