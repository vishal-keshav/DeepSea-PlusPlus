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
