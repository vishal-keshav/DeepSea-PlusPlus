/* Extended class version of segment tree implementation
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

#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <random>
#include <algorithm>

#define DEBUG 1

using namespace std;

vector<vector<double> > randomized_weight(int row, int col){
	default_random_engine generator;
	normal_distribution<double> distribution(0.0,1.0);
	vector<vector<double> > ret(row, vector<int>(col));
	for(int i=0;i<row;i++){
		for(int j=0;j<col;j++){
			ret[i][j] = distribution(generator);
		}
	}
	return ret;
}

vector<vector<double> > zeroed_weight(int row, int col){
	vector<vector<double> > ret(row, vector<double>(col,0));
	return ret;
}

vector<double> matrix_multiply(vector<vector<double> > A, vector<double> B){
	vector<double> ret(A.size(),0);
	for(int i=0;i<A.size();i++){
		for(int j=0;j<A[0].size();j++){
			ret[i]+=A[i][j]*B[j];
		}
	}
	return ret;
}

vector<double> matrix_add(vector<double> A, vector<double> B){
	for(int i=0;i<A.size();i++){
		A[i]+= B[i];
	}
	return A;
}

vector<double> apply_relu(vector<double> A){
	for(int i=0;i<A.size();i++){
		A[i] = max(A[i],0);
	}
	return A;
}

vector<double> apply_sigmoid(vector<double> A){
	for(int i=0;i<A.size();i++){
		A[i] = exp(A[i])/(1+exp(A[i]));
	}
	return A;
}

double calculate_mean_square_sum(vector<double> A, vector<double> B){
	int nr_elem = A.size();
	double ret = 0;
	for(int i=0;i<nr_elem;i++){
		ret+= ((A[i] - B[i])*(A[i] - B[i]))/2;
	}
	ret = ret/nr_elem;
	return ret;
}

vector<double>  derivative_mean_square_sum(vector<double> A, vector<double> B){
	return matrix_substract_elem(A,B);
}

vector<double> matrix_multiply_elem(vector<double> A, vector<double> B){
	int nr_elem = A.size();
	for(int i=0;i<nr_elem;i++){
		A[i] = A[i]*B[i];
	}
	return A;
}

vector<double> matrix_substract_elem(vector<double> A, vector<double> B){
	int nr_elem = A.size();
	for(int i=0;i<nr_elem;i++){
		A[i] = A[i] - B[i];
	}
	return A;
}

vector<double> matrix_multiply_divide(vector<double> A, vector<double> B, int m){
	vector<double> ret = matrix_multiply_elem(A,B);
	for(int i=0;i<ret.size();i++){
		ret[i] = ret[i]/((double))m;
	}
	return ret;
}

vector<vector<double> > matrix_multiply_scalar(vector<vector<double> > A, double a){
	for(int i=0;i<A.size();i++){
		for(int j=0;j<A[0].size();j++){
			A[i][j] = A[i][j]*a;
		}
	}
	return A;
}

vector<double> relu_derivative(vector<double> A){
	for(int i=0;i<A.size();i++){
		if(A[i]<=0){
			A[i] = 0;
		}
		else{
			A[i] = 1;
		}
	}
	return A;
}

int main(){
	
	//Initialize layer description
	int arr[] = { 3, 5, 1};
	vector<int> layer_des;
	layer_des.insert(layer_des.begin(), arr, &arr[sizeof(arr)/ sizeof(*arr)]);
	
	//Initialize randomized weights according to defined layer description
	vector<vector<double> > W1 = randomized_weight(5,3);
	vector<vector<double> > b1 = zeroed_weight(5,1);
	vector<vector<double> > W2 = randomized_weight(1,5);
	vector<vector<double> > b2 = zeroed_weight(1,1);
	
	//Define input and output
#ifdef DEBUG
	int nr_train = 3;
	double temp_input[] = {30, 54, 12};	
	vector<double> X_input;
	X_input.insert(X_input.begin(), temp_input, &temp_input[sizeof(temp_input)/ sizeof(*temp_input)]);
	double temp_output[] = {15, 27, 6};
	vector<double> Y_input;
	Y_input.insert(Y_input.begin(),temp_output, &temp_output[sizeof(temp_output)/ sizeof(*temp_output)]);
#endif

	//Forward propagation
	vector<double> Z1 = matrix_add(matrix_multiply(W1, X_input), b1);
	vector<double> A1 = apply_relu(Z1);
	vector<double> Z2 = matrix_add(matrix_multiply(W2, A1), b2);
	vector<double> A2 = apply_sigmoid(Z2);
	
	vector<double> I(A2.size(),1);
	double learning_rate = 0.2;
	//Cost function step
	double cost = calculate_mean_square_sum(A2, Y_input);
	
	//Backpropagation, based on assumed activations
	vector<double> dA2 = derivative_mean_square_sum(A2, Y_input);
	vector<double> dZ2 = matrix_multiply_elem(dA2, matrix_multiply_elem(A2,matrix_substract_elem(I, A2)));
	vector<double> dW2 = matrix_multiply_divide(dZ2, A1, nr_train);
	vector<double> db2 = matrix_sum_divide(dZ2, nr_train);
	
	//Correct weights and bias
	W2 = matrix_sub(W2, matrix_multiply_scalar(dW2, learning_rate));
	b2 = b2 - learning_rate*b2;
	
	vector<double> dA1 = matrix_multiply(W2, dZ2);
	vector<double> dZ1 = matrix_multiply_scalar(dA1,relu_derivative(Z1));
	vector<double> dW1 = matrix_multiply_divide(dZ1, X_input, nr_train);
	vector<double> db1 = matrix_sum_divide(dZ1, nr_train);
	
	W1 = matrix_sub(W1, matrix_multiply_scalar(dW1, learning_rate));
	b1 = b1 - learning_rate*b1;
	
	
	return 0;
}