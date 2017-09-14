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
	vector<double> Z2 = matrix_add(matrix_multiply(W2, A), b2);
	vector<double> A2 = apply_sigmoid(Z2);
	
	//Cost function step
	double cost = calculate_mean_square_sum(A2, Y_input);
	
	
	return 0;
}