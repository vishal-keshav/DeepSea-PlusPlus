#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <random>

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
#endif
	//Forward propagation
	vector<double> Z1 = matrix_add(matrix_multiply(W1, X_input), b1);
	vector<double> A1 = apply_relu(Z1);
	vector<double> Z2 = matrix_add(matrix_multiply(W2, A), b2);
	vector<double> A2 = apply_sigmoid(Z2);
	
	
	
	
	return 0;
}