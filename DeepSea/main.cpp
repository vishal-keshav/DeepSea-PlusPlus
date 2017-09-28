#include <iostream>
#include "math_util.h"
#include "network_builder.h"
#include "read_write_util.h"
#include "prediction.h"
#include <blaze/Math.h>
#include <vector>

using namespace std;
int main()
{
	//Define layer structure
	static const int arr[] = {16, 20, 35, 10, 26};
	vector<int> layer (arr, arr + sizeof(arr)/sizeof(arr[0]));
	
	//Declare training params
	int batch_size = 100;
	int nr_epoch = 10;
	double learning_rate = 0.2;
	
	//Intialize model and train graph parameters
	model_param m_p(layer);
	initialize_param(&m_p);
	
	forward_param f_p(layer, batch_size);
	backward_param b_p(layer, batch_size);
	
	//read datafile, seperate test data X_test, Y_test
	
	//Start training
	for(int i=0;i<nr_epoch;i++){
		nr_batch = ; //Based on read datafile
		for(int j=0;j<nr_batch;j++){
			//Preapare input batch X and output label Y
			feed_forward(&m_p, &f_p, X);
			back_prop(&m_p, &f_p, &b_p, Y);
			gradient_descent(&m_p, &b_p, learning_rate);
		}
		//Print accuracy and cost on test data set
		O = predict(&m_p, &f_p, X_test);
		cout << "Cost for epoch " << i << " is " << mean_cross_entropy_loss(Y_test, O) << endl;
		cout << "Accuracy for epoch " << i << " is " << accuracy(Y_test, O) << endl;
	}
	
	write_model(&m_p, "model_2.txt");
	
    return 0;
}
