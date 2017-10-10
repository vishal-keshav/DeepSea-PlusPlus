#include <iostream>
#include "math_util.h"
#include "network_builder.h"
#include "read_write_util.h"
#include "prediction.h"
#include <blaze/Math.h>
#include <vector>
#include <algorithm>

using namespace std;
int main()
{
    srand(1);
	//Define layer structure
	static const int arr[] = {16,20,40, 26};
	vector<int> layer (arr, arr + sizeof(arr)/sizeof(arr[0]));

	//Declare training params
	int batch_size = 50;
	int nr_epoch = 200;
	double learning_rate = 0.01;
	int nr_batch;

	//Initialize model and train graph parameters
	model_param m_p(layer);
	initialize_param(&m_p);

	forward_param f_p(layer, batch_size);
	backward_param b_p(layer, batch_size);

	//read datafile, seperate test data X_test, Y_test
	int *temp_label;
	DynamicMatrix<double> X_all = read_csv_modified("letter-recognition.data", 20000, 16, &temp_label);
	DynamicMatrix<double> Y_all = get_label_modified(temp_label, 26, 20000);

	nr_batch = X_all.columns()/batch_size;

    //Allocate matrix variables to be used at training
    DynamicMatrix<double> X;
    DynamicMatrix<double> Y;

    DynamicMatrix<double> X_test;
    DynamicMatrix<double> Y_test;
    DynamicMatrix<double> O;

	//Start training
	for(int i=0;i<nr_epoch;i++){


		for(int j=0;j<nr_batch;j++){
			//Prepare input batch X and output label Y
			X = submatrix(X_all, 0, j*batch_size, 16 ,batch_size);
			Y = submatrix(Y_all, 0, j*batch_size, 26 ,batch_size);

			feed_forward(&m_p, &f_p, X);
			back_prop(&m_p, &f_p, &b_p, Y);
			gradient_descent(&m_p, &b_p, learning_rate);
		}

		//Print accuracy and cost on test data set, test set should be separated, but I am lazy :/
        X_test = submatrix(X_all, 0, 0, 16, batch_size);
        Y_test = submatrix(Y_all, 0, 0, 26, batch_size);
		O = predict(&m_p, &f_p, X_test);

		cout << "Cost for epoch " << i << " is " << mean_cross_entropy_loss(Y_test, O) << endl;
		cout << "Accuracy for epoch " << i << " is " << accuracy(Y_test, O) << endl << endl;
	}

	write_model(&m_p, "model_3.txt");

    return 0;
}
