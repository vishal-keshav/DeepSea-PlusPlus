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
	static const int arr[] = {3,4,2};
	vector<int> layer (arr, arr + sizeof(arr)/sizeof(arr[0]));

	//Declare training params
	int batch_size = 3;
	int nr_epoch = 1;
	double learning_rate = 0.2;
	//int nr_batch;

	//Intialize model and train graph parameters
	//model_param m_p(layer);
	//initialize_param(&m_p);
	//m_p.print_weight();
	//m_p.print_bias();
	model_param m_p = read_model("model_1.txt");
    //m_p.print_weight();
    //m_p.print_bias();
	//initialize_param(&m_p);

	forward_param f_p(layer, batch_size);
	backward_param b_p(layer, batch_size);

	//read datafile, seperate test data X_test, Y_test
	//int *temp_label;
	//DynamicMatrix<double> X_all = read_csv_modified("letter-recognition.data", 20000, 16, &temp_label);
	DynamicMatrix<double> X_all = read_csv("sample_data1.csv", 3, 3);
	DynamicMatrix<double> Y_all = read_csv("sample_data2.csv", 2, 3);

	//cout << X_all << endl;
	//cout << Y_all << endl;

	//nr_batch = X_all.columns()/batch_size;

	//Start training
	for(int i=0;i<nr_epoch;i++){
		//Allocate matrix space
		DynamicMatrix<double> X;
		DynamicMatrix<double> Y;

		DynamicMatrix<double> X_test;
		DynamicMatrix<double> Y_test;
		DynamicMatrix<double> O;

		for(int j=0;j<1;j++){
			//Preapare input batch X and output label Y
			//X = submatrix(X_all, 0, j*batch_size, 16 ,batch_size);
			//Y = submatrix(Y_all, 0, j*batch_size, 26 ,batch_size);

			//cout << X << endl;
			//cout << Y << endl;

			feed_forward(&m_p, &f_p, X_all);
			back_prop(&m_p, &f_p, &b_p, Y_all);
			gradient_descent(&m_p, &b_p, learning_rate);
			//m_p.print_weight();
		}
		//f_p.print_linear();
		//f_p.print_activated();
		//Print accuracy and cost on test data set
        //X_test = submatrix(X_all, 0, 0, 16, batch_size);
        //Y_test = submatrix(Y_all, 0, 0, 26, batch_size);
		O = predict(&m_p, &f_p, X_all);
		//cout << Y_test << endl;
		//cout << O << endl;
		//m_p.print_weight();
		cout << "Cost for epoch " << i << " is " << mean_cross_entropy_loss(Y_all, O) << endl;
		cout << "Accuracy for epoch " << i << " is " << accuracy(Y_all, O) << endl << endl;
	}

	//write_model(&m_p, "model_2.txt");

    return 0;
}
