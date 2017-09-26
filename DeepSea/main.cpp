#include <iostream>
#include "math_util.h"
#include "network_builder.h"
#include "read_write_util.h"
#include <blaze/Math.h>
#include <vector>
//#define MATRIX blaze::DynamicMatrix<double>
//#define print std::cout<<

using namespace std;
int main()
{
    /*MATRIX mat1(3,3,1);
    MATRIX mat2(3,3,2);
    MATRIX mat3 = add(mat1, mat2);
    print mat3 ;

    MATRIX vec1(3,1,5);
    MATRIX mat5 = add(mat1, vec1);
    print mat5 ;

    MATRIX vec2(4,1,9);
    MATRIX scalar(1,1,5);
    MATRIX vec3 = subs(vec2,scalar);
    print vec3 ;*/
    static const int arr[] = {3,5,4};
    vector<int> layer (arr, arr + sizeof(arr) / sizeof(arr[0]) );
    model_param test(layer);
    //test.print_weight();
    //test.print_bias();
    initialize_param(&test);
    //test.print_weight();
    //test.print_bias();
    //cout << apply_relu(test.W[0]) << endl;
    //cout << apply_sigmoid(test.W[1]) << endl;

    blaze::DynamicMatrix<double> X = read_csv("sample_data1.csv", 3, 3);
    blaze::DynamicMatrix<double> Y = read_csv("sample_data2.csv", 4, 3);
    //cout << apply_log(D) << endl;
    //cout << apply_softmax(D) << endl;
    //cout << mean_cross_entropy_loss(D, apply_softmax(D)) << endl;
    forward_param f_test(layer,3);
    f_test.print_linear();
    cout << endl;
    f_test.print_activated();
    cout << endl;
    feed_forward(&test, &f_test, X);
    f_test.print_linear();
    cout << endl;
    f_test.print_activated();
    backward_param b_test(layer,3);

    back_prop(&test,&f_test,&b_test,Y);
    gradient_descent(&test, &b_test, 0.2);
    //write_model(&test, "model_1.txt");

    return 0;
}
