#include <iostream>
//#include "math_util.h"
#include "network_builder.h"
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
    test.print_weight();
    test.print_bias();
    initialize_param(&test);
    test.print_weight();
    test.print_bias();

    return 0;
}
