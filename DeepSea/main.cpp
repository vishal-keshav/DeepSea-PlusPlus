#include <iostream>
#include "math_util.h"
#include <blaze/Math.h>

int main()
{
    blaze::DynamicMatrix<double> mat1(3,3,1);
    blaze::DynamicMatrix<double> mat2(3,3,2);
    blaze::DynamicMatrix<double> mat3 = add(mat1, mat2);

    std::cout << mat3 << std::endl;

    return 0;
}
