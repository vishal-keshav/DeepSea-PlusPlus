#include <iostream>
#include "math_util.h"
#include <blaze/Math.h>

#define MATRIX blaze::DynamicMatrix<double>
#define print std::cout<<
int main()
{
    MATRIX mat1(3,3,1);
    MATRIX mat2(3,3,2);
    MATRIX mat3 = add(mat1, mat2);
    print mat3 ;

    MATRIX vec1(3,1,5);
    MATRIX mat5 = add(mat1, vec1);
    print mat5 ;

    MATRIX vec2(4,1,9);
    MATRIX scalar(1,1,5);
    MATRIX vec3 = subs(vec2,scalar);
    print vec3 ;

    return 0;
}
