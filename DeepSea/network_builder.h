/* Header for network builder
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

#include "math_util.h"
#include <blaze/Math.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <random>

#define DEBUG 0

using namespace std;

class model_param{
	public:
        int nr_layer;
		vector<int> layer_info;

		vector<DynamicMatrix<double> > W;
		vector<DynamicMatrix<double> > b;
		model_param(vector<int > vec){
			nr_layer = vec.size();
			layer_info = vec;
			W.resize(nr_layer-1);
			b.resize(nr_layer-1);
			for(int i=0;i<nr_layer-1;i++){
				W[i].resize(vec[i+1],vec[i]);
				b[i].resize(vec[i+1],1);
			}
		}
		void print_weight(void){
			for(int i=0;i<nr_layer-1;i++){
				std::cout << W[i];
			}
		}
		void print_bias(void){
			for(int i=0;i<nr_layer-1;i++){
				std::cout << b[i];
			}
		}
};

class forward_param{
	public:
        int nr_layer;
		int batch_size;
		vector<int> layer_info;

		vector<DynamicMatrix<double> > Z;
		vector<DynamicMatrix<double> > A;
		forward_param(vector<int> vec, int b_size){
			nr_layer = vec.size();
			batch_size = b_size;
			layer_info = vec;
			Z.resize(nr_layer);
			A.resize(nr_layer);
			for(int i=0;i<nr_layer;i++){
				Z[i].resize(vec[i], b_size);
				A[i].resize(vec[i], b_size);
			}
		}
		void print_linear(void){
			for(int i=0;i<nr_layer;i++){
				std::cout << Z[i];
			}
		}
		void print_activated(void){
			for(int i=0;i<nr_layer;i++){
				std::cout << A[i];
			}
		}

};

class backward_param{
	public:
        int nr_layer;
		int batch_size;
		vector<int> layer_info;

		vector<DynamicMatrix<double> > dW;
		vector<DynamicMatrix<double> > db;
		vector<DynamicMatrix<double> > dA;
		vector<DynamicMatrix<double> > dZ;
		backward_param(vector<int> vec, int b_size){
			nr_layer = vec.size();
			batch_size = b_size;
			layer_info = vec;
			dW.resize(nr_layer-1);
			db.resize(nr_layer-1);
			dA.resize(nr_layer-1);
			dZ.resize(nr_layer-1);
			for(int i=0;i<nr_layer-1;i++){
				dW[i].resize(vec[i+1],vec[i]);
				db[i].resize(vec[i+1],1);
				dA[i].resize(vec[i+1],b_size);
				dZ[i].resize(vec[i+1],b_size);
			}
		}
		void print_linear_derivative(void){
			for(int i=0;i<nr_layer-1;i++){
				std::cout << dZ[i];
			}
		}
		void print_activated_derivative(void){
			for(int i=0;i<nr_layer-1;i++){
				std::cout << dA[i];
			}
		}
		void print_weight_derivative(void){
			for(int i=0;i<nr_layer-1;i++){
				std::cout << dW[i];
			}
		}
		void print_bias_derivative(void){
			for(int i=0;i<nr_layer-1;i++){
				std::cout << db[i];
			}
		}
};


void initialize_param(model_param * m_p){
	default_random_engine generator;
	normal_distribution<double> distribution(0.0,1.0);
	for(int i=0;i<m_p->nr_layer-1;i++){
		for(int j=0;j<m_p->W[i].rows();j++){
			for(int k=0;k<m_p->W[i].columns();k++){
				m_p->W[i](j,k) = distribution(generator);
			}
		}
	}
	for(int i=0;i<m_p->nr_layer-1;i++){
		for(int j=0;j<m_p->b[i].rows();j++){
			for(int k=0;k<m_p->b[i].columns();k++){
				m_p->b[i](j,k) = 0;
			}
		}
	}
}

void feed_forward(model_param *m_p, forward_param *f_p, DynamicMatrix<double> X){
	int nr_layer = m_p->nr_layer;
	//For now we can assume relu has been applied in hidden units and ends with softmax
#ifdef DEBUG
	if(f_p->Z[0].columns()!=X.columns()){
		std::cout << "Warning: dimention mismatch" << std::endl;
	}
#endif
	f_p->Z[0] = X;
	f_p->A[0] = X;
	for(int i=1;i<nr_layer-1;i++){
		f_p->Z[i] = add(mul(m_p->W[i-1], f_p->A[i-1]), m_p->b[i-1]);
		f_p->A[i] = apply_relu(f_p->Z[i]);
	}
	f_p->Z[nr_layer-1] = add(mul(m_p->W[nr_layer-2], f_p->A[nr_layer-2]), m_p->b[nr_layer-2]);
	f_p->A[nr_layer-1] = apply_softmax(f_p->Z[nr_layer-1]);
}
