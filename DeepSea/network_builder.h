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

using namespace std;

class model_param{
	public:
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
    private:
        int nr_layer;
		vector<int> layer_info;
};
