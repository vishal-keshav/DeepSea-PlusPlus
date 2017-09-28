/* Header for optimized inference
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

#include <iostream>
#include <blaze/Math.h>

//This is un-optimized version of predictor, will be extended after performance test
DynamicMatrix<double> predict(model_param *m_p, forward_param *f_p, DynamicMatrix<double> input){
	feed_forward(m_p, f_p, input);
	return f_p->A[f_p->nr_layer-1];
}
