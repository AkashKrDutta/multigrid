
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <cfloat>


using namespace std;

#define X_SIZE 17
#define Y_SIZE 17
#define Z_SIZE 17
#define h_x 1
#define h_y 1
#define h_z 1
#define pos(x,y,z) ((x) + ((y)*X_SIZE) + ((z)*Y_SIZE*X_SIZE))
#define SIZE  X_SIZE*Y_SIZE*Z_SIZE
#define Max 500
#define EPSILON .0000000001
#define lx 9
#define ly 9
#define lz 9


//for thrust
#include <cmath>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
int jump_x=1;
int jump_y=1;
int jump_z=1;
dim3 blockSize(4, 4, 1);
dim3 gridSize((X_SIZE / blockSize.x) + 1, (Y_SIZE / blockSize.y) + 1, 1);

void restrict(int rx,int ry,int rz)
{
	int mult=(X_SIZE-1)/(rx-1);
	jump_x*=mult;
	jump_y*=mult;
	jump_z*=mult;
}
void host_interpolate()
{
	jump_x/=2;
	jump_y/=2;
	jump_z/=2;
}
__device__ float particle_interploate(int x_state,int y_state,int z_state,float * d_arr,int x_idx,int y_idx,int z_idx)
{
	float value=0.0;
	if(x_state + y_state +z_state ==1)
	{
		value=(d_arr[pos(x_idx-x_state,y_idx-y_state,z_idx-z_state)]+d_arr[pos(x_idx+x_state,y_idx+y_state,z_idx+z_state)])/2.0;
	}
	else if(x_state + y_state +z_state ==3)
	{
		value= (d_arr[pos(x_idx-x_state,y_idx-y_state,z_idx-z_state)]+d_arr[pos(x_idx+x_state,y_idx+y_state,z_idx+z_state)]+d_arr[pos(x_idx-x_state,y_idx-y_state,z_idx+z_state)]+
				d_arr[pos(x_idx-x_state,y_idx+y_state,z_idx-z_state)]+d_arr[pos(x_idx+x_state,y_idx-y_state,z_idx-z_state)]+d_arr[pos(x_idx+x_state,y_idx+y_state,z_idx-z_state)]+
				d_arr[pos(x_idx+x_state,y_idx-y_state,z_idx+z_state)]+d_arr[pos(x_idx-x_state,y_idx+y_state,z_idx+z_state)])/8.0;
	}
	else
	{
		if(!x_state)
		{
			value=(d_arr[pos(x_idx,y_idx-1,z_idx-1)]+d_arr[pos(x_idx,y_idx-1,z_idx+1)]+d_arr[pos(x_idx,y_idx+1,z_idx-1)]+d_arr[pos(x_idx,y_idx+1,z_idx+1)])/4.0;
		}
		else if(!y_state)
		{
			value=(d_arr[pos(x_idx-1,y_idx,z_idx-1)]+d_arr[pos(x_idx-1,y_idx,z_idx+1)]+d_arr[pos(x_idx+1,y_idx,z_idx-1)]+d_arr[pos(x_idx+1,y_idx,z_idx+1)])/4.0;
		}
		else if(!z_state)
		{
			value=(d_arr[pos(x_idx-1,y_idx-1,z_idx)]+d_arr[pos(x_idx+1,y_idx-1,z_idx)]+d_arr[pos(x_idx-1,y_idx+1,z_idx)]+d_arr[pos(x_idx+1,y_idx+1,z_idx)])/4.0;
		}
	}
	return value;
}

//to opitmize by calling no wastage of thread---reduced trhead divergence..
__global__ void dev_interpolate(float *d_arr,int jump_x,int jump_y,int jump_z)
{
	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
	int x_state=0;
	int y_state=0;
	int z_state=0;
	if(x_idx<X_SIZE && y_idx<Y_SIZE)
	{
		if((x_idx/jump_x)%2==1)
			x_state=1;
		if((y_idx/jump_y)%2==1)
			y_state=1;
		// if( x_idx<X_SIZE && y_idx<Y_SIZE )
		// {
		// 	for(int i=0;i<Z_SIZE;i+=jump_z*2)
		// 	{
		// 		if( (y_idx%(jump_y*2))==jump_y)
		// 			d_arr[pos(x_idx,y_idx,i)]=0.0;
		// 		else if((x_idx%(jump_x*2))==jump_x)
		// 			d_arr[pos(x_idx,y_idx,i)]=0.0;
		// 	}
		// 	for(int i=jump_z;i<Z_SIZE;i+=jump_z*2)
		// 		d_arr[pos(x_idx,y_idx,i)]=0;
		// }
		for(int i=0;i<Z_SIZE;i+=jump_z)
		{	
			if((i/jump_z)%2==1)
				z_state=1;
			if(x_state || y_state || z_state )
				d_arr[pos(x_idx,y_idx,i)]=particle_interploate(x_state,y_state,z_state,d_arr,x_idx,y_idx,i);
			z_state=0;
		}
	}
}
// __global__ void max(float *a, float *c)
// {
// 	extern __shared__ float sdata[];
//
// 	unsigned int tid = threadIdx.x;
// 	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
//
// 	if (i < SIZE)
// 		sdata[tid] = a[i];
// 	else
// 		sdata[tid] = FLT_MIN;
//
// 	__syncthreads();
// 	for (unsigned int s = blockDim.x / 2; s >= 1; s = s / 2)
// 	{
// 		if (tid<= s)
// 		{
// 			if (sdata[tid]<sdata[tid + s])
// 				sdata[tid] = sdata[tid + s];
// 		}
// 		//////////////////////////////
// 		__syncthreads();
// 	}
// 	if (tid == 0) c[blockIdx.x] = sdata[0];
// }
__device__ float funFinite(float *arr, int x_idx, int y_idx, int z_idx)
{
	float deriv = 0.0;

	if ((x_idx > 0) && (x_idx<(X_SIZE - 1)))
	{
		deriv += (arr[pos(x_idx - 1, y_idx, z_idx)] - (2 * arr[pos(x_idx, y_idx, z_idx)]) + arr[pos(x_idx + 1, y_idx, z_idx)]) / (h_x*h_x);
	}
	else if (x_idx == 0)
	{
		deriv += ((2 * arr[pos(x_idx, y_idx, z_idx)]) - (5 * arr[pos(x_idx + 1, y_idx, z_idx)]) +
			(4 * arr[pos(x_idx + 2, y_idx, z_idx)]) - (arr[pos(x_idx + 3, y_idx, z_idx)])) / (h_x*h_x);
	}
	else if (x_idx == (X_SIZE - 1))
	{
		deriv += ((2 * arr[pos(x_idx, y_idx, z_idx)]) - (5 * arr[pos(x_idx - 1, y_idx, z_idx)]) +
			(4 * arr[pos(x_idx - 2, y_idx, z_idx)]) - (arr[pos(x_idx - 3, y_idx, z_idx)])) / (h_x*h_x);
	}
	
	if ((y_idx > 0) && (y_idx<(Y_SIZE - 1)))
	{
		deriv += (arr[pos(x_idx, y_idx - 1, z_idx)] - (2 * arr[pos(x_idx, y_idx, z_idx)]) + arr[pos(x_idx, y_idx + 1, z_idx)]) / (h_y*h_y);
	}
	else if (y_idx == 0)
	{		
		deriv += ((2 * arr[pos(x_idx, y_idx, z_idx)]) - (5 * arr[pos(x_idx, y_idx + 1, z_idx)]) +
			(4 * arr[pos(x_idx, y_idx + 2, z_idx)]) - (arr[pos(x_idx, y_idx + 3, z_idx)])) / (h_y*h_y);
	}
	else if (y_idx == (Y_SIZE - 1))
	{
		deriv += ((2 * arr[pos(x_idx, y_idx, z_idx)]) - (5 * arr[pos(x_idx, y_idx - 1, z_idx)]) +
			(4 * arr[pos(x_idx, y_idx - 2, z_idx)]) - (arr[pos(x_idx, y_idx - 3, z_idx)])) / (h_y*h_y);
	}
	if ((z_idx > 0) && (z_idx<(Z_SIZE - 1)))
	{
		deriv += (arr[pos(x_idx, y_idx, z_idx - 1)] - (2 * arr[pos(x_idx, y_idx, z_idx)]) + arr[pos(x_idx, y_idx, z_idx + 1)]) / (h_z*h_z);
	}
	else if (z_idx == 0)
	{
		deriv += ((2 * arr[pos(x_idx, y_idx, z_idx)]) - (5 * arr[pos(x_idx, y_idx, z_idx + 1)]) +
			(4 * arr[pos(x_idx, y_idx, z_idx + 2)]) - (arr[pos(x_idx, y_idx, z_idx + 3)])) / (h_z*h_z);
	}
	else if (z_idx == (Z_SIZE - 1))
	{
		deriv += ((2 * arr[pos(x_idx, y_idx, z_idx)]) - (5 * arr[pos(x_idx, y_idx, z_idx - 1)]) +
			(4 * arr[pos(x_idx, y_idx, z_idx - 2)]) - (arr[pos(x_idx, y_idx, z_idx - 3)])) / (h_z*h_z);
	}
	return deriv;
}

__global__ void laplacian(float *arr, float *ans)
{
	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
	//int z_idx = (blockIdx.z*blockDim.z) + threadIdx.z;
	int i;
	if(x_idx<X_SIZE && y_idx<Y_SIZE){
		for (i = 0; i < Z_SIZE; i++){
			ans[pos(x_idx, y_idx, i)] = funFinite(arr, x_idx, y_idx, i);
		}
	}
}


__global__ void jacobi(float *d_Arr, float * d_rho, float * d_ans, float *d_ANS)
{
	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
	int i;
	if( (x_idx <(X_SIZE-1)) && (y_idx<(Y_SIZE-1)) && (x_idx>0) && (y_idx>0)){
			for (i = 1; i < Z_SIZE-1; i++){
				d_ANS[pos(x_idx, y_idx, i)] = d_Arr[pos(x_idx, y_idx, i)] + (-d_rho[pos(x_idx, y_idx, i)] + d_ans[pos(x_idx, y_idx, i)]) / (2*((1/(h_x*h_x))+(1/(h_y*h_y))+(1/(h_z*h_z))));
			}
		}
		
	else if(x_idx==0 || x_idx == X_SIZE-1 || y_idx == 0 || y_idx==Y_SIZE-1)
	{
		for(int i=1;i<Z_SIZE-1;i++)
			d_ANS[pos(x_idx,y_idx,i)]=0;
	}
	d_ANS[pos(x_idx,y_idx,0)]=0;
	d_ANS[pos(x_idx,y_idx,Z_SIZE-1)]=0;
}

__global__ void subtract(float *d_Arr, float * d_ANS, float* d_sub)
{
	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
	int i;
	if(x_idx<X_SIZE && y_idx < Y_SIZE){
		for (i = 0; i < Z_SIZE; i++){
			d_sub[pos(x_idx, y_idx, i)] = d_Arr[pos(x_idx, y_idx, i)]-d_ANS[pos(x_idx, y_idx, i)];
		}
	}
}

__global__ void abs_subtract(float *d_Arr, float * d_ANS, float* d_sub)
{
	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
	int i;
	if(x_idx<X_SIZE && y_idx < Y_SIZE){
		for (i = 0; i < Z_SIZE; i++){
			d_sub[pos(x_idx, y_idx, i)] =abs(d_Arr[pos(x_idx, y_idx, i)]-d_ANS[pos(x_idx, y_idx, i)]);
		}
	}
}

// __global__ void absolute(float *d_Arr,float* d_ans)
// {
// 	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
// 	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
// 	int i;
// 	if(x_idx<X_SIZE && y_idx < Y_SIZE){
// 		for (i = 0; i < Z_SIZE; i++){
// 			d_ans[pos(x_idx, y_idx, i)] =abs(d_Arr[pos(x_idx, y_idx, i)]);
// 		}
// }

void interpolate(float * d_ANS)
{
	host_interpolate();
	dev_interpolate <<<gridSize,blockSize>>>(d_ANS,jump_x,jump_y,jump_z);
	// call smoother();
}
__global__ void justtest(float*d_ANS)
{
	for(int k=0;k<Z_SIZE;k++)
		for(int j=0;j<Y_SIZE;j++)
			for(int i=0;i<X_SIZE;i++)
				d_ANS[pos(i,j,k)]=i+j+k;
}

__global__ void copy(float* d_to,float* d_from)
{
	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
	if(x_idx<X_SIZE && y_idx < Y_SIZE)
	{
		for(int i=0;i<Z_SIZE;i++)
			d_to[pos(x_idx,y_idx,i)]=d_from[pos(x_idx,y_idx,i)];
	}
}

void smoother(float* d_rho,float * d_ANS,int N)
{
	dim3 gridsize((((((gridSize.x-1)*blockSize.x)-1)/jump_x)+1)/blockSize.x +1,(((((gridSize.y-1)*blockSize.y)-1)/jump_y)+1)/blockSize.y +1, 1);
	float * d_ans,* dummy,*d_Arr;
	cudaMalloc((void**)&d_Arr,sizeof(SIZE));
	cudaMalloc((void**)&d_ans,sizeof(SIZE));
	copy<<<gridsize,blockSize>>>(d_Arr,d_ANS);
	for(int i=0;i<N;i++)
	{
		laplacian << <gridsize, blockSize >> > (d_Arr, d_ans);
		jacobi << <gridsize, blockSize >> > (d_Arr, d_rho, d_ans, d_ANS);
		dummy = d_Arr;
		d_Arr = d_ANS;
		d_ANS=dummy;
	}
	d_ANS=d_Arr;
	cudaFree(d_ans);
	cudaFree(d_Arr);
}

void residual(float * d_residual,float *d_ANS,float* d_rho)
{
	dim3 gridsize((((((gridSize.x-1)*blockSize.x)-1)/jump_x)+1)/blockSize.x +1,(((((gridSize.y-1)*blockSize.y)-1)/jump_y)+1)/blockSize.y +1, 1);
	float * d_laplace;
	cudaMalloc((void**)&d_laplace,sizeof(SIZE));
	laplacian<<<gridsize,blockSize>>>(d_ANS,d_laplace);
	//cudaDeviceSynchronize();
	subtract<<<gridsize,blockSize>>>(d_rho,d_laplace,d_residual);
	cudaFree(d_laplace);
}

void solver(float * d_residual,float* d_error, float eps)
{
	dim3 gridsize((((((gridSize.x-1)*blockSize.x)-1)/jump_x)+1)/blockSize.x +1,(((((gridSize.y-1)*blockSize.y)-1)/jump_y)+1)/blockSize.y +1, 1);
	float * d_Arr,*d_ans,max_error,*d_sub,*dummy;
	cudaMalloc((void**)&d_ans,sizeof(SIZE));
	cudaMalloc((void**)&d_sub,sizeof(SIZE));
	cudaMalloc((void**)&d_Arr,sizeof(SIZE));
	copy<<<gridsize,blockSize>>>(d_Arr,d_error);
	do{
		laplacian << <gridsize, blockSize >> > (d_Arr, d_ans);
		jacobi << <gridsize, blockSize >> > (d_Arr, d_residual, d_ans, d_error);
		abs_subtract << <gridsize, blockSize >> >(d_Arr, d_error, d_sub);
		thrust::device_ptr<float> dev_ptr(d_sub);
		thrust::device_ptr<float> devsptr=(thrust::max_element(dev_ptr, dev_ptr + SIZE));
		max_error=*devsptr;
		dummy=d_Arr;
		d_Arr=d_error;
		d_error=dummy;
	}while(max_error<=eps);
	d_error=d_Arr;
	cudaFree(d_ans);
	cudaFree(d_sub);
}

void Vcycle(float * d_rho,float* d_Arr,float error,int N,int rx,int ry,int rz)
{
	smoother(d_rho,d_Arr,N);
	float * d_residual,max_error;
	cudaMalloc((void**)&d_residual,sizeof(float)*SIZE);
	residual(d_residual,d_Arr,d_rho);
	thrust::device_ptr<float> dev_ptr(d_residual);
	thrust::device_ptr<float> devsptr=(thrust::max_element(dev_ptr, dev_ptr + SIZE));
	max_error=*devsptr;
	while(max_error>error)
	{
		restrict(rx,ry,rz);

	}
}

int main()
{

	float Array[SIZE];
	//float h_Arr[SIZE];
	float h_rho[SIZE];
	//float h_Arr[SIZE];

	for (int k = 0; k<Z_SIZE; ++k)
		for (int j = 0; j<Y_SIZE; ++j)
			for (int i = 0; i<X_SIZE; ++i)
			{
				Array[(i)+(j*X_SIZE) + (k*Y_SIZE*X_SIZE)] = 0;
				h_rho[(i)+(j*X_SIZE) + (k*Y_SIZE*X_SIZE)] = 0;
				//h_Arr[(i)+(j*X_SIZE) + (k*Y_SIZE*X_SIZE)] = 0;
			}
	h_rho[pos(X_SIZE / 2-1, Y_SIZE / 2, Z_SIZE/2 )] = 100;
	h_rho[pos(X_SIZE / 2+1, Y_SIZE / 2, Z_SIZE / 2 )] = -100;
	// dim3 blockSize(4, 4, 1);
	dim3 gridsize((((((gridSize.x-1)*blockSize.x)-1)/jump_x)+1)/blockSize.x +1,(((((gridSize.y-1)*blockSize.y)-1)/jump_y)+1)/blockSize.y +1, 1);

	float *d_Arr;
	float *d_ans;
	float *d_rho;
	//input h_rho; and memcpy
	float *d_ANS;
	//thrust::device_vector<float> d_subtract;
	//float *d_blockmax;
	float *d_sub;
	float *dummy;
	//float h_sub[SIZE];
	float h_ANS[SIZE];
	//float h_blockmax[1];
	//malloc

	cudaMalloc((void**)&d_Arr, SIZE*sizeof(float));
	cudaMalloc((void**)&d_ans, SIZE*sizeof(float));
	cudaMalloc((void**)&d_rho, SIZE*sizeof(float));
	cudaMalloc((void**)&d_sub, SIZE*sizeof(float));
	cudaMalloc((void**)&d_ANS, SIZE*sizeof(float));
	//cudaMalloc((void**)&d_blockmax, sizeof(float));
	cudaMemcpy(d_Arr, Array, SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rho, h_rho, SIZE*sizeof(float), cudaMemcpyHostToDevice);


	for (int i = 0; i<200; i++)
	{


		laplacian << <gridsize, blockSize >> > (d_Arr, d_ans);

		//cudaMemcpy(h_Arr, d_ans, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
		jacobi << <gridsize, blockSize >> > (d_Arr, d_rho, d_ans, d_ANS);

		abs_subtract << <gridsize, blockSize >> >(d_Arr, d_ANS, d_sub);


		//thrust::host_vector<float> h_vec(SIZE);
		//thrust::generate(h_vec.begin(), h_vec.end(), rand);
		//for (int i = 0; i < 100; i++)
		//h_vec[i] = i;
		//thrust::device_vector<float>::iterator iter =
		//thrust::max_element(d_subtract.begin(), d_subtract.end());

		//unsigned int position = iter - d_sub.begin();

		// to make manually

		/*cudaMemcpy(h_sub, d_sub, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
		float max_val = fabs(h_sub[0]);
		for (int j = 0; j < SIZE; j++)
		{

			if (max_val < fabs(h_sub[j]))
				max_val = h_sub[j];

		}
		cout << max_val << endl;
		*/
		//max << <1, SIZE ,SIZE*sizeof(float)>> > (d_sub, d_blockmax);
		//cudaMemcpy(h_blockmax, d_blockmax, sizeof(float), cudaMemcpyDeviceToHost);
		//cout << *h_blockmax << endl;

		//cudaDeviceSynchronize();
		thrust::device_ptr<float> dev_ptr(d_sub);
		thrust::device_ptr<float> devsptr=(thrust::max_element(dev_ptr, dev_ptr + SIZE));
		//cudaDeviceSynchronize();
		cout << *devsptr<<endl;

		//cudaMemcpy(h_Arr, d_Arr, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
		//for (int k = 0; k < SIZE; k++)
		//cout << h_ANS[k]<<" ";
		//cout << endl;

		//for (int i = 0; i < 5; i++)
		//cout << h_ANS[i] << " ";
		//cout << endl;

		// if(max_val<EPSILON)
		// 	break;
		if(*devsptr<EPSILON)
			break;
		dummy = d_Arr;
		d_Arr = d_ANS;
		d_ANS = dummy;
	}
	d_ANS=d_Arr;
	//justtest<<<1,1>>>(d_ANS);
	// restrict(9,9,9);
	// interpolate(gridSize,blockSize,d_ANS);
	cudaMemcpy(h_ANS, d_ANS, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
	int count=0;
	for(int k=0;k<Z_SIZE;k+=jump_z)
	{
		for(int j=0;j<Y_SIZE;j+=jump_y)
		{
			for(int i=0;i<X_SIZE;i+=jump_x)
				{
					if(h_ANS[pos(i,j,k)]<0)
						count++;
					printf("%9.6f ",h_ANS[pos(i,j,k)]);
				}
			cout<<endl;
		}
		cout<<endl<<endl;
	}
	cout << h_ANS[pos(X_SIZE / 2-1, Y_SIZE / 2, Z_SIZE / 2 )]<<" "<<count;;
	/*for (int k = 0; k<Z_SIZE; ++k)
	{
	for (int j = 0; j<Y_SIZE; ++j)
	{
	for (int i = 0; i<X_SIZE; ++i)
	cout << h_Arr[(i)+(j*X_SIZE) + (k*Y_SIZE*X_SIZE)] << " ";
	cout << endl;
	}
	cout << endl << endl;
	}*/
	//cout << h_Arr[pos(0, 5, 5)] << endl;
	cudaFree(d_Arr);
	cudaFree(d_ans);
	cudaFree(d_ANS);
	cudaFree(d_rho);
	cudaFree(d_sub);
	cudaFree(dummy);
//	getchar();
	return 0;
}
