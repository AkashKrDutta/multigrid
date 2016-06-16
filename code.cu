
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
//#include <cdouble>


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
int jump=1;
dim3 blockSize(32,32,1);
dim3 gridSize((X_SIZE / blockSize.x)+1, (Y_SIZE / blockSize.y)+1,1);

void rstrict(int rx,int ry,int rz)
{
	int mult=(X_SIZE-1)/(rx-1);
	jump*=mult;
}
void host_interpolate()
{
	jump/=2;
}
__device__ double particle_interploate(int x_state,int y_state,int z_state,double * d_arr,int x_idx,int y_idx,int z_idx)
{
	double value=0.0;
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
__global__ void dev_interpolate(double *d_arr,int jump)
{
	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
	int x_state=0;
	int y_state=0;
	int z_state=0;
	if(x_idx<X_SIZE && y_idx<Y_SIZE)
	{
		if((x_idx/jump)%2==1)
			x_state=1;
		if((y_idx/jump)%2==1)
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
		for(int i=0;i<Z_SIZE;i+=jump)
		{	
			if((i/jump)%2==1)
				z_state=1;
			if(x_state || y_state || z_state )
				d_arr[pos(x_idx,y_idx,i)]=particle_interploate(x_state,y_state,z_state,d_arr,x_idx,y_idx,i);
			z_state=0;
		}
	}
}
// __global__ void max(double *a, double *c)
// {
// 	extern __shared__ double sdata[];
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
__device__ double funFinite(double *arr, int x_idx, int y_idx, int z_idx,int jump)
{
	double deriv = 0.0;

	if ((x_idx > 0) && (x_idx<(X_SIZE - 1)))
	{
		deriv += (arr[pos(x_idx - 1*jump, y_idx, z_idx)] - (2 * arr[pos(x_idx, y_idx, z_idx)]) + arr[pos(x_idx + 1*jump, y_idx, z_idx)]) / (h_x*h_x);
	}
	else if (x_idx == 0)
	{
		deriv += ((2 * arr[pos(x_idx, y_idx, z_idx)]) - (5 * arr[pos(x_idx + 1*jump, y_idx, z_idx)]) +
			(4 * arr[pos(x_idx + 2*jump, y_idx, z_idx)]) - (arr[pos(x_idx + 3*jump, y_idx, z_idx)])) / (h_x*h_x);
	}
	else if (x_idx == (X_SIZE - 1))
	{
		deriv += ((2 * arr[pos(x_idx, y_idx, z_idx)]) - (5 * arr[pos(x_idx - 1*jump, y_idx, z_idx)]) +
			(4 * arr[pos(x_idx - 2*jump, y_idx, z_idx)]) - (arr[pos(x_idx - 3*jump, y_idx, z_idx)])) / (h_x*h_x);
	}
	
	if ((y_idx > 0) && (y_idx<(Y_SIZE - 1)))
	{
		deriv += (arr[pos(x_idx, y_idx - 1*jump, z_idx)] - (2 * arr[pos(x_idx, y_idx, z_idx)]) + arr[pos(x_idx, y_idx + 1*jump, z_idx)]) / (h_y*h_y);
	}
	else if (y_idx == 0)
	{		
		deriv += ((2 * arr[pos(x_idx, y_idx, z_idx)]) - (5 * arr[pos(x_idx, y_idx + 1*jump, z_idx)]) +
			(4 * arr[pos(x_idx, y_idx + 2*jump, z_idx)]) - (arr[pos(x_idx, y_idx + 3*jump, z_idx)])) / (h_y*h_y);
	}
	else if (y_idx == (Y_SIZE - 1))
	{
		deriv += ((2 * arr[pos(x_idx, y_idx, z_idx)]) - (5 * arr[pos(x_idx, y_idx - 1*jump, z_idx)]) +
			(4 * arr[pos(x_idx, y_idx - 2*jump, z_idx)]) - (arr[pos(x_idx, y_idx - 3*jump, z_idx)])) / (h_y*h_y);
	}
	if ((z_idx > 0) && (z_idx<(Z_SIZE - 1)))
	{
		deriv += (arr[pos(x_idx, y_idx, z_idx - 1*jump)] - (2 * arr[pos(x_idx, y_idx, z_idx)]) + arr[pos(x_idx, y_idx, z_idx + 1*jump)]) / (h_z*h_z);
	}
	else if (z_idx == 0)
	{
		deriv += ((2 * arr[pos(x_idx, y_idx, z_idx)]) - (5 * arr[pos(x_idx, y_idx, z_idx + 1*jump)]) +
			(4 * arr[pos(x_idx, y_idx, z_idx + 2*jump)]) - (arr[pos(x_idx, y_idx, z_idx + 3*jump)])) / (h_z*h_z);
	}
	else if (z_idx == (Z_SIZE - 1))
	{
		deriv += ((2 * arr[pos(x_idx, y_idx, z_idx)]) - (5 * arr[pos(x_idx, y_idx, z_idx - 1*jump)]) +
			(4 * arr[pos(x_idx, y_idx, z_idx - 2*jump)]) - (arr[pos(x_idx, y_idx, z_idx - 3*jump)])) / (h_z*h_z);
	}
	return deriv;
}

__global__ void laplacian(double *arr, double *ans,int jump)
{

	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
	x_idx*=jump;
	y_idx*=jump;
	//int z_idx = (blockIdx.z*blockDim.z) + threadIdx.z;
	int i;
	if(x_idx<X_SIZE && y_idx<Y_SIZE){
		for (i = 0; i < Z_SIZE; i+=jump){
			ans[pos(x_idx, y_idx, i)] = funFinite(arr, x_idx, y_idx, i,jump);
		}
	}
}


__global__ void jacobi(double *d_Arr, double * d_rho, double * d_ans, double *d_ANS,int jump)
{
	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
	x_idx*=jump;
	y_idx*=jump;
	int i;
	if(x_idx<X_SIZE && y_idx <Y_SIZE){
		if( (x_idx <(X_SIZE-1)) && (y_idx<(Y_SIZE-1)) && (x_idx>0) && (y_idx>0)){
				for (i = jump; i < Z_SIZE-1; i+=jump){
					d_ANS[pos(x_idx, y_idx, i)] = d_Arr[pos(x_idx, y_idx, i)] + (-d_rho[pos(x_idx, y_idx, i)] + d_ans[pos(x_idx, y_idx, i)]) / (2*((1/(h_x*h_x))+(1/(h_y*h_y))+(1/(h_z*h_z))));
				}
			}
			
		else if(x_idx==0 || x_idx == X_SIZE-1 || y_idx == 0 || y_idx==Y_SIZE-1)
		{
			for(int i=jump;i<Z_SIZE-jump;i+=jump)
				d_ANS[pos(x_idx,y_idx,i)]=0;
		}
		d_ANS[pos(x_idx,y_idx,0)]=0;
		d_ANS[pos(x_idx,y_idx,Z_SIZE-1)]=0;
	}

}

__global__ void subtract(double *d_Arr, double * d_ANS, double* d_sub)
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

__global__ void abs_subtract(double *d_Arr, double * d_ANS, double* d_sub,int jump)
{
	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
	x_idx*=jump;
	y_idx*=jump;
	int i;
	if(x_idx<X_SIZE && y_idx < Y_SIZE){
		for (i = 0; i < Z_SIZE; i+=jump){
			d_sub[pos(x_idx, y_idx, i)] =abs(d_Arr[pos(x_idx, y_idx, i)]-d_ANS[pos(x_idx, y_idx, i)]);
		}
	}
}

__global__ void absolute(double *d_Arr,double* d_ans)
{
	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
	int i;
	if(x_idx<X_SIZE && y_idx < Y_SIZE){
		for (i = 0; i < Z_SIZE; i++){
			d_ans[pos(x_idx, y_idx, i)] =abs(d_Arr[pos(x_idx, y_idx, i)]);
		}
	}
}
__global__ void all_zero(double *arr,int jump)
{
	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
	x_idx*=jump;
	y_idx*=jump;
	if(x_idx<X_SIZE && y_idx < Y_SIZE)
	{
		for(int i=0;i<Z_SIZE;i+=jump)
			arr[pos(x_idx,y_idx,i)]=0;
	}
}

__global__ void justtest(double*d_ANS)
{
	for(int k=0;k<Z_SIZE;k++)
		for(int j=0;j<Y_SIZE;j++)
			for(int i=0;i<X_SIZE;i++)
				d_ANS[pos(i,j,k)]=i+j+k;
}

__global__ void copy(double* d_to,double* d_from,int jump)
{
	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
	x_idx*=jump;
	y_idx*=jump;
	if(x_idx<X_SIZE && y_idx < Y_SIZE)
	{
		for(int i=0;i<Z_SIZE;i+=jump)
			d_to[pos(x_idx,y_idx,i)]=d_from[pos(x_idx,y_idx,i)];
	}
}

double* smoother(double* d_rho,double * d_ANS,int N)
{
	dim3 gridsize;
	gridsize.x=(((X_SIZE-1)/jump)+1)/blockSize.x + 1;
	gridsize.y=(((X_SIZE-1)/jump)+1)/blockSize.x + 1;
	gridsize.z=1;
	double * d_ans,* dummy,*d_Arr;
	cudaMalloc((void**)&d_Arr,sizeof(double)*(SIZE));
	cudaMalloc((void**)&d_ans,sizeof(double)*(SIZE));
	copy<<<gridsize,blockSize>>>(d_Arr,d_ANS,jump);
	for(int i=0;i<N;i++)
	{
		laplacian << <gridsize, blockSize >> > (d_Arr, d_ans,jump);
		jacobi << <gridsize, blockSize >> > (d_Arr, d_rho, d_ans, d_ANS,jump);
		dummy = d_Arr;
		d_Arr = d_ANS;
		d_ANS=dummy;
	}
	d_ANS=d_Arr;
	cudaFree(d_ans);
	cudaFree(dummy);
	return d_ANS;
}
double* interpolate(double * d_ANS,double * d_residual,int N)
{
	host_interpolate();
	cudaDeviceSynchronize();
	dev_interpolate <<<gridSize,blockSize>>>(d_ANS,jump);
	d_ANS=smoother(d_residual,d_ANS,N);
	return d_ANS;
}

void residual(double * d_residual,double *d_ANS,double* d_rho)
{
	dim3 gridsize;
	gridsize.x=(((X_SIZE-1)/jump)+1)/blockSize.x + 1;
	gridsize.y=(((X_SIZE-1)/jump)+1)/blockSize.x + 1;
	gridsize.z=1;
	double * d_laplace;
	cudaMalloc((void**)&d_laplace,sizeof(double)*(SIZE));
	laplacian<<<gridsize,blockSize>>>(d_ANS,d_laplace,jump);
	//cudaDeviceSynchronize();
	subtract<<<gridsize,blockSize>>>(d_rho,d_laplace,d_residual);
	cudaFree(d_laplace);
}

double* solver(double * d_residual,double* d_error, double eps)
{
	dim3 gridsize;
	gridsize.x=(((X_SIZE-1)/jump)+1)/blockSize.x + 1;
	gridsize.y=(((X_SIZE-1)/jump)+1)/blockSize.x + 1;
	gridsize.z=1;
	double * d_Arr,*d_ans,max_error,*d_sub,*dummy;
	cudaMalloc((void**)&d_ans,sizeof(double)*(SIZE));
	cudaMalloc((void**)&d_sub,sizeof(double)*(SIZE));
	cudaMalloc((void**)&d_Arr,sizeof(double)*(SIZE));
	copy<<<gridsize,blockSize>>>(d_Arr,d_error,jump);
	all_zero<<<gridsize,blockSize>>>(d_sub,1);
	do{
		laplacian << <gridsize, blockSize >> > (d_Arr, d_ans,jump);
		jacobi << <gridsize, blockSize >> > (d_Arr, d_residual, d_ans, d_error,jump);
		abs_subtract << <gridsize, blockSize >> >(d_Arr, d_error, d_sub,jump);
		thrust::device_ptr<double> dev_ptr(d_sub);
		thrust::device_ptr<double> devsptr=(thrust::max_element(dev_ptr, dev_ptr + SIZE));
		max_error=*devsptr;
		dummy=d_Arr;
		d_Arr=d_error;
		d_error=dummy;
		//cout<<max_error<<endl;
	}while(max_error>eps);
	d_error=d_Arr;
	cudaFree(d_ans);
	cudaFree(d_sub);
	return d_error;
	//cout<<gridsize.x<<" "<<gridsize.y;
}

__global__ void add(double * d1,double* d2,double* dest,int jump)
{
	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
	x_idx*=jump;
	y_idx*=jump;
	if(x_idx<X_SIZE && y_idx < Y_SIZE)
	{
		for(int i=0;i<Z_SIZE;i+=jump)
			dest[pos(x_idx,y_idx,i)]=d1[pos(x_idx,y_idx,i)]+d2[pos(x_idx,y_idx,i)];
	}
}

__global__ void boundary_zero(double* d_zero)
{
	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
	if(x_idx<X_SIZE && y_idx < Y_SIZE)
	{
		if(x_idx==0 || x_idx == X_SIZE-1 || y_idx == 0 || y_idx==Y_SIZE-1)
		{
			for(int i=1;i<Z_SIZE-1;i+=1)
				d_zero[pos(x_idx,y_idx,i)]=0;
		}
		d_zero[pos(x_idx,y_idx,0)]=0;
		d_zero[pos(x_idx,y_idx,Z_SIZE-1)]=0;
	}
}



void Vcycle(double * d_rho,double* d_Arr,double* d_error,double error,int N,int rx,int ry,int rz)
{
	//dim3 gridsize((((((gridSize.x-1)*blockSize.x)-1)/jump_x)+1)/blockSize.x +1,(((((gridSize.y-1)*blockSize.y)-1)/jump_y)+1)/blockSize.y +1, 1);
	smoother(d_rho,d_Arr,N);
	double * d_residual,max_error;
	cudaMalloc((void**)&d_residual,sizeof(double)*SIZE);
	residual(d_residual,d_Arr,d_rho);
	rstrict(rx,ry,rz);
	solver(d_residual,d_error,error);//init error to all 0 before calling Vcylce
	while(jump>1)
		interpolate(d_error,d_residual,N);
	add<<<gridSize,blockSize>>>(d_error,d_Arr,d_Arr,jump);
	residual(d_residual,d_Arr,d_rho);
	thrust::device_ptr<double> dev_ptr(d_residual);
	thrust::device_ptr<double> devsptr=(thrust::max_element(dev_ptr, dev_ptr + SIZE));
	max_error=*devsptr;
	cout<<"Current error= "<<max_error;
}

int main()
{

	double Array[SIZE];
	//double h_Arr[SIZE];
	double h_rho[SIZE];
	//double h_Arr[SIZE];

	for (int k = 0; k<Z_SIZE; ++k)
		for (int j = 0; j<Y_SIZE; ++j)
			for (int i = 0; i<X_SIZE; ++i)
			{
				Array[(i)+(j*X_SIZE) + (k*Y_SIZE*X_SIZE)] = 0;
				h_rho[(i)+(j*X_SIZE) + (k*Y_SIZE*X_SIZE)] = 0;
				//h_Arr[(i)+(j*X_SIZE) + (k*Y_SIZE*X_SIZE)] = 0;
			}
	h_rho[pos(X_SIZE / 2-2, Y_SIZE / 2, Z_SIZE/2 )] = 100;
	h_rho[pos(X_SIZE / 2+2, Y_SIZE / 2, Z_SIZE / 2 )] = -100;
	// dim3 blockSize(4, 4, 1);
	dim3 gridsize;
	gridsize.x=(((X_SIZE-1)/jump)+1)/blockSize.x + 1;
	gridsize.y=(((X_SIZE-1)/jump)+1)/blockSize.x + 1;
	gridsize.z=1;

	double *d_Arr;
	double *d_ans;
	double *d_rho;
	//input h_rho; and memcpy
	double *d_ANS;
	//thrust::device_vector<double> d_subtract;
	//double *d_blockmax;
	double *d_sub;
	double *dummy;
	double* d_error;
	//double h_sub[SIZE];
	double h_ANS[SIZE];
	//double h_blockmax[1];
	//malloc

	cudaMalloc((void**)&d_Arr, SIZE*sizeof(double));
	cudaMalloc((void**)&d_ans, SIZE*sizeof(double));
	cudaMalloc((void**)&d_rho, SIZE*sizeof(double));
	cudaMalloc((void**)&d_sub, SIZE*sizeof(double));
	cudaMalloc((void**)&d_ANS, SIZE*sizeof(double));
	cudaMalloc((void**)&d_error, SIZE*sizeof(double));
	//cudaMalloc((void**)&d_blockmax, sizeof(double));
	cudaMemcpy(d_Arr, Array, SIZE*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rho, h_rho, SIZE*sizeof(double), cudaMemcpyHostToDevice);
	//rstrict(9,9,9);
	//cout<<jump;
	// for (int i = 0; i<250; i++)
	// {


	// 	laplacian << <gridsize, blockSize >> > (d_Arr, d_ans,jump);

	// 	//cudaMemcpy(h_Arr, d_ans, SIZE*sizeof(double), cudaMemcpyDeviceToHost);
	// 	jacobi << <gridsize, blockSize >> > (d_Arr, d_rho, d_ans, d_ANS,jump);

	// 	abs_subtract << <gridsize, blockSize >> >(d_Arr, d_ANS, d_sub,jump);


	// 	//thrust::host_vector<double> h_vec(SIZE);
	// 	//thrust::generate(h_vec.begin(), h_vec.end(), rand);
	// 	//for (int i = 0; i < 100; i++)
	// 	//h_vec[i] = i;
	// 	//thrust::device_vector<double>::iterator iter =
	// 	//thrust::max_element(d_subtract.begin(), d_subtract.end());

	// 	//unsigned int position = iter - d_sub.begin();

	// 	// to make manually

	// 	// cudaMemcpy(h_sub, d_sub, SIZE*sizeof(double), cudaMemcpyDeviceToHost);
	// 	// double max_val = fabs(h_sub[0]);
	// 	// for (int j = 0; j < SIZE; j++)
	// 	// {

	// 	// 	if (max_val < fabs(h_sub[j]))
	// 	// 		max_val = h_sub[j];

	// 	// }
	// 	// cout << max_val << endl;
		
	// 	//max << <1, SIZE ,SIZE*sizeof(double)>> > (d_sub, d_blockmax);
	// 	//cudaMemcpy(h_blockmax, d_blockmax, sizeof(double), cudaMemcpyDeviceToHost);
	// 	//cout << *h_blockmax << endl;

	// 	//cudaDeviceSynchronize();
	// 	thrust::device_ptr<double> dev_ptr(d_sub);
	// 	thrust::device_ptr<double> devsptr=(thrust::max_element(dev_ptr, dev_ptr + SIZE));
	// 	//cudaDeviceSynchronize();
	// 	cout << *devsptr<<endl;

	// 	//cudaMemcpy(h_Arr, d_Arr, SIZE*sizeof(double), cudaMemcpyDeviceToHost);
	// 	//for (int k = 0; k < SIZE; k++)
	// 	//cout << h_ANS[k]<<" ";
	// 	//cout << endl;

	// 	//for (int i = 0; i < 5; i++)
	// 	//cout << h_ANS[i] << " ";
	// 	//cout << endl;

	// 	// if(max_val<EPSILON)
	// 	// 	break;
	// 	if(*devsptr<EPSILON)
	// 		break;
	// 	dummy = d_Arr;
	// 	d_Arr = d_ANS;
	// 	d_ANS = dummy;
	// }
	// d_ANS=d_Arr;
	//residual(d_ANS,d_Arr,d_rho);
	all_zero<<<gridsize,blockSize>>>(d_error,1);
	double * d_residual,max_error;
	cudaMalloc((void**)&d_residual,sizeof(double)*SIZE);
	for(int i=0;i<100;i++){
		d_Arr=smoother(d_rho,d_Arr,10);
		residual(d_residual,d_Arr,d_rho);
		rstrict(9,9,9);
		d_error=solver(d_residual,d_error,.00001);

		d_error=interpolate(d_error,d_residual,100);
		add<<<gridsize,blockSize>>>(d_error,d_Arr,d_Arr,jump);
		residual(d_residual,d_Arr,d_rho);
		boundary_zero<<<gridsize,blockSize>>>(d_residual);
		absolute<<<gridsize,blockSize>>>(d_residual,d_residual);
		thrust::device_ptr<double> dev_ptr(d_residual);
		thrust::device_ptr<double> devsptr=(thrust::max_element(dev_ptr, dev_ptr + SIZE));
		max_error=*devsptr;
		cout<<"Current error= "<<max_error<<endl;
	}
	// cudaMemcpy(h_ANS, d_residual, SIZE*sizeof(double), cudaMemcpyDeviceToHost);
	// int count=0;
	// // //cout<<jump;
	// for(int k=0;k<Z_SIZE;k+=jump)
	// {
	// 	for(int j=0;j<Y_SIZE;j+=jump)
	// 	{
	// 		for(int i=0;i<X_SIZE;i+=jump)
	// 			{
	// 				if(h_ANS[pos(i,j,k)]>.01)
	// 					count++;
	// 				printf("%9.6lf ",h_ANS[pos(i,j,k)]);
	// 			}
	// 		cout<<endl;
	// 	}
	// 	cout<<endl<<endl;
	// }
	// cout << h_ANS[pos(X_SIZE / 2-2, Y_SIZE / 2, Z_SIZE / 2 )]<<" "<<count<<" "<<jump;
	// 	// residual(d_residual,d_Arr,d_rho);
	// d_Arr=smoother(d_rho,d_Arr,10);
	// residual(d_residual,d_Arr,d_rho);
	// rstrict(9,9,9);

	//justtest<<<1,1>>>(d_ANS);
	//dim3 gridsize((((((gridSize.x-1)*blockSize.x)-1)/jump_x)+1)/blockSize.x +1,(((((gridSize.y-1)*blockSize.y)-1)/jump_y)+1)/blockSize.y +1, 1);
	// rstrict(9,9,9);
	//laplacian << <gridsize, blockSize >> > (d_Arr, d_ans,jump);
	// gridsize.x=(((((gridSize.x-1)*blockSize.x)-1)/jump_x)+1)/blockSize.x +1;
	// gridsize.y=(((((gridSize.y-1)*blockSize.y)-1)/jump_y)+1)/blockSize.y +1;
	// gridsize.z=1;
	// add<<<gridsize,blockSize>>>(d_ANS,d_ANS,d_ANS,jump_x,jump_y,jump_z);
	// interpolate(gridSize,blockSize,d_ANS);
	// cudaMemcpy(h_ANS, d_error, SIZE*sizeof(double), cudaMemcpyDeviceToHost);
	// int count=0;
	// // //cout<<jump;
	// for(int k=0;k<Z_SIZE;k+=jump)
	// {
	// 	for(int j=0;j<Y_SIZE;j+=jump)
	// 	{
	// 		for(int i=0;i<X_SIZE;i+=jump)
	// 			{
	// 				if(h_ANS[pos(i,j,k)]<0)
	// 					count++;
	// 				printf("%9.6lf ",h_ANS[pos(i,j,k)]);
	// 			}
	// 		cout<<endl;
	// 	}
	// 	cout<<endl<<endl;
	// }
	// cout << h_ANS[pos(X_SIZE / 2-2, Y_SIZE / 2, Z_SIZE / 2 )]<<" "<<count<<" "<<jump;
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
