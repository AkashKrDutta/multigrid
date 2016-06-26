
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
#define pos(x,y,z) ((x) + ((y)*X_SIZE) + ((z)*Y_SIZE*X_SIZE))		//macro to find the index of (x,y,z) in the grid
#define SIZE  X_SIZE*Y_SIZE*Z_SIZE 									//total size of grid
#define Max 500
#define EPSILON .0000000001
#define lx 9
#define ly 9
#define lz 9
#define ITER 300					//no. of iterations used to calculate error

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
__device__ double particle_interploate(int x_state,int y_state,int z_state,double * d_arr,int x_idx,int y_idx,int z_idx,int jump)
{
	double value=0.0;
	x_state*=jump;
	y_state*=jump;
	z_state*=jump;
	if(x_state + y_state +z_state ==jump)
	{

		value=(d_arr[pos(x_idx-x_state,y_idx-y_state,z_idx-z_state)]+d_arr[pos(x_idx+x_state,y_idx+y_state,z_idx+z_state)])/2.0;
	}
	else if(x_state + y_state +z_state ==3*jump)
	{
		value= (d_arr[pos(x_idx-x_state,y_idx-y_state,z_idx-z_state)]+d_arr[pos(x_idx+x_state,y_idx+y_state,z_idx+z_state)]+d_arr[pos(x_idx-x_state,y_idx-y_state,z_idx+z_state)]+
				d_arr[pos(x_idx-x_state,y_idx+y_state,z_idx-z_state)]+d_arr[pos(x_idx+x_state,y_idx-y_state,z_idx-z_state)]+d_arr[pos(x_idx+x_state,y_idx+y_state,z_idx-z_state)]+
				d_arr[pos(x_idx+x_state,y_idx-y_state,z_idx+z_state)]+d_arr[pos(x_idx-x_state,y_idx+y_state,z_idx+z_state)])/8.0;
	}
	else
	{
		if(!(x_state/jump))
		{
			value=(d_arr[pos(x_idx,y_idx-1*jump,z_idx-1*jump)]+d_arr[pos(x_idx,y_idx-1*jump,z_idx+1*jump)]+d_arr[pos(x_idx,y_idx+1*jump,z_idx-1*jump)]+d_arr[pos(x_idx,y_idx+1*jump,z_idx+1*jump)])/4.0;
		}
		else if(!(y_state/jump))
		{
			value=(d_arr[pos(x_idx-1*jump,y_idx,z_idx-1*jump)]+d_arr[pos(x_idx-1*jump,y_idx,z_idx+1*jump)]+d_arr[pos(x_idx+1*jump,y_idx,z_idx-1*jump)]+d_arr[pos(x_idx+1*jump,y_idx,z_idx+1*jump)])/4.0;
		}
		else if(!(z_state/jump))
		{
			value=(d_arr[pos(x_idx-1*jump,y_idx-1*jump,z_idx)]+d_arr[pos(x_idx+1*jump,y_idx-1*jump,z_idx)]+d_arr[pos(x_idx-1*jump,y_idx+1*jump,z_idx)]+d_arr[pos(x_idx+1*jump,y_idx+1*jump,z_idx)])/4.0;
		}
	}
	return value;
}

//to opitmize by calling no wastage of thread---reduced trhead divergence..
__global__ void dev_interpolate(double *d_arr,int jump)
{
	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
	x_idx*=jump;
	y_idx*=jump;
	int x_state=0;
	int y_state=0;
	int z_state=0;
	if(x_idx<X_SIZE && y_idx<Y_SIZE)
	{
		if((x_idx/jump)%2==1)
			x_state=1;
		if((y_idx/jump)%2==1)
			y_state=1;
		
		for(int i=0;i<Z_SIZE;i+=jump)
		{	
			if((i/jump)%2==1)
				z_state=1;
			if(x_state || y_state || z_state )
				d_arr[pos(x_idx,y_idx,i)]=particle_interploate(x_state,y_state,z_state,d_arr,x_idx,y_idx,i,jump);
			z_state=0;
		}
	}
}

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
	int i;
	if(x_idx<X_SIZE && y_idx<Y_SIZE){
		for (i = 0; i < Z_SIZE; i+=jump){
			ans[pos(x_idx, y_idx, i)] = funFinite(arr, x_idx, y_idx, i,jump);
		}
	}
}


__global__ void jacobi(double *d_tempphi, double * d_rho, double * d_laplacian, double *d_phi,int jump)
{
	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
	x_idx*=jump;
	y_idx*=jump;
	int i;
	if(x_idx<X_SIZE && y_idx <Y_SIZE){
		if( (x_idx <(X_SIZE-1)) && (y_idx<(Y_SIZE-1)) && (x_idx>0) && (y_idx>0)){
				for (i = jump; i < Z_SIZE-1; i+=jump){
					d_phi[pos(x_idx, y_idx, i)] = d_tempphi[pos(x_idx, y_idx, i)] + (-d_rho[pos(x_idx, y_idx, i)] + d_laplacian[pos(x_idx, y_idx, i)]) / (2*((1/(h_x*h_x))+(1/(h_y*h_y))+(1/(h_z*h_z))));
				}
			}
			
		else if(x_idx==0 || x_idx == X_SIZE-1 || y_idx == 0 || y_idx==Y_SIZE-1)
		{
			for(int i=jump;i<Z_SIZE-1;i+=jump)
				d_phi[pos(x_idx,y_idx,i)]=0;
		}
		d_phi[pos(x_idx,y_idx,0)]=0;
		d_phi[pos(x_idx,y_idx,Z_SIZE-1)]=0;
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

__global__ void absolute(double *d_Arr,double* abs_d_Arr)
{
	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
	int i;
	if(x_idx<X_SIZE && y_idx < Y_SIZE){
		for (i = 0; i < Z_SIZE; i++){
			abs_d_Arr[pos(x_idx, y_idx, i)] =abs(d_Arr[pos(x_idx, y_idx, i)]);
		}
	}
}
__global__ void all_zero(double *arr,int jump) //initiaizes to zero
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

double* smoother(double* d_rho,double * d_phi,int N)
{
	dim3 gridsize;
	gridsize.x=(((X_SIZE-1)/jump)+1)/blockSize.x + 1;
	gridsize.y=(((X_SIZE-1)/jump)+1)/blockSize.x + 1;
	gridsize.z=1;
	double * d_laplacian,* dummy,*d_tempphi;
	cudaMalloc((void**)&d_tempphi,sizeof(double)*(SIZE));
	cudaMalloc((void**)&d_laplacian,sizeof(double)*(SIZE));
	copy<<<gridsize,blockSize>>>(d_tempphi,d_phi,1);
	for(int i=0;i<N;i++)
	{
		laplacian << <gridsize, blockSize >> > (d_tempphi, d_laplacian,jump);
		cudaDeviceSynchronize();
		jacobi << <gridsize, blockSize >> > (d_tempphi, d_rho, d_laplacian, d_phi,jump);
		dummy = d_tempphi;
		d_tempphi = d_phi;
		d_phi=dummy;
	}
	d_phi=d_tempphi;
	cudaFree(d_laplacian);
	cudaFree(dummy);
	return d_phi;
}
double* interpolate(double * d_error)
{
	dim3 gridsize;
	gridsize.x=(((X_SIZE-1)/jump)+1)/blockSize.x + 1;
	gridsize.y=(((X_SIZE-1)/jump)+1)/blockSize.x + 1;
	gridsize.z=1;
	host_interpolate();
	dev_interpolate <<<gridsize,blockSize>>>(d_error,jump);
	return d_error;
}

void residual(double * d_residual,double *d_guess,double* d_rho)
{
	dim3 gridsize;
	gridsize.x=(((X_SIZE-1)/jump)+1)/blockSize.x + 1;
	gridsize.y=(((X_SIZE-1)/jump)+1)/blockSize.x + 1;
	gridsize.z=1;
	double * d_laplace;
	cudaMalloc((void**)&d_laplace,sizeof(double)*(SIZE));
	laplacian<<<gridsize,blockSize>>>(d_guess,d_laplace,1);
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
	double h_ans[1],*ans;
	cudaMalloc((void**)&d_ans,sizeof(double)*(SIZE));
	cudaMalloc((void**)&d_sub,sizeof(double)*(SIZE));
	cudaMalloc((void**)&d_Arr,sizeof(double)*(SIZE));
	copy<<<gridsize,blockSize>>>(d_Arr,d_error,jump);
	all_zero<<<gridsize,blockSize>>>(d_sub,1);
	do{
		laplacian << <gridsize, blockSize >> > (d_Arr, d_ans,jump);
		jacobi << <gridsize, blockSize >> > (d_Arr, d_residual, d_ans, d_error,jump);
		abs_subtract << <gridsize, blockSize >> >(d_Arr, d_error, d_sub,jump);

		findmax<<<1,1>>>(d_sub,ans,jump);
		cudaMemcpy(h_ans,ans,sizeof(double),cudaMemcpyDeviceToHost);
		max_error=*h_ans;
		cout<<"hehe "<<max_error<<endl;
		dummy=d_Arr;
		d_Arr=d_error;
		d_error=dummy;
	}while(max_error>eps);

	d_error=d_Arr;
	cudaFree(d_ans);
	cudaFree(d_sub);
	return d_error;
}
double* new_solver(double* d_residual,double * d_error,int N)
{
	dim3 gridsize;
	gridsize.x=(((X_SIZE-1)/jump)+1)/blockSize.x + 1;
	gridsize.y=(((X_SIZE-1)/jump)+1)/blockSize.x + 1;
	gridsize.z=1;
	double * d_ans,* dummy,*d_Arr;
	cudaMalloc((void**)&d_Arr,sizeof(double)*(SIZE));
	cudaMalloc((void**)&d_ans,sizeof(double)*(SIZE));
	copy<<<gridsize,blockSize>>>(d_Arr,d_error,1);
	for(int i=0;i<N;i++)
	{
		laplacian << <gridsize, blockSize >> > (d_Arr, d_ans,jump);
		jacobi << <gridsize, blockSize >> > (d_Arr, d_residual, d_ans, d_error,jump);
		dummy = d_Arr;
		d_Arr = d_error;
		d_error=dummy;
	}
	d_error=d_Arr;
	return d_error;
}

__global__ void add(double * d1,double* d2,double* sum,int jump)
{
	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
	x_idx*=jump;
	y_idx*=jump;
	if(x_idx<X_SIZE && y_idx < Y_SIZE)
	{
		for(int i=0;i<Z_SIZE;i+=jump)
			sum[pos(x_idx,y_idx,i)]=d1[pos(x_idx,y_idx,i)]+d2[pos(x_idx,y_idx,i)];
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
	smoother(d_rho,d_Arr,N);
	double * d_residual,max_error;
	cudaMalloc((void**)&d_residual,sizeof(double)*SIZE);
	residual(d_residual,d_Arr,d_rho);
	rstrict(rx,ry,rz);
	solver(d_residual,d_error,error);//init error to all 0 before calling Vcylce
	while(jump>1)
		interpolate(d_error);
	add<<<gridSize,blockSize>>>(d_error,d_Arr,d_Arr,jump);
	residual(d_residual,d_Arr,d_rho);
	thrust::device_ptr<double> dev_ptr(d_residual);
	thrust::device_ptr<double> devsptr=(thrust::max_element(dev_ptr, dev_ptr + SIZE));
	max_error=*devsptr;
	cout<<"Current error= "<<max_error;
}

__global__ void do_negative(double * d_Arr)
{
	int x_idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	int y_idx = (blockIdx.y*blockDim.y) + threadIdx.y;
	if(x_idx<X_SIZE && y_idx < Y_SIZE)
	{
		for(int i=0;i<Z_SIZE;i+=1)
			if(d_Arr[pos(x_idx,y_idx,i)]!=0)
			d_Arr[pos(x_idx,y_idx,i)]=-d_Arr[pos(x_idx,y_idx,i)];
	}
}

int main()
{
	//declaration of host variables
	double guess[SIZE];
	double h_rho[SIZE];
	double h_ANS[SIZE];

	//initialization of h_rho and guess array
	for (int k = 0; k<Z_SIZE; ++k)
		for (int j = 0; j<Y_SIZE; ++j)
			for (int i = 0; i<X_SIZE; ++i)
			{
				guess[(i)+(j*X_SIZE) + (k*Y_SIZE*X_SIZE)] = 0;
				h_rho[(i)+(j*X_SIZE) + (k*Y_SIZE*X_SIZE)] = 0;
			}
	h_rho[pos(X_SIZE / 2-4, Y_SIZE / 2, Z_SIZE/2 )] = 100;
	h_rho[pos(X_SIZE / 2+4, Y_SIZE / 2, Z_SIZE / 2 )] = -100;
	
	//declaring the no. of blocks
	dim3 gridsize;
	gridsize.x=(((X_SIZE-1)/jump)+1)/blockSize.x + 1;
	gridsize.y=(((X_SIZE-1)/jump)+1)/blockSize.x + 1;
	gridsize.z=1;

	//declaration of device variables
	double *d_guess;
	double *d_ans;
	double *d_rho;
	double *d_ANS;
	double *d_sub;
	double* d_error;
	double * d_residual,max_error;

	//malloc
	cudaMalloc((void**)&d_guess, SIZE*sizeof(double));
	cudaMalloc((void**)&d_ans, SIZE*sizeof(double));
	cudaMalloc((void**)&d_rho, SIZE*sizeof(double));
	cudaMalloc((void**)&d_sub, SIZE*sizeof(double));
	cudaMalloc((void**)&d_ANS, SIZE*sizeof(double));
	cudaMalloc((void**)&d_error, SIZE*sizeof(double));
	cudaMalloc((void**)&d_residual,sizeof(double)*SIZE);

	//copying memory from host to device
	cudaMemcpy(d_guess, guess, SIZE*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rho, h_rho, SIZE*sizeof(double), cudaMemcpyHostToDevice);
	
	//initializing error array with 0
	all_zero<<<gridsize,blockSize>>>(d_error,1);

	
	for(int i=0;i<ITER;i++)
	{
		cudaDeviceSynchronize();
		jump=1;
		d_guess=smoother(d_rho,d_guess,10);
		cudaDeviceSynchronize();
		
		residual(d_residual,d_guess,d_rho);
		cudaDeviceSynchronize();
		
		rstrict(9,9,9);
		cudaDeviceSynchronize();
		
		d_error=new_solver(d_residual,d_error,30);
		cudaDeviceSynchronize();
		
		while(jump>1)
		{	
			d_error=interpolate(d_error);
			cudaDeviceSynchronize();
		}
		cudaDeviceSynchronize();
		
		d_error=smoother(d_residual,d_error,10);		
		cudaDeviceSynchronize();
		add<<<gridSize,blockSize>>>(d_error,d_guess,d_guess,1);
		
		cudaDeviceSynchronize();
		residual(d_residual,d_guess,d_rho);
		
		cudaDeviceSynchronize();
		boundary_zero<<<gridSize,blockSize>>>(d_residual);
		
		cudaDeviceSynchronize();
		absolute<<<gridSize,blockSize>>>(d_residual,d_residual);
		
		//calculating maximum value of error in the grid
		cudaDeviceSynchronize();
		thrust::device_ptr<double> dev_ptr(d_residual);
		cudaDeviceSynchronize();
		thrust::device_ptr<double> devsptr=(thrust::max_element(dev_ptr, dev_ptr + SIZE));
		cudaDeviceSynchronize();
		max_error=*devsptr;
		cout<<"Current Error = "<<max_error<<endl;
		cudaDeviceSynchronize();
	}

	do_negative<<<gridsize,blockSize>>>(d_guess);
	cudaDeviceSynchronize();
	
	int inc=jump;
	jump=2;
	cudaDeviceSynchronize();
	
	cudaMemcpy(h_ANS, d_guess, SIZE*sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	int count=0;
	
	//printing of x-y plane
	for(int k=0;k<Z_SIZE;k+=jump)
	{
		for(int j=0;j<Y_SIZE;j+=jump)
		{
			for(int i=0;i<X_SIZE;i+=jump)
				{
					if(h_ANS[pos(i,j,k)]<0)
						count++;
					printf("%9.6lf ",h_ANS[pos(i,j,k)]);
				}
			cout<<endl;
		}
		cout<<endl<<endl;

	}
	
	jump=inc;
	cudaDeviceSynchronize();
	
	cout << h_ANS[pos(X_SIZE / 2-4, Y_SIZE / 2, Z_SIZE / 2 )]<<" "<<count<<" "<<jump;
	
	cudaFree(d_guess);
	cudaFree(d_ans);
	cudaFree(d_ANS);
	cudaFree(d_rho);
	cudaFree(d_sub);
	cudaFree(d_residual);
	cudaFree(d_error);

	cout<<endl;
	return 0;
}
