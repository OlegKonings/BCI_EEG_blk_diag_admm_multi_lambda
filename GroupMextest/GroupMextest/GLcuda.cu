#include "stdafx.h"
#include "stdio.h"
#include <cuda.h>
#include <math_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 16
#define BLOCKSIZE BLOCK_SIZE
#define AVOIDBANKCONFLICTS 0
#define USELOOPUNROLLING 1
#define TESTBLOCKS 16
#define IDC2D(i,j,ld) (((j)*(ld))+(i))

#define THREADS 64//this is 64 because for this version of ADMM group lasso, data sets will be small. For later data sets use 256
//make sure matches cpp CPPTHREADS
#define Z_THREADS 256
#define LINEAR_BLOCK_SIZE THREADS

//const int blockSizeLocal=64;

//general use kernels
/////////////////////////////////////////////////////////////////////////////////////
__global__ void generateEye(float *E, const int size){
    int offset = blockIdx.x*blockDim.x + threadIdx.x;
	if(offset<(size*size)){
		int y = offset/size,x = offset - y*size;
		E[offset] = (x == y) ? 1.0f:0.0f;
	}
}
extern "C" void generateEye_wrap(float *E, const int N,const int numBlocks){
	generateEye<<<numBlocks,THREADS>>>(E,N);
	
}
//////////////////////////////////////////////////////////////////////////////////////

__global__ void set_ATA(const float* __restrict__ ATA, float* __restrict__ TempATA, const int N,
	const int padd_N){
		const int i=blockIdx.y;//0 to padd_N
		const int j=threadIdx.x+blockIdx.x*blockDim.x;//0 to padd_N
		if(j<padd_N){
			TempATA[i*padd_N+j]= (j<N && i<N) ? ATA[i*N+j]:(1.0f*float(int(i==j)));
		}
}
extern "C" void pad_ATA(const float *ATA, float *TempATA,  const int N,
	const int padd_N){
		dim3 grid((padd_N+THREADS-1)/THREADS,padd_N,1);
		set_ATA<<<grid,THREADS>>>(ATA,TempATA,N,padd_N);
		

}

//////////////////////////////////////////////////////////////////////////////////////
__global__ void d_choldc_topleft(float *M, int boffset,const int N){
    const int tx = threadIdx.x,ty = threadIdx.y;

    __shared__ float topleft[BLOCK_SIZE][BLOCK_SIZE+1];
	int idx0=ty+BLOCK_SIZE*boffset,adj=tx+BLOCK_SIZE*boffset;

    topleft[ty][tx]=M[idx0*N+adj];
    __syncthreads();

    float fac;
    for(int k=0;k<BLOCK_SIZE;k++){
		__syncthreads();
		fac=rsqrtf(topleft[k][k]);
		//__syncthreads();
		if((ty==k)&&(tx>=k)){
			topleft[tx][ty]=(topleft[tx][ty])*fac;
		}
		__syncthreads();
		if ((ty>=tx)&&(tx>k)){
			topleft[ty][tx]=topleft[ty][tx]-topleft[tx][k]*topleft[ty][k]; 
		}
	}

    __syncthreads();
// here, tx labels column, ty row	
    if(ty>=tx){
		M[idx0*N+adj]=topleft[ty][tx];
    }
}
extern "C" void d_choldc_topleft_wrap(float *M, int boffset,const int N,const dim3 t_block){
	d_choldc_topleft<<<1,t_block>>>(M,boffset,N);
	
}
//////////////////////////////////////////////////////////////////////////////////////



__global__ void d_choldc_strip(float *M,int boffset,const int N){
// +1 since blockoffset labels the "topleft" position
// and boff is the working position...
    const int boffx = blockIdx.x+boffset+1; 
    const int tx = threadIdx.x,ty = threadIdx.y;
	int idx0=ty+BLOCK_SIZE*boffset,adj=tx+BLOCK_SIZE*boffset;
	int idx1=ty+boffx*BLOCK_SIZE,adj1=tx+boffset*BLOCK_SIZE;

    __shared__ float topleft[BLOCK_SIZE][BLOCK_SIZE+1];
    __shared__ float workingmat[BLOCK_SIZE][BLOCK_SIZE+1];

    topleft[ty][tx]=M[idx0*N+adj];
// actually read in transposed...
    workingmat[tx][ty]=M[idx1*N+adj1];

    __syncthreads();
    // now we forward-substitute for the new strip-elements...
    // one thread per column (a bit inefficient I'm afraid)
    if(ty==0){
		float dotprod;
		for(int k=0;k<BLOCK_SIZE;k++){
			dotprod=0.0f;
			for (int m=0;m<k;m++){
				dotprod+=topleft[k][m]*workingmat[m][tx];
			}
			workingmat[k][tx]=(workingmat[k][tx]-dotprod)/topleft[k][k];
		}
    }
    __syncthreads();
// is correctly transposed...
    M[idx1*N+adj1]=workingmat[tx][ty];
}
extern "C" void d_choldc_strip_wrap(float *M, int boffset,const int N,const dim3 stripgrid,const dim3 t_block){
	d_choldc_strip<<<stripgrid,t_block>>>(M,boffset,N);
	
}
///////////////////////////////////////////////////////////////////////////

__global__ void adj_chol(const float* __restrict__ Pad_L, float* __restrict__ L,  const int N,const int padd_N){
		const int i=blockIdx.y;
		const int j=threadIdx.x+blockIdx.x*blockDim.x;
		if(j<N){
			L[i*N+j]=Pad_L[i*padd_N+j];
		}
}
extern "C" void get_L(const float *Pad_L, float *L, const int N,const int padd_N){
		dim3 grid((N+THREADS-1)/THREADS,N,1);
		adj_chol<<<grid,THREADS>>>(Pad_L,L,N,padd_N);
		
}


///////////////////////////////////////////////////////////////////////////
__global__ void update_q(const float* __restrict__ Atb, const float* __restrict__  z, const float* __restrict__ u,
	float* __restrict__ q, const float rho, const int length,const int mask){

		if(mask&(1<<blockIdx.y))return;

		const int offset=threadIdx.x+blockIdx.x*blockDim.x;
		const int index=blockIdx.y*length+offset;
		if(offset<length){
			q[index]=(Atb[offset]+rho*(z[index]-u[index]));
		}
}
extern "C" void update_vector_q(const float *Atb, const float *z, const float *u, float *q, const float rho,const int length,
	const dim3 &grid,const int mask){
	update_q<<<grid,THREADS>>>(Atb,z,u,q,rho,length,mask);
	//cudaError_t err=cudaThreadSynchronize();
	//if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
}

__global__ void finish_x_fat(const float* __restrict__ q, float* __restrict__ x, const float rho,const int length,
	const int mask){

		if(mask&(1<<blockIdx.y))return;

		const int offset=threadIdx.x+blockIdx.x*blockDim.x;
		const int index=blockIdx.y*length+offset;
		if(offset<length){		
			x[index]/=-(rho*rho);
			x[index]+=(q[index]/rho);
		}
}
extern "C" void finish_all_x_fat(const float *q, float *x, const float rho, const int length, const dim3 &grid,
	const int mask){
	finish_x_fat<<<grid,THREADS>>>(q,x,rho,length,mask);
	
}

////////////////////////////////////////////////////////////////////////////////////

__global__ void update_x_hat_multi(const float* __restrict__ x, const float* __restrict__ z, float* __restrict__ x_hat,
	const int length, const float alpha,const int mask){

		if(mask&(1<<blockIdx.y))return;

		const int offset=threadIdx.x+blockIdx.x*blockDim.x;
		const int index=blockIdx.y*length+offset;
		if(offset<length){
			x_hat[index]=alpha*x[index]+(1.0f-alpha)*z[index];
		}

}
extern "C" void x_hat_update_helper(const float *x, const float *zold, float *x_hat, const int length, const float alpha, const dim3 &grid,
	const int mask){
		update_x_hat_multi<<<grid,THREADS>>>(x,zold,x_hat, length, alpha,mask);
		
}
//////////////////////////////////////////////////////////////////////////////////////


//the first step gets the norm (without sqrt part) of each x_hat and u vector per each lambda
__global__ void GPU_step0(const float* __restrict__ x_hat,const float* __restrict__ u, float* __restrict__ nrms, 
	const int Psize,const int admm_blocks_size,const int length,const int mask){

		if(mask&(1<<blockIdx.y))return;
		//have each thread do admm block size of work
		const int start=threadIdx.x*admm_blocks_size;
		const int index=blockIdx.y*length+start;
		if(threadIdx.x<Psize){
			float val=0.0f,tmp;

			for(int i=0;i<admm_blocks_size;i++){
				tmp=x_hat[index+i]+u[index+i];
				val+=(tmp*tmp);
			}

			nrms[blockIdx.y*Psize+threadIdx.x]=sqrtf(val);//check, different method that general case
		}
}
//the second step updates the z sections
__global__ void GPU_step1(const float* __restrict__ x_hat,const float* __restrict__ u, const float* __restrict__ nrms, 
	float* __restrict__ z,const float* __restrict__ lam_arr,const int Psize,const int admm_blocks_size,
	const int mask,const int length,const float _rho){

		if(mask&(1<<blockIdx.y))return;

		__shared__ float cur_lam_div_rho;
		if(threadIdx.x==0){
			cur_lam_div_rho=(lam_arr[blockIdx.y]/_rho);
		}
		__syncthreads();

		const int start=threadIdx.x*admm_blocks_size;
		const int index=blockIdx.y*length+start;
		if(threadIdx.x<Psize){
			const float posknormzu= (nrms[blockIdx.y*Psize+threadIdx.x]>0.0f) ? max(0.0f,(1.0f-(cur_lam_div_rho/nrms[blockIdx.y*Psize+threadIdx.x]) ) ):0.0f;

			for(int i=0;i<admm_blocks_size;i++){
				z[index+i]=(x_hat[index+i]+u[index+i])*posknormzu;
			}

		}
}
//NOTE: assumes equal sub-sections of p and Psize is LESS than Z_THREADS(default 256)
extern "C" void z_shrinkage_wrap(float *D_z,const float *x_hat, const float *D_u,float *norm_s, const float *lam_arr,const int Psize,
	const int admm_blocks_size, const int num_lambdas, const float _rho,const int length,const int mask){

		dim3 Z_grid(1,num_lambdas,1);
		GPU_step0<<<Z_grid,Z_THREADS>>>(x_hat,D_u,norm_s,Psize,admm_blocks_size,length,mask);
	
		GPU_step1<<<Z_grid,Z_THREADS>>>(x_hat,D_u,norm_s,D_z,lam_arr,Psize,admm_blocks_size,mask,length,_rho);
	
}
///////////////////////////////////////////////////////////////////////////////

__global__ void gpu_lasso_u_update(float* __restrict__ u,const float* __restrict__ xh, const float* __restrict__ z,const int length,
	const int mask){

		if(mask&(1<<blockIdx.y))return;

		const int offset=threadIdx.x+blockIdx.x*blockDim.x;
		const int index=blockIdx.y*length+offset;
		if(offset<length){
			u[index]+=(xh[index]-z[index]);
		}
}
extern "C" void gpu_lasso_u_update_wrap(float *u,const float *xh, const float *z,const int length,const dim3 &grid,
	const int mask){
	gpu_lasso_u_update<<<grid,THREADS>>>(u,xh,z,length,mask);
	
}

/////////////////////////////////////////////////////////////////////////////////////
//need norm of x, -z , (x-z), -rho*(z-zold), rho*u
__global__ void _get_norm_all(const float* __restrict__ x, const float * __restrict__ z, const float* __restrict__ zold, const float* __restrict__ u,
	float* __restrict__ xnorm, float* __restrict__ znorm, float* __restrict__ xznorm, float* __restrict__ rzznorm,float* __restrict__ runorm,
	const int length, const float _rho,const int mask){

		if(mask&(1<<blockIdx.y))return;

		const int offset=threadIdx.x+blockIdx.x*blockDim.x;
		const int warp_idx=threadIdx.x%32;
		__shared__ float x_sq_sum[2],
			z_sq_sum[2],
			x_minus_z_sum[2],
			neg_rho_zzold_sum[2],
			rho_u_sum[2];

		float xx=0.0f,zz=0.0f,xz=0.0f, zzold=0.0f,ru=0.0f, tmp=0.0f;

		if(offset<length){
			xx=x[blockIdx.y*length+offset];
			zz=z[blockIdx.y*length+offset];
			xz=(xx-zz)*(xx-zz);//rnorm calc
			tmp= -_rho*(zz-zold[blockIdx.y*length+offset]);
			zzold=tmp*tmp;//snorm calc
			xx*=xx;//norm x
			zz*=zz;//norm (-z)
			ru=_rho*u[blockIdx.y*length+offset];
			ru*=ru;//norm (rho*u)
		}
		for(int ii=16;ii>0;ii>>=1){
			xx += __shfl(xx, warp_idx + ii);
			zz += __shfl(zz, warp_idx + ii);
			xz += __shfl(xz, warp_idx + ii);
			zzold += __shfl(zzold, warp_idx + ii);
			ru += __shfl(ru, warp_idx + ii);
			
		}
		if(warp_idx==0){
			x_sq_sum[threadIdx.x>>5]=xx;
			z_sq_sum[threadIdx.x>>5]=zz;
			x_minus_z_sum[threadIdx.x>>5]=xz;
			neg_rho_zzold_sum[threadIdx.x>>5]=zzold;
			rho_u_sum[threadIdx.x>>5]=ru;		
		}
		__syncthreads();

		if(threadIdx.x==0){
			atomicAdd(&xnorm[blockIdx.y],(x_sq_sum[0]+x_sq_sum[1]));
			atomicAdd(&znorm[blockIdx.y],(z_sq_sum[0]+z_sq_sum[1]));
			atomicAdd(&xznorm[blockIdx.y],(x_minus_z_sum[0]+x_minus_z_sum[1]));
			atomicAdd(&rzznorm[blockIdx.y],(neg_rho_zzold_sum[0]+neg_rho_zzold_sum[1]));
			atomicAdd(&runorm[blockIdx.y],(rho_u_sum[0]+rho_u_sum[1]));

		}

}

extern "C" void get_multi_norms(const float *x, const float *z, const float *zold, const float *u,float *norm_arr,
	const float _rho, const int length,const dim3 &grid,const int num_lambdas,const int mask){

		_get_norm_all<<<grid,THREADS>>>(x,z,zold,u,(float*)&norm_arr[0], (float*)&norm_arr[num_lambdas],
			(float*)&norm_arr[num_lambdas*2],(float*)&norm_arr[num_lambdas*3],
			(float*)&norm_arr[num_lambdas*4], length,_rho,mask);
		
}

__global__ void fill_Row_Ptr(int* __restrict__ csrRowPtrA,const int num_per_row,const int N){
	const int offset=threadIdx.x+blockIdx.x*blockDim.x;//will rand from 0 to N inclusive
	if(offset>N)return;
	csrRowPtrA[offset]=offset*num_per_row;
}
extern "C" void fill_Row_Ptr_helper(int *csrRowPtrA,const int num_per_row,const int N){
	fill_Row_Ptr<<<(N+THREADS)/THREADS,THREADS>>>(csrRowPtrA,num_per_row,N);
}

__global__ void rep_mat_diag_to_CSR(const float* __restrict__ subA,float* __restrict__ csrValA, int* __restrict__ csrColIndA,
	const int n, const int num_ROIs, const int N){
		const int i=threadIdx.x+blockIdx.x*blockDim.x;//will range from 0 to n
		const int j=blockIdx.y;//will range from 0 to n
		if(i<n){
			const float val=subA[j*n+i];
			int idx,col,num_elem;

			for(int k=0;k<num_ROIs;k++){
				idx=N*(k*n+i)+k*n+j;
				col=idx%N;
				num_elem=k*(n*n)+(i*n+j);
				csrValA[num_elem]=val;
				csrColIndA[num_elem]=col;
			}

		}
}
extern "C" void rep_mat_diag_to_CSR_helper(const float *subA, float *csrValA,int *csrColIndA,const int n, const int num_ROIs, const int N){
	dim3 Test_Grid((n+THREADS-1)/THREADS,n,1);
	rep_mat_diag_to_CSR<<<Test_Grid,THREADS>>>(subA,csrValA,csrColIndA,n,num_ROIs,N);
}