#include <cstring>
#include <cmath>
#include "stdafx.h"
#include "stdio.h"
#include "mex.h"
#include "matrix.h"
#include <cuda.h>//CUDA version 5.0 SDK, 64 bit
//#include <math_functions.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cassert>

#define _DTH cudaMemcpyDeviceToHost
#define _DTD cudaMemcpyDeviceToDevice
#define _HTD cudaMemcpyHostToDevice

//make sure these match the .cu THREADS and BLOCK_SIZE
#define CPPTHREADS 64//This is 64 due to expectation that data sets will be small, if (m*n)>= 1e6 then use 256 THREADS(adjust here and in .cu file)
#define CPPBLOCK_SIZE 16
#define MEGA (1<<8)
#define MAX_LAMBDAS 32


void inline checkError(cublasStatus_t status, const char *msg){if (status != CUBLAS_STATUS_SUCCESS){printf("%s", msg);exit(EXIT_FAILURE);}}

//NOTE: all cublas calls are done in this cpp, and all CUDA kernels are in GLcuda.cu, and accessed via extern "C"
// using wraps for all GPU kernel calls

extern "C" void generateEye_wrap(float *E, const int N,const int numBlocks);

extern "C" void pad_ATA(const float *ATA, float *TempATA,  const int N,const int padd_N);

extern "C" void d_choldc_topleft_wrap(float *M, int boffset,const int N,const dim3 t_block);

extern "C" void d_choldc_strip_wrap(float *M, int boffset,const int N,const dim3 stripgrid,const dim3 t_block);

extern "C" void get_L(const float *Pad_L, float *L, const int N,const int padd_N);

extern "C" void update_vector_q(const float *Atb, const float *z, const float *u, float *q, const float rho,const int length,
	const dim3 &grid,const int mask);

extern "C" void finish_all_x_fat(const float *q, float *x, const float rho, const int length, const dim3 &grid,
	const int mask);

extern "C" void x_hat_update_helper(const float *x, const float *zold, float *x_hat, const int length, const float alpha, const dim3 &grid,
	const int mask);

extern "C" void z_shrinkage_wrap(float *D_z,const float *x_hat, const float *D_u,float *norm_s, const float *lam_arr,const int Psize,
	const int admm_blocks_size, const int num_lambdas, const float _rho,const int length,const int mask);

extern "C" void gpu_lasso_u_update_wrap(float *u,const float *xh, const float *z,const int length,const dim3 &grid,
	const int mask);

extern "C" void get_multi_norms(const float *x, const float *z, const float *zold, const float *u,float *norm_arr,
	const float _rho, const int length,const dim3 &grid,const int num_lambdas,const int mask);

extern "C" void fill_Row_Ptr_helper(int *csrRowPtrA,const int num_per_row,const int N);

extern "C" void rep_mat_diag_to_CSR_helper(const float *subA, float *csrValA,int *csrColIndA,const int n, const int num_ROIs, const int N);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]){//will take in 12 inputs and return 3 outputs
//will take in 12 inputs and return 3 outputs
	/*inputs are (in order)
	0) sub Matrix A, 32 bit float in passed in TRANSPOSE state, of dimensions (m,n)
	1) vector b (M,1) single precision floating point numbers
	2) vector p (Psize length) 32 bit integer of K(Psize) length (partitions)
	3) vector u (N,num_lambdas) single precision floating point numbers
	4) vector z (N,num_lambdas) single precision floating point numbers
	5) float (single) rho
	6) float (single) alpha
	7) integer max_iter
	8) float (single) abstol
	9) float (single) reltol
	10) lambda array(single) with limit of 32 lambdas
	11) num ROIs (32 bit int)

	outputs are (in order)
	0) vector u (n,lambdas) single precision floating point numbers
	1) vector z (n,lambdas) single precision floating point numbers
	2) vector iter (num_lambdas,1) int 32

	*/
	//get parameters of inputs
	float *A=(float *)mxGetPr(prhs[0]);//matrix A in MATLAB column-major format
	const int Arows=(int)mxGetN(prhs[0]);//since this Group Lasso solver internally uses row major, need to pass in input A as A' from Matlab, then swap (m,n)
	const int Acols=(int)mxGetM(prhs[0]);//ditto, swaping due to A' being passed in to mex interface

	float *b=(float *)mxGetPr(prhs[1]);
	int *p=(int *)mxGetPr(prhs[2]);
	float *u=(float *)mxGetPr(prhs[3]);//vector u
	float *z=(float *)mxGetPr(prhs[4]);//vector z
	float *lambda_array=(float *)mxGetPr(prhs[10]);
	const int Psize=(int)mxGetM(prhs[2]);

	const float _rho=(float)mxGetScalar(prhs[5]);
	const float _alpha=(float)mxGetScalar(prhs[6]);
	const int max_iter=(int)mxGetScalar(prhs[7]);
	const float abstol=(float)mxGetScalar(prhs[8]);
	const float reltol=(float)mxGetScalar(prhs[9]);
	const int num_lambdas=(int)mxGetM(prhs[10]);
	const int num_ROIs=(int)mxGetScalar(prhs[11]);

	const int adj_Arows=(((Arows+16-1)>>4)<<4),adj_Acols=(((Acols+16-1)>>4)<<4);//padding for Cholesky factor

	cublasHandle_t handle;//init cublas_v2
	cublasStatus_t cur;
	cur = cublasCreate(&handle);
	if(cur!=CUBLAS_STATUS_SUCCESS){printf("error code %d, line(%d)\n", cur, __LINE__);exit(EXIT_FAILURE);}

	const int admm_blk_size=p[0];//assumes all blocks are same size, if not there will be issues (use general case version instead)

	const float _beta=0.0f,t_alphA=1.0f;
	const unsigned int numbytesM=Arows*Acols*sizeof(float);

	float *D_A,*L,*tmpM2,*tmpM3,*D_b,*D_xresult,*D_Atb,*D_u,*D_z,*tempvecC,*norms_array,*x_hat;
	float *norm_s,*lam_arr;
	
	const int BigRows=Arows*num_ROIs,BigCols=Acols*num_ROIs;
	const int numbytesVC=num_lambdas*BigCols*sizeof(float);
	cudaError_t err=cudaMalloc((void **)&D_A,numbytesM);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	cudaMalloc((void **)&D_b,BigRows*sizeof(float));//does not need to be mulit-dimensional
	cudaMalloc((void **)&D_xresult,numbytesVC);
	cudaMalloc((void **)&D_Atb,BigCols*sizeof(float));
	cudaMalloc((void **)&D_u,numbytesVC);
	cudaMalloc((void **)&D_z,numbytesVC);
	cudaMalloc((void **)&tempvecC,numbytesVC);
	cudaMalloc((void **)&norms_array,(num_lambdas*6)*sizeof(float));
	cudaMalloc((void **)&x_hat,numbytesVC);

	const int num_k_bytes=num_lambdas*Psize*sizeof(float);

	cudaMalloc((void **)&norm_s,num_k_bytes);
	cudaMalloc((void **)&lam_arr,num_lambdas*sizeof(float));

	const bool skinny= (Arows>=Acols);
	const int N= (skinny) ? Acols:Arows;
	const int regN= (skinny) ? adj_Acols:adj_Arows;
	const int numBytesT=N*N*sizeof(float),numBlocks=((N*N + CPPTHREADS-1)/CPPTHREADS);
	
	cudaMalloc((void **)&L,numBytesT);
	cudaMalloc((void **)&tmpM2,numBytesT);
	cudaMalloc((void **)&tmpM3,regN*regN*sizeof(float));

	generateEye_wrap(tmpM2,N,numBlocks);

	cudaMemcpy(L,tmpM2,numBytesT,_DTD);
	cudaMemcpy(D_A,A,numbytesM,_HTD);
	cudaMemcpy(D_b,b,BigRows*sizeof(float),_HTD);
	cudaMemcpy(D_u,u,numbytesVC,_HTD);
	cudaMemcpy(D_z,z,numbytesVC,_HTD);
	cudaMemcpy(lam_arr,lambda_array,num_lambdas*sizeof(float),_HTD);

	cudaMemset(D_xresult,0,numbytesVC);

	const int n_streams=max(num_ROIs,num_lambdas);
	int multi=0;
	cudaStream_t *streams = (cudaStream_t *)malloc(n_streams*sizeof(cudaStream_t));
	for(;multi<n_streams;multi++){
		cudaStreamCreate(&(streams[multi]));
	}

	//Atb = A'*b, in num_ROIs groups using cuda streams
	for(multi=0;multi<num_ROIs;multi++){
		cublasSetStream(handle,streams[multi]);	
		cublasSgemv_v2(handle,CUBLAS_OP_N,Acols,Arows,&t_alphA,D_A,Acols,D_b+multi*Arows,1,&_beta,D_Atb+multi*Acols,1);
	}
	//will be 'fat' but just in case
	const float _InvRho=1.0f/_rho;
	if(skinny){//A'*A + rho*eye(n)
		cublasSgemm_v2(handle,CUBLAS_OP_N,CUBLAS_OP_T,Acols,Acols,Arows,&t_alphA,D_A,Acols,D_A,Acols,&_rho,tmpM2,Acols);//tmpM2=AT*A+rho*eyeMatrix(tmpM2)
		//if(cur!=CUBLAS_STATUS_SUCCESS){printf("error code %d, line(%d)\n", cur, __LINE__);exit(EXIT_FAILURE);}
	}else{//speye(m) + 1/rho*(A*A')
		
		cublasSgemm_v2(handle,CUBLAS_OP_T,CUBLAS_OP_N,Arows,Arows,Acols,&_InvRho,D_A,Acols,D_A,Acols,&t_alphA,tmpM2,Arows);//tmpM2=1/rho*(A*AT)+eye(Arows)
		//if(cur!=CUBLAS_STATUS_SUCCESS){printf("error code %d, line(%d)\n", cur, __LINE__);exit(EXIT_FAILURE);}
	}
	pad_ATA(tmpM2,tmpM3,N,regN);//padd out matrix for Cholesky factor
	
	//Now cholesky factor of tmpM2
	dim3 threads(CPPBLOCK_SIZE,CPPBLOCK_SIZE);
	int todo=(regN/CPPBLOCK_SIZE);
	int reps=todo,k=CPPBLOCK_SIZE;
	float al=-1.0f,ba=1.0f;
	int n,rloc,cloc,cloc2;
	dim3 stripgrid(1,1,1);
	while(reps>2){
		stripgrid.x=reps-1;
		d_choldc_topleft_wrap(tmpM3,todo-reps,regN,threads);//d_choldc_topleft<<<1,threads>>>(tmpM3,todo-reps,regN)
		d_choldc_strip_wrap(tmpM3,todo-reps,regN,stripgrid,threads);//d_choldc_strip<<<stripgrid,threads>>>(tmpM3,todo-reps,regN)
		n=CPPBLOCK_SIZE*(reps-1);
		rloc=(CPPBLOCK_SIZE*(todo-reps+1))*regN;
		cloc=CPPBLOCK_SIZE*(todo-reps);
		cloc2=CPPBLOCK_SIZE*(todo-reps+1);
		cublasSsyrk_v2(handle,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T,n,k,&al,(float *)&tmpM3[rloc+cloc],regN,&ba,(float *)&tmpM3[rloc+cloc2],regN);
		reps--;
	}
	if(todo>1){
		stripgrid.x=1;
		d_choldc_topleft_wrap(tmpM3,todo-2,regN,threads);//d_choldc_topleft<<<1,threads>>>(tmpM3,todo-2,regN)
		d_choldc_strip_wrap(tmpM3,todo-2,regN,stripgrid,threads);//d_choldc_strip<<<1,threads>>>(tmpM3,todo-2,regN)
		cublasSsyrk_v2(handle,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T,k,k,&al,(float *)&tmpM3[(k*(todo-1))*regN +(k*(todo-2))],regN,&ba,(float *)&tmpM3[(k*(todo-1))*regN +(k*(todo-1))],regN);	
	}
	d_choldc_topleft_wrap(tmpM3,todo-1,regN,threads);//d_choldc_topleft<<<1,threads>>>(tmpM3,todo-1,regN)
	get_L(tmpM3,tmpM2,N,regN);
	//now have cholesky with padding, going forward using N

	cublasStrsm_v2(handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,N,N,&t_alphA,tmpM2,N,L,N);

	//tmpM2 will have value (inv(U))*inv(L))
	cublasSgemm_v2(handle,CUBLAS_OP_N,CUBLAS_OP_T,N,N,N,&t_alphA,L,N,L,N,&_beta,tmpM2,N);
	
	float *tmpFat,*FatATULA;
	cudaMalloc((void**)&tmpFat,Arows*Acols*sizeof(float));
	cudaMalloc((void**)&FatATULA,Acols*Acols*sizeof(float));
	
	//A'*(inv(U)*inv(L))
	cublasSgemm_v2(handle,CUBLAS_OP_N,CUBLAS_OP_N,Acols,Arows,Arows,&t_alphA,D_A,Acols,tmpM2,Arows,&_beta,tmpFat,Acols);

	//tmpFat*A
	cublasSgemm_v2(handle,CUBLAS_OP_N,CUBLAS_OP_T,Acols,Acols,Arows,&t_alphA,tmpFat,Acols,D_A,Acols,&_beta,FatATULA,Acols);
	
	//NOTE: assuming only 'fat' sub-matrix A, and FatATULA will be of size Acols x Acols (the larger dimension of A)

	//sparse stuff
	const int nnz=(Acols*Acols)*num_ROIs;
	float *csrValA;
	int *csrRowPtrA,*csrColIndA;
	cudaMalloc((void**)&csrValA,nnz*sizeof(float));
	cudaMalloc((void**)&csrColIndA,nnz*sizeof(int));
	cudaMalloc((void**)&csrRowPtrA,(BigCols+1)*sizeof(int));
	
	//rep mat FatATULA across diag of sparse CSR matrix (BigCols x BigCols)
	fill_Row_Ptr_helper(csrRowPtrA,Acols,BigCols);
	rep_mat_diag_to_CSR_helper(FatATULA,csrValA,csrColIndA,Acols,num_ROIs,BigCols);

	//now have blk diag matrix in sparse form
	cusparseHandle_t cusparseHandle = 0;
	cusparseStatus_t cusparseStatus = cusparseCreate(&cusparseHandle);
	if(cusparseStatus!=CUSPARSE_STATUS_SUCCESS)fprintf(stderr, "cusparseCreate returned error code %d !\n", cusparseStatus);
	cusparseMatDescr_t descr=0;
	cusparseStatus=cusparseCreateMatDescr(&descr);
	if(cusparseStatus!=CUSPARSE_STATUS_SUCCESS)fprintf(stderr, "cusparseCreateMat returned error code %d !\n", cusparseStatus);
	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

	dim3 Grid((BigCols+CPPTHREADS-1)/CPPTHREADS,num_lambdas,1);
	

	float history_r_norm[MAX_LAMBDAS],history_s_norm[MAX_LAMBDAS],history_eps_pri[MAX_LAMBDAS],history_eps_dual[MAX_LAMBDAS],xnorm[MAX_LAMBDAS],znorm[MAX_LAMBDAS];
	const float c0=sqrtf(float(BigCols))*abstol;

	int* num_iters=(int*)malloc(num_lambdas*sizeof(int));
	memset(num_iters,-1,num_lambdas*sizeof(int));//memset to -1 which will indicate that did not converge within MAX_ITER iterations

	int mask=0;

	for(int i=1;i<=max_iter;i++){

		//update q in parallel (y dimension)
		update_vector_q(D_Atb,D_z,D_u,tempvecC,_rho,BigCols,Grid,mask);//tempvecC will act as vector 'q'
		
		//ok gnarly part, use streams (num_lambdas) to do sparse matrix-vector multiply for x-update assuming 'fat' A matrix

		for(multi=0;multi<num_lambdas;multi++)if(!(mask&(1<<multi))){/*cusparseStatus=*/
			cusparseSetStream(cusparseHandle,streams[multi]);
			//if(cusparseStatus!=CUSPARSE_STATUS_SUCCESS)fprintf(stderr, "cusparse returned error code %d !\n", cusparseStatus);
			/*cusparseStatus=*/
			cusparseScsrmv_v2(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,BigCols,BigCols,nnz,&t_alphA,descr,
				csrValA,csrRowPtrA,csrColIndA,tempvecC+multi*BigCols,&_beta,D_xresult+multi*BigCols);
			//if(cusparseStatus!=CUSPARSE_STATUS_SUCCESS)fprintf(stderr, "cusparse returned error code %d !\n", cusparseStatus);
		}
		
		finish_all_x_fat(tempvecC,D_xresult,_rho,BigCols,Grid,mask);
		
		cudaMemcpy(tempvecC,D_z,numbytesVC,cudaMemcpyDeviceToDevice);//tempvecC is zold
		
		x_hat_update_helper(D_xresult,tempvecC,x_hat,BigCols,_alpha,Grid,mask);

		cudaMemset(norm_s,0,num_k_bytes);
		
		z_shrinkage_wrap(D_z,x_hat,D_u,norm_s,lam_arr,Psize,admm_blk_size,num_lambdas,_rho,BigCols,mask);

		gpu_lasso_u_update_wrap(D_u,x_hat,D_z,BigCols,Grid,mask);

		cudaMemset(norms_array,0,(num_lambdas*6)*sizeof(float));
		get_multi_norms(D_xresult,D_z,tempvecC,D_u,norms_array,_rho,BigCols,Grid,num_lambdas,mask);
		
		cudaMemcpy(xnorm,norms_array,num_lambdas*sizeof(float),_DTH);
		cudaMemcpy(znorm,(float*)&norms_array[num_lambdas],num_lambdas*sizeof(float),_DTH);
		cudaMemcpy(history_r_norm,(float*)&norms_array[num_lambdas*2],num_lambdas*sizeof(float),_DTH);
		cudaMemcpy(history_s_norm,(float*)&norms_array[num_lambdas*3],num_lambdas*sizeof(float),_DTH);
		cudaMemcpy(history_eps_dual,(float*)&norms_array[num_lambdas*4],num_lambdas*sizeof(float),_DTH);
		
		for(multi=0;multi<num_lambdas;multi++)if(!(mask&(1<<multi))){
			history_r_norm[multi]=sqrtf(history_r_norm[multi]);
			history_s_norm[multi]=sqrtf(history_s_norm[multi]);
			xnorm[multi]=sqrtf(xnorm[multi]);
			znorm[multi]=sqrtf(znorm[multi]);
			history_eps_pri[multi]=c0+reltol*max(xnorm[multi],znorm[multi]);
			history_eps_dual[multi]=c0+reltol*sqrtf(history_eps_dual[multi]);
			//printf("\nr_norm= %f, s_norm= %f, eps_pri= %f, eps_dual= %f",history_r_norm[multi],history_s_norm[multi],
				//history_eps_pri[multi],history_eps_dual[multi]);
			if(history_r_norm[multi]<history_eps_pri[multi] && history_s_norm[multi]<history_eps_dual[multi]){
				mask|=(1<<multi);
				num_iters[multi]=i;	
			}		
		}
		if(mask==((1<<num_lambdas)-1))break;
	}
	//create answer for Matlab and copy back vectors u and z
	plhs[0]=mxCreateNumericMatrix(BigCols,num_lambdas,mxSINGLE_CLASS,mxREAL);
	plhs[1]=mxCreateNumericMatrix(BigCols,num_lambdas,mxSINGLE_CLASS,mxREAL);
	plhs[2]=mxCreateNumericMatrix(num_lambdas,1,mxINT32_CLASS,mxREAL);

	float *u_result=(float *)mxGetPr(plhs[0]);
	float *z_result=(float *)mxGetPr(plhs[1]);
	int *num_iter_result=(int *)mxGetPr(plhs[2]);

	cudaMemcpy(u_result,D_u,numbytesVC,_DTH);//copy  u info back to host
	cudaMemcpy(z_result,D_z,numbytesVC,_DTH);//copy  z info back to host
	memcpy(num_iter_result,num_iters,num_lambdas*sizeof(int));

	cudaFree(D_A);
	cudaFree(L);
	cudaFree(tmpM2);
	cudaFree(tmpM3);
	cudaFree(tmpFat);
	cudaFree(FatATULA);
	cudaFree(D_b);
	cudaFree(D_xresult);
	cudaFree(D_Atb);
	cudaFree(D_u);
	cudaFree(D_z);
	cudaFree(tempvecC);
	cudaFree(norms_array);
	cudaFree(x_hat);
	cudaFree(norm_s);
	cudaFree(lam_arr);

	cudaFree(csrValA);
	cudaFree(csrRowPtrA);
	cudaFree(csrColIndA);
	
	checkError(cublasDestroy(handle), "cublasDestroy() error!\n");
	cusparseDestroyMatDescr(descr);
	cusparseDestroy(cusparseHandle);

	for(multi=0;multi<n_streams;multi++){
		cudaStreamDestroy(streams[multi]);
	}
	free(streams);
	free(num_iters);
}
