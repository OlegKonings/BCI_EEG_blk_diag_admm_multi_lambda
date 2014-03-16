BCI_EEG_blk_diag_admm_multi_lambda
==================================

CUDA ADMM UCSD EEG


will take in 12 inputs and return 3 outputs.


	inputs are (in order)
	
	
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



Assumptions:

	1) single precision
	2) matrices in standard dense format must be passed into function in transpose state
	3) the sub-matrix is passed in and will be replicated (in sparse format) num_ROIs times.
	4) the sub-matrix is 'fat' (m>n)

First test the using the project's ' BCI_blk_diag_admm_multi_lambda.m' file. That gives a clear example of how to call, and how it is implemented in MATLAB (not optimal, but somewhat optimized).
