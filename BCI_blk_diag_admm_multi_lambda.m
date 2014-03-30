S=load('admm_data_1');
%parameters in order

A=single(full(S.A));
b=single(S.y);
blen=int32(length(b));
partition=int32(S.blks);
partition = partition(:);
 
rho=single(S.rho);
alpha=single(S.alpha);
lambda=single(S.lambda);
MAX_ITER = int32(100);
ABSTOL   = single(1e-4);
RELTOL   = single(1e-2);
num_ROIs=int32(14);
num_samples=127;
[bigM,bigN]=size(A);
m=int32(S.designMatrixBlockSize(1));
n=int32(S.designMatrixBlockSize(2));
fprintf('m= %d, n= %d\n',m,n);
 
if (sum(partition) ~= bigN)
    error('invalid partition');
end
 
cum_part = int32(cumsum(double(partition)));
% optimized group lasso for block diag
subMatrixA=A((1:m),(1:n));

%lambda_array=single([.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0,1.1,.05,1.3,1.4,1.7,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,3.0]');
lambda_array=single([.1,.2,.3,.4]');
%lambda_array=single([.1,.125,.15,.2125,.25,.275,.28125,.3,.35,.375]');
%lambda_array=single(.2);
num_lambdas=int32(length(lambda_array));
fprintf('num lambdas=%d\n',num_lambdas);


x=single(zeros(bigN,num_lambdas));
z=single(zeros(bigN,num_lambdas));
u=single(zeros(bigN,num_lambdas));

t=tic();

atb_sub=single(zeros(bigN,1));
for i=1:num_ROIs
    atb_sub((((i-1)*n)+1:i*n),1)=subMatrixA'*b((((i-1)*m)+1:i*m),1);
end

[L,U]=factor(subMatrixA,rho);
IUL=inv(U)*inv(L);% sub-matrix for use in parallel in iteer loop
fatT=subMatrixA'*IUL*subMatrixA;

has_converged = false(num_lambdas,1);
num_iterations_matlab=int32(zeros(num_lambdas,1));
num_iterations_matlab(:,1)=int32(-1);
for k = 1:MAX_ITER
    for kk=1:num_lambdas
        if has_converged(kk,1)
            continue;
        end
        q=atb_sub+rho*(z(:,kk) - u(:,kk));
        for j=1:num_ROIs
            x((((j-1)*n)+1:j*n),kk)=(q((((j-1)*n)+1:j*n),1))/rho-(fatT*q((((j-1)*n)+1:j*n),1))/rho^2; 
        end;
        zold = z(:,kk);
        start_ind = 1;
        x_hat = alpha*x(:,kk) + (1-alpha)*zold;
        for i = 1:length(partition),
            sel = start_ind:cum_part(i);
            z(sel,kk) = shrinkage(x_hat(sel) + u(sel,kk), lambda_array(kk)/rho);
            start_ind = cum_part(i) + 1;
        end
        u(:,kk) = u(:,kk) + (x_hat - z(:,kk));
        
        history.r_norm(k,kk)  = norm(x(:,kk) - z(:,kk));
        history.s_norm(k,kk)  = norm(-rho*(z(:,kk) - zold));
        
        history.eps_pri(k,kk) = sqrt(single(bigN))*ABSTOL + RELTOL*max(norm(x(:,kk)), norm(z(:,kk) ));
        history.eps_dual(k,kk)= sqrt(single(bigN))*ABSTOL + RELTOL*norm(rho*u(:,kk));
        
          %fprintf('r_norm=%f,s_norm=%f,eps_pri=%f,eps_dual=%f\n',history.r_norm(k,kk),history.s_norm(k,kk), history.eps_pri(k,kk),history.eps_dual(k,kk));
        if (history.r_norm(k,kk) <history.eps_pri(k,kk) && ...
                history.s_norm(k,kk) <history.eps_dual(k,kk))
            has_converged(kk,1)=true;
            num_iterations_matlab(kk,1)=k;
        end
    end;
    if all(has_converged)
            break;
    end
   % fprintf('\n');
    
end;

matlab_time=toc(t);
fprintf('matlab multi-lambda admm blk_diag time=%f\n',matlab_time);

uu=single(zeros(bigN,num_lambdas));
zz=single(zeros(bigN,num_lambdas));
t=tic();

[gpu_u,gpu_z,num_iters_gpu]=EEG_blk_diag_multi_all_shapes(subMatrixA',b,partition,uu,zz,rho,alpha,MAX_ITER,ABSTOL,RELTOL,lambda_array,num_ROIs);

gpu_time=toc(t);
fprintf(' gpu time= %f \n',gpu_time);

disp('____________Error comparison of MATLAB vs. CUDAmex_______________');


for i=1:num_lambdas
    fprintf('lambda value=%f, Num iters gpu= %d, Num iters matlab= %d \n',lambda_array(i),num_iters_gpu(i,1),num_iterations_matlab(i,1));
    fprintf('norm u dif= %f ,',norm(u(:,i))-norm(gpu_u(:,i)));
    fprintf('norm z dif= %f\n\n',norm(z(:,i))-norm(gpu_z(:,i)));
end
%exit;

