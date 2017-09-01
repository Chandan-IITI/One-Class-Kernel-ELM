function W = aaelm_thr3_kernel_dd( a, fracrej, para, Kernel_type)
%AAELM_THR3_KERNEL_DD Extreme Learning Machine Kernel Data Description
%
%   W = AAELM_THR3_KERNEL_DD(A, FRACREJ, PARA, KERNEL_TYPE)
%
% Optimizes a Extreme learning machine kernel data description for the
% dataset A. KERNEL_TYPE represents the type of adopted kernel.
% KERNEL_TPYE could be: 'RBF_kernel' for RBF kernel; 'Random_kernel' 
% for Random feature mapping using sigmoid activation function; 
% 'Lin_kernel' for Linear kernel; 'Poly_kernel' for Polynomial kernel.
% PARA contains all the hyperparameters. 
% PARA(1) is always the regularization coefficient C.
% When KERNEL_TYPE = 'RBF_kernel', PARA(2) is SIGMA;
% When KERNEL_TYPE = 'Random_kernel', PARA(2) is the number of hidden
% neurons L.
% FRACREJ gives the fraction of the target set which will be rejected.
%
% An example for RBF_kernel:
%     w = aaelm_thr3_kernel_dd(a, 0.1, [power(10,8), 1.41], 'RBF_kernel')   
% An exmple for Poly_kernel:
%     w = aaelm_thr3_kernel_dd(a, 0.1, [power(10,8), 2, 5], 'Poly_kernel')   
% An example for Random_kernel:
%     w = aaelm_thr3_kernel_dd(a, 0.1, [power(10,8), 50], 'Random_kernel')
%
% Default:  FRACREJ=0.1; KERNEL_TYPE='Random_kernel'; C=10^8; L=1000.
%
% See also: datasets, mapppings, dd_roc.
%

%%% Please cite following paper:
%
%{
@article{GAUTAM2017126,
title = "On the construction of extreme learning machine for online and offline one-class classification—An expanded toolbox",
journal = "Neurocomputing",
volume = "261",
number = "",
pages = "126 - 143",
year = "2017",
note = "Advances in Extreme Learning Machines (ELM 2015)",
issn = "0925-2312",
author = "Chandan Gautam and Aruna Tiwari and Qian Leng",
}
%}


% Do some checking
if nargin < 4 || isempty(Kernel_type), Kernel_type='Random_kernel'; end;
if nargin < 3 || isempty(para), para(1)=power(10,8); para(2)=1000; end;
if nargin < 2 || isempty(fracrej), fracrej=0.1; end;
if nargin < 1 || isempty(a)
    W = prmapping(mfilename,{fracrej, para, Kernel_type});
    W = setname(W,'AAELM_Thr3 Kernel Data Description');
    return ;
end

if ~ismapping(fracrej)
    
%============================ training ============================ 

	% Make sure a is a OC dataset:
	if ~isocset(a), error('one-class dataset expected'); end
    
    a = +target_class(a);
    [m,k] = size(a);
    
    % The regularization coefficient
    C = para(1);
    if strcmp(Kernel_type, 'Random_kernel')
        L = para(2);
        % Random Kernel parameters
        InputWeight = rand(k,L)*2-1;
        BiasofHiddenNeurons = rand(1,L);
        Kernel_para = [InputWeight; BiasofHiddenNeurons];
    else
        Kernel_para = para(2:end);
    end
    
    % Implicit normallization
    Omega_train = kernel_matrix(a, Kernel_type, Kernel_para);
    OutputWeight = ((Omega_train+speye(m)/C)\(a));
    
    % The distances of the training samples
    Y = (Omega_train*OutputWeight);
    
    for i=1:m
    count=0;
    for j=1:k        
        rerr(i,j)=abs((a(i,j)-Y(i,j))/(a(i,j)+Y(i,j)));
    if (rerr(i,j)>0.5) 
     count=count+1;
    end
    end
    out(i,1)=count;
    end
    
    W.training_a = a;
    W.threshold = round(k*fracrej);
    W.Kernel_type = Kernel_type;
    W.Kernel_para =  Kernel_para;
    W.OutputWeight = OutputWeight;
    W = prmapping(mfilename,'trained',W, char('target','outlier'),k,2);
    W = setname(W,'AAELM_Thr3 Data Description');
else
    
%============================ testing ============================
    W = getdata(fracrej);
    
    training_a = W.training_a;
    Kernel_type = W.Kernel_type;
    Kernel_para = W.Kernel_para;
    OutputWeight = W.OutputWeight;
    
    [m,k] = size(a);
    
    Omega_test = kernel_matrix(training_a, Kernel_type, Kernel_para, +a);
    Y = (Omega_test'*OutputWeight);
    test_data=+a;
    for i=1:m
        count=0;
        for j=1:k
            rerr(i,j)=abs((test_data(i,j)-Y(i,j))/(test_data(i,j)+Y(i,j)));
            if (rerr(i,j)>0.5)
                count=count+1;
            end
        end
        out(i,1)=count;
    end
    
    % The smaller distances are more likely to belong to the target class.
    new_out = -[out, repmat(W.threshold,m,1)];
    
    %new_out
    W = setdat(a,new_out,fracrej);
    
end

%%%%%%%%%%%%%%%%%% Kernel Matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function omega = kernel_matrix(Xtrain,kernel_type, kernel_pars,Xtest)

nb_data = size(Xtrain,1);

% Additional kernel type could be added here.
if strcmp(kernel_type, 'Random_kernel')
    [k,L] = size(kernel_pars);
    InputWeight = kernel_pars(1:k-1, 1:L);
    BiasofHiddenNeurons = kernel_pars(k, 1:L);
    BiasMatrix= repmat(BiasofHiddenNeurons, nb_data, 1);
    if nargin<4
        tempH = Xtrain*InputWeight;
        tempH = tempH+BiasMatrix;
        Htrain = 1 ./ (1 + exp(-tempH));
        omega = Htrain*Htrain';
    else
        tempH = Xtrain*InputWeight;
        tempH = tempH+BiasMatrix;
        Htrain = 1 ./ (1 + exp(-tempH));
        tempH = Xtest*InputWeight;
        BiasMatrix= repmat(BiasofHiddenNeurons, size(Xtest, 1), 1);
        tempH = tempH+BiasMatrix;
        Htest = 1 ./ (1 + exp(-tempH));
        omega = Htrain*Htest';
    end
    
elseif strcmp(kernel_type,'RBF_kernel')
    if nargin<4,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = exp(-omega./(kernel_pars(1)*kernel_pars(1)));
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xtest,1));
        XXh2 = sum(Xtest.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*Xtrain*Xtest';
        omega = exp(-omega./(kernel_pars(1)*kernel_pars(1)));
    end
    
elseif strcmp(kernel_type,'Lin_kernel')
    if nargin<4,
        omega = Xtrain*Xtrain';
    else
        omega = Xtrain*Xtest';
    end
    
elseif strcmp(kernel_type,'Poly_kernel')
    if nargin<4,
        omega = (Xtrain*Xtrain'+kernel_pars(1)).^kernel_pars(2);
    else
        omega = (Xtrain*Xtest'+kernel_pars(1)).^kernel_pars(2);
    end
    
elseif strcmp(kernel_type,'wav_kernel')
    if nargin<4,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        
        XXh1 = sum(Xtrain,2)*ones(1,nb_data);
        omega1 = XXh1-XXh1';
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
        
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xtest,1));
        XXh2 = sum(Xtest.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*(Xtrain*Xtest');
        
        XXh11 = sum(Xtrain,2)*ones(1,size(Xtest,1));
        XXh22 = sum(Xtest,2)*ones(1,nb_data);
        omega1 = XXh11-XXh22';
        
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
    end
end
