
%%% Please cite following paper:
%
%@article{gautam2016one,
%  title={On The Construction of Extreme Learning Machine for Online and Offline One Class Classification - An Expanded Toolbox},
%  author={Gautam, C. and Tiwari, A. and Leng, Q.},
%  journal={Neurocomputing},
%  year={2016 (Accepted, Download the preprint version of accepted paper from: https://arxiv.org/abs/1701.04516 )},
%  publisher={Elsevier}
%}

%%%% These codes simply show how to use the ELM based one-class
%%%% classifiers with consistensy based optimal parameter or modele selection on Breast Cancer dataset%%% 
%%%% It will show that how to use consistent_mocc.m %%%
%%% Dataset is downloaded from http://homepage.tudelft.nl/n9d04/occ/index.html

%%% See also: consistent_mocc, elm_kernel_dd, elm_thr2_kernel_dd, aaelm_kernel_dd, aaelm_thr2_kernel_dd, aaelm_thr3_kernel_dd 

clear all;
tot_data=load('oc_505.mat');  %% Load the Dataset
data=tot_data.x.data;   %% Total Data Without label
labels=tot_data.x.nlab; %% Label of Whole Data
data_pr = prdataset(data,labels); %% Convert into 'prmapping' format 
A=oc_set(data_pr,'2'); %% Set label '2' as one-class dataset 

[B, outlier]  = target_class(A); %% Extract target and outlier sperately
thr = 0.1;    %% Fraction of Rejection 

%%%% Initializeng all Performance Evaluation Criteria %%%%
training_time = 0;
testing_time =0; 
sum_error_fn = 0 ;  
sum_error_fp = 0 ;
t_N_TP1=0;  t_N_TN1=0;  t_N_FP1=0;  t_N_FN1=0;
t_N_TP2=0;  t_N_TN2=0;  t_N_FP2=0;  t_N_FN2=0;
t_N_TP3=0;  t_N_TN3=0;  t_N_FP3=0;  t_N_FN3=0;
npt=0;

mean_1=zeros(1,tot_run);    mean_2=zeros(1,tot_run);    mean_acc1=zeros(1,tot_run);    mean_auc=zeros(1,tot_run);
%%%% End of Initialization %%%%
tot_run=20; %%% Just taking average over 20 runs
for k = 1:tot_run
    k
                  %%% Divide target data into two equal parts for training and testing %%%
        [train_target, test_target] =  gendat(B ,int32(size(B,1)*0.5));
                  %%% Combine the target and outlier data for testing and
                  %%% convert into prmapping format %%%
        test = prdataset([+test_target; +outlier], [ones(size(test_target, 1),1)*2; ones(size(outlier,1),1)]);
        test = oc_set(test, '2'); %%% Set label '2' as target data %%%

%%% size Calc and model selection%%%%
    if(k==1)
        %%% These will be required during Performance evaluation Later %%%
        [ff ll]=size(test_target);
        [kk rr]=size(outlier);
        [np1 np2]= size(train_target);
        [np3 np4]= size(test_target);
        npt= np1+np3;
 %%%% End of the collection of information for performance evaluation %%%%
     
     %%%%% Optimal Parameter Selection for Training during First run (i.e. k=1) only%%%%%%
                           
               %%%%% OCKELM_Thr1 (Kernel Feature Mapping) %%%%%%
%        range = {power(10,-8:8), scale_range(train_target)};
%        [w, para] = consistent_mocc(train_target, 'elm_kernel_dd', thr, range, [5 1],'RBF_kernel');

               %%%%% OCKELM_Thr2 (Kernel Feature Mapping) %%%%%%
%        range = {power(10,-8:8), scale_range(train_target)};
%        [w, para] = consistent_mocc(train_target, 'elm_thr2_kernel_dd', thr, range, [5 4],'RBF_kernel');

               %%%%% AAKELM_Thr1 (Kernel Feature Mapping) %%%%%%
       range = {power(10,-8:8), scale_range(train_target)};
       [w, para] = consistent_mocc(train_target, 'aaelm_kernel_dd', thr, range, [5 2],'RBF_kernel');

               %%%%% AAKELM_Thr2 (Kernel Feature Mapping) %%%%%%
%        range = {power(10,-8:8), scale_range(train_target)};
%        [w, para] = consistent_mocc(train_target, 'aaelm_thr2_kernel_dd', thr, range, [5 4],'RBF_kernel');

               %%%%% AAKELM_Thr3 (Kernel Feature Mapping) %%%%%%
%        range = {power(10,-8:8), scale_range(train_target)};
%        [w, para] = consistent_mocc(train_target, 'aaelm_thr3_kernel_dd', thr, range, [5 4 0.1],'RBF_kernel');
    
                     %%%%% OCELM_Thr1 (Random Feature Mapping) %%%%%%
%         range = {power(10,-8:8), 1:400};           
%         [w, para] = consistent_mocc(train_target, 'elm_kernel_dd', thr, range, [5 10],'Random_kernel');

                     %%%%% OCELM_Thr2 (Random Feature Mapping) %%%%%%
%         range = {power(10,-8:8), 1:400};           
%         [w, para] = consistent_mocc(train_target, 'elm_thr2_kernel_dd', thr, range, [5 8],'Random_kernel');

%                     %%%%% AAELM_Thr1 (Random Feature Mapping) %%%%%%
%         range = {power(10,-8:8), 1:400};           
%         [w, para] = consistent_mocc(train_target, 'aaelm_kernel_dd', thr, range, [5 1],'Random_kernel');

                      %%%%% AAELM_Thr2 (Random Feature Mapping) %%%%%%
%         range = {power(10,-8:8), 1:400};           
%         [w, para] = consistent_mocc(train_target, 'aaelm_thr2_kernel_dd', thr, range, [5 6],'Random_kernel');

                    %%%%% AAELM_Thr3 (Random Feature Mapping) %%%%%%
%         range = {power(10,-8:8), 1:400};           
%         [w, para] = consistent_mocc(train_target, 'aaelm_thr3_kernel_dd', thr, range, [5 1 0.1],'Random_kernel');

    end
%%%% End of size Calc and model selection%%%%%
    tic;
    
    %%%%%%%%% Training for Kernel Feature Mapping Based One-class classifiers  %%%%%%%%%
%%%%%% Select the same classifier as above selected for optimal parameter selection during first run (i.e. k-1)%%%%
    
%    w = elm_kernel_dd(train_target, thr, para, 'RBF_kernel');
%    w = elm_thr2_kernel_dd(train_target, thr, para, 'RBF_kernel');
   w = aaelm_kernel_dd(train_target, thr, para, 'RBF_kernel');
%    w = aaelm_thr2_kernel_dd(train_target, thr, para, 'RBF_kernel');
%    w = aaelm_thr3_kernel_dd(train_target, thr, para, 'RBF_kernel');

  %%%%%%%%%%% End of Kernel Feature Mapping based One-class classifiers %%%%%%%%%%%%%%%%%

    %%%%%%%%% Training for Random Feature Mapping Based One-class classifiers  %%%%%%%%%
    
%    w = elm_kernel_dd(train_target, thr, para, 'Random_kernel');
%    w = elm_thr2_kernel_dd(train_target, thr, para, 'Random_kernel');
%    w = aaelm_kernel_dd(train_target, thr, para, 'Random_kernel');
%    w = aaelm_thr2_kernel_dd(train_target, thr, para, 'Random_kernel');
%    w = aaelm_thr3_kernel_dd(train_target, thr, para, 'Random_kernel');

   %%%%%%%%%%% End of Random Feature Mapping based One-class classifiers %%%%%%%%%%%%%%%%%

      %%%% Performance Calculation %%%%%%
        t= toc;
        training_time = training_time + t;  %%% Training Time Calculation
        tic;
        Z = test*w;  %%%%% Testing on the 'test' data
        t = toc;
        testing_time = testing_time + t; % %%% Testing Time Calculation
        
        %%%%% Performance Evaluation %%%%
        [err1,F] = dd_error(Z);
        precision(k) = F(1);
        recall(k) = F(2);
        f1(k) = (2*F(1)*F(2))/(F(1)+F(2));
        
     N_FN1=err1(1)*ff;     %Number of false negative
     N_TP1=ff-N_FN1 ;      %Number of true positive
    
    t_N_FN1=t_N_FN1+N_FN1; %total number of false negative
    t_N_TP1=t_N_TP1+N_TP1; %total number of true positive
    mean_tp1(k)=1-err1(1);  %true positive rate or sensitivity

    N_FP1=err1(2)*kk;     %Number of false positive
    N_TN1=kk-N_FP1;        %Number of true negative
    t_N_FP1=t_N_FP1+N_FP1; %total number of false positive
    t_N_TN1=t_N_TN1+N_TN1; %total number of true negative
    mean_tn1(k)=1-err1(2); %true negative rate or specificity
    mean_auc(k)=0.5*(mean_tp1(k)+ mean_tn1(k)); %%AUC for each iteration saved in a vector for average calculation
    ACC1=(N_TP1+N_TN1)/(N_FN1+N_FP1+N_TP1+N_TN1);  %accuracy for each iteration
    mean_acc1(k)=ACC1; %%accuracy for each iteration saved in a vector for average calculation
end

%%%%%% caluclation of Average of the performances over 'tot_run', here,'tot_run'=20%%%%%%
result.f1 = mean(f1)*100;  %%% Average of F1
result.precision = mean(precision)*100; %%% Average of Precision
result.recall = mean(recall)*100; %%% Average of Recall
result.m_tp1=mean(mean_tp1)*100; %%% True Positive Rate or sensitivity
result.m_tn1=mean(mean_tn1)*100; %%% True Negative Rate or specificity
result.m_auc=mean(mean_auc)*100; %%% Area under curve
result.m_acc1=mean(mean_acc1); %%% Average Accuracy same as t_ACC1
t_ACC1=(t_N_TP1+t_N_TN1)/(t_N_FN1+t_N_FP1+t_N_TP1+t_N_TN1)*100;  %Average Accuracy 
             %%%%% Matthew'scorrelation coefficient (MCC) %%%%
result.MCC=(((t_N_TP1*t_N_TN1)- (t_N_FP1*t_N_FN1))/sqrt((t_N_TP1+t_N_FP1)*(t_N_TP1+t_N_FN1)*(t_N_TN1+t_N_FP1)*(t_N_TN1+t_N_FN1)))*100; 

%%%%% Standard Deviation over 'tot_run', here,'tot_run'=20%%%%%%
result.std_f1 = std(f1);  %%% Standard Deviation of F1
result.s_tp1=std(mean_tp1); %%% Standard Deviation of Sensitivity
result.s_tn1=std(mean_tn1); %%% Standard Deviation of Specificity
result.s_acc1=std(mean_acc1); %%% Standard Deviation of Accuracy
result.s_auc=std(mean_auc); %%% Standard Deviation of AUC
 
%%%% Training and Testing Time %%%%
result.training_time = training_time;
result.testing_time = testing_time;

result
