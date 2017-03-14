
%%% Please cite the following paper:
%
%{
@article{Gautam2017,
title = "On the construction of extreme learning machine for online and offline one-class classification—An expanded toolbox ",
journal = "Neurocomputing ",
volume = "",
number = "",
pages = " - ",
year = "2017",
issn = "0925-2312",
doi = "http://dx.doi.org/10.1016/j.neucom.2016.04.070",
author = "Chandan Gautam and Aruna Tiwari and Qian Leng",
}
%}

%%%% These codes simply show how to use the ELM based one-class
%%%% classifiers without optimal parameter selection on Breast Cancer dataset%%%
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

                  %%% Divide target data into two equal parts for training and testing %%%
        [train_target, test_target] =  gendat(B ,int32(size(B,1)*0.5));
                  %%% Combine the target and outlier data for testing and
                  %%% convert into prmapping format %%%
        test = prdataset([+test_target; +outlier], [ones(size(test_target, 1),1)*2; ones(size(outlier,1),1)]);
        test = oc_set(test, '2'); %%% Set label '2' as target data %%%

        %%% These will be required during Performance evaluation Later %%%
        [ff ll]=size(test_target);
        [kk rr]=size(outlier);
        [np1 np2]= size(train_target);
        [np3 np4]= size(test_target);
        npt= np1+np3;
 %%%% End of the collection of information for performance evaluation %%%%

       %%%%%%%%% Training for Kernel Feature Mapping Based One-class classifiers  %%%%%%%%%
    tic;
%    w = elm_kernel_dd(train_target, thr, [10^-4 1.4], 'RBF_kernel');
   w = aaelm_kernel_dd(train_target, thr, [10^-4 1], 'RBF_kernel');
%     w = aaelm_thr3_kernel_dd(train_target, thr, [10^8 1.9193], 'RBF_kernel');

  %%%%%%%%%%% End of Kernel Feature Mapping based One-class classifiers %%%%%%%%%%%%%%%%%

      %%%% Performance Calculation %%%%%%
        training_time = 0;    testing_time = 0;
        t= toc;
        training_time = training_time + t;  %%% Training Time Calculation
        tic;
        Z = test*w;  %%%%% Testing on the 'test' data
        t = toc;
        testing_time = testing_time + t; % %%% Testing Time Calculation
        
        %%%%% Performance Evaluation %%%%
        [err1,F] = dd_error(Z);
        precision = F(1);
        recall = F(2);
        f1 = (2*F(1)*F(2))/(F(1)+F(2));
        
     N_FN1=err1(1)*ff;     %Number of false negative
     N_TP1=ff-N_FN1 ;      %Number of true positive
    tp1=1-err1(1);  %true positive rate or sensitivity
    N_FP1=err1(2)*kk;     %Number of false positive
    N_TN1=kk-N_FP1;       %Number of true negative
    tn1=1-err1(2); %true negative rate or specificity
    auc=0.5*(tp1 + tn1); %%AUC calculation
    acc1=(N_TP1+N_TN1)/(N_FN1+N_FP1+N_TP1+N_TN1);  %Accuracy calculation 

%%%%%% caluclation of Average of the performances over 'tot_run', here,'tot_run'=20%%%%%%
result.f1 = f1*100;  %%% Average of F1
result.precision = precision*100; %%% Average of Precision
result.recall = recall*100; %%% Average of Recall
result.tp1=tp1*100; %%% True Positive Rate or sensitivity
result.tn1=tn1*100; %%% True Negative Rate or specificity
result.auc=auc*100; %%% Area under curve
result.acc1=acc1; %%% Average Accuracy same as t_ACC1

             %%%%% Matthew'scorrelation coefficient (MCC) %%%%
result.MCC=(((N_TP1*N_TN1)- (N_FP1*N_FN1))/sqrt((N_TP1+N_FP1)*(N_TP1+N_FN1)*(N_TN1+N_FP1)*(N_TN1+N_FN1)))*100; 

%%%% Training and Testing Time %%%%
result.training_time = training_time;
result.testing_time = testing_time;

result
