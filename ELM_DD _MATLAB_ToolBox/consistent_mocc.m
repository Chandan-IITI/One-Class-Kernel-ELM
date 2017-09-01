function [w1, para] = consistent_mocc(x, w, fracrej, range, nrbags, varargin)
%CONSISTENT_MOCC
%For One Class Classifiers (OCCs) having more than one hyperparameters.
%
%     W = CONSISTENT_NOCC(X,w,FRACREJ,RANGE,NRBAGS)
%
% Optimize the hyperparameters of method w. w should contain the
% (string) name of a one-class classifier. Using crossvalidation on
% dataset X (containing just target objects!), this classifier is
% trained using the target rejection rate FRACREJ and the values of
% the hyperparameter given in RANGE. For OCCs having K > 1 hyperparameters,
% RANGE in fact have K cells, where each cell contains the possible 
% range of different hyperparameter. The hyperparameters in each cell
% should be ordered such that the most simple classifier comes
% first. Every combination of hyperparameters is tried so that the most
% complex classifier without over-fitting the target data W is returned.
% The hyperparameters in the latter cell of RANGE have higher priority.
% Per default NRBAGS-crossvalidation is used.
%
% An example for kmeans_dd (One hyperparameter), where k is optimized:
%     W = consistent_aocc(x,'kmeans_dd',0.1, {[1:20]})
% An example for svdd (One hyperparameter), where sigma is optimized:
%     W = consistent_aocc(x,'svdd',0.1, scale_range(x))
%
%
% Finally, some classifiers require additional parameters, they
% should be given in P1,P2,... at the end.
%
%     W = CONSISTENT_OCC(X,w,FRACREJ,RANGE,NRBAGS,P1,P2,...)
%
% An example for elm_kernel_dd (Two hyperparameters with RBF_kernel),
% where C and sigma are optimized:
%     W = consistent_mocc(x,'aaelm_kernel_dd',0.1, {power(10,-10:10),
%          scale_range(x)}, 5, 'RBF_kernel')
% where 'power(10,-10:10)' is the range of C and 'scale_range(x)' is the 
% range of sigma, and sigma has higher priority than C.
% 
% An example for aaelm_kernel_dd (Polynomial Kernel):
%     W = consistent_aocc(x,'aaelm_kernel_dd',0.1, {power(10,-10:10),
%          [1:3], [1:10]}, 5, 'Poly_kernel')
%
% Default: NRBAGS=5
%
% See also: scale_range, dd_crossval
%
% Notes: This is a rewritten version of DD_Tools file consistent_occ.m 

% Just Go through the file demo_elm_dd_tuned.m for better understanding for
% how to use this for ELM based one-class classifiers

if nargin<6 || isempty(nrbags)
	nrbags = 5;
end

% Check some things:
if ~isa(w,'char')
	error('Expecting the name (string!) of the classifier');
end
if length(fracrej)>1
	error('Fracrej should be a scalar');
end

% Default target error bound 
sigma_thr = 2;
fracrej_thr=fracrej; %%% For all threshold in default

%%%% For Thr3 %%%%%
if length(nrbags)==3
   fracrej_thr=nrbags(3);
end

if length(nrbags)>1
   % target error bound could be set in nrbags
   sigma_thr = nrbags(2);
   nrbags = nrbags(1);
end

%%%% End for Thr3 %%%%%

% Setup the consistency threshold, say the two sigma bound:
nrx = size(x,1);
if fracrej ~= 0
    add_thr = sqrt(fracrej*(1-fracrej)*nrbags/nrx);
else
    add_thr = 0.005;
end
err_thr = sigma_thr*add_thr;


% AI!---------------^

% OCCS having K hyperparameters
K = length(range);

cur_pos = zeros(1,K);
para = zeros(1,K);
max_times = 1;

for i=1:K
    % Try from the most complex to simplest classifier.
    range{i} = fliplr(range{i});
    % The total number of parameters combination.
    max_times = max_times*length(range{i});
    % Hyperparameter index.
    cur_pos(i)=1;
end


for t=1:max_times+1
    
    % Get the current para combination.
    for j=1:K
        para(j) = range{j}(cur_pos(j));
    end
    
    I = nrbags;
    for i=1:nrbags
    % Compute the target error on the leave-out bags
        [xtr,xte,I] = dd_crossval(x,I);
        w1 = feval(w,xtr,fracrej_thr, para, varargin{:});
        res = dd_error(xte,w1);
        err(i) = res(1);
    end
    fracout = mean(err)-fracrej;
    % IF it doesn't overfit the targer data, we choose it and this is the
    % most complex classifier.
    if fracout < err_thr
        break;
    end
    
    % Update the index of hyperparameters.
    cur_pos(1) = cur_pos(1)+1;
    for j=1:K
        if cur_pos(j)>length(range{j})
            cur_pos(j) = 1;
            if j~= K
            cur_pos(j+1) = cur_pos(j+1)+1;
            end
        else
            break;
        end
    end
end

if t > max_times
% Umm... all the classifiers are inconsistent:
    error('The most simple classifier is still inconsistent!');
end

w1 = feval(w, x, fracrej_thr, para, varargin{:});

return