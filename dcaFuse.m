function [Ax,Ay,Xs,Ys] = dcaFuse(X,Y,label)
% DCAFUSE calculates the Discriminant Correlation Analysis (DCA) for 
% feature-level fusion in multimodal systems.
% 
% 
% Inputs:
%       X       :   pxn matrix containing the first set of training feature vectors
%                   p:  dimensionality of the first feature set
%                   n:  number of training samples
% 
%       Y       :   qxn matrix containing the second set of training feature vectors
%                   q:  dimensionality of the second feature set
% 
%       label   :   1xn row vector of length n containing the class labels
%               
% Outputs:
%       Ax  :   Transformation matrix for the first data set (rxp)
%               r:  maximum dimensionality in the new subspace
%       Ay  :   Transformation matrix for the second data set (rxq)
%       Xs  :   First set of transformed feature vectors (rxn)
%       Xy  :   Second set of transformed feature vectors (rxn)
% 
% 
% Sample use:
% 
%   % Calculate the transformation matrices Ax and Ay and project the
%   % training data into the DCA subspace
%   >> [Ax, Ay, trainXdca, trainYdca] = dcaFuse(trainX, trainY, label);
% 
%   % Project the test data into the DCA subspace
%   >> testXdca = Ax * testX;
%   >> testYdca = Ay * testY;
% 
%   % Fuse the two transformed feature matrices with either concatenation or summation:
%   % Fusion by concatenation (Z1)
%   >> trainZ1 = [trainXdca ; trainYdca];
%   >> testZ1  = [testXdca ; testYdca];
% 
%   % Fusion by summation (Z2)
%   >> trainZ2 = [trainXdca + trainYdca];
%   >> testZ2  = [testXdca + testYdca];
% 
% 
% 
%   Details can be found in:
%   
%   M. Haghighat, M. Abdel-Mottaleb, W. Alhalabi, "Discriminant Correlation 
%   Analysis: Real-Time Feature Level Fusion for Multimodal Biometric 
%   Recognition," IEEE Transactions on Information Forensics and Security, 
%   vol. 11, no. 9, pp. 1984-1996, Sept. 2016. 
%   http://dx.doi.org/10.1109/TIFS.2016.2569061
% 
%   and
% 
%   M. Haghighat, M. Abdel-Mottaleb W. Alhalabi, "Discriminant Correlation 
%   Analysis for Feature Level Fusion with application to multimodal 
%   biometrics," IEEE International Conference on Acoustics, Speech and 
%   Signal Processing (ICASSP), 2016, pp. 1866-1870. 
%   http://dx.doi.org/10.1109/ICASSP.2016.7472000
% 
% 
% 
% (C)   Mohammad Haghighat, University of Miami
%       haghighat@ieee.org
%       PLEASE CITE THE ABOVE PAPERS IF YOU USE THIS CODE.


[p,n] = size(X);
if size(Y,2) ~= n
    error('X and Y must have the same number of columns (samples).');
elseif length(label) ~= n
    error('The length of the label must be equal to the number of samples.');
elseif n == 1
    error('X and Y must have more than one column (samples)');
end
q = size(Y,1);

% % Normalize features (this has to be done for both train and test data)
% X = (X - repmat(mean(X,2), 1, n))./repmat(std(X,0,2), 1, n);
% Y = (Y - repmat(mean(Y,2), 1, n))./repmat(std(Y,0,2), 1, n);

%% Compute mean vectors for each class and for all training data

classes = unique(label);
c = numel(classes);
cellX = cell(1,c);
cellY = cell(1,c);
nSample = zeros(1,c);
for i = 1:c
    index = find(label==classes(i));
    nSample(i) = length(index);
    cellX{1,i} = X(:,index);
    cellY{1,i} = Y(:,index);
end

meanX = mean(X,2);  % Mean of all training data in X
meanY = mean(Y,2);  % Mean of all training data in Y

classMeanX = zeros(p,c);
classMeanY = zeros(q,c);
for i = 1:c
    classMeanX(:,i) = mean(cellX{1,i},2);   % Mean of each class in X
    classMeanY(:,i) = mean(cellY{1,i},2);   % Mean of each class in Y
end

PhibX = zeros(p,c);
PhibY = zeros(q,c);
for i = 1:c
    PhibX(:,i) = sqrt(nSample(i)) * (classMeanX(:,i)-meanX);
    PhibY(:,i) = sqrt(nSample(i)) * (classMeanY(:,i)-meanY);
end

clear label index cellX cellY meanX meanY classMeanX classMeanY

%% Diagolalize the between-class scatter matrix (Sb) for X

artSbx = (PhibX') * (PhibX);   % Artificial Sbx (artSbx) is a (c x c) matrix
[eigVecs,eigVals] = eig(artSbx);
eigVals = abs(diag(eigVals));

% Ignore zero eigenvalues
maxEigVal = max(eigVals);
zeroEigIndx = find(eigVals/maxEigVal<1e-6);
eigVals(zeroEigIndx) = [];
eigVecs(:,zeroEigIndx) = [];

% Sort in descending order
[~,index] = sort(eigVals,'descend');
eigVals = eigVals(index);
eigVecs = eigVecs(:,index);

% Calculate the actual eigenvectors for the between-class scatter matrix (Sbx)
SbxEigVecs = (PhibX) * (eigVecs);

% Normalize to unit length to create orthonormal eigenvectors for Sbx:
cx = length(eigVals);   % Rank of Sbx
for i = 1:cx
    SbxEigVecs(:,i) = SbxEigVecs(:,i)/norm(SbxEigVecs(:,i));
end

% Unitize the between-class scatter matrix (Sbx) for X
SbxEigVals = diag(eigVals);                 % SbxEigVals is a (cx x cx) diagonal matrix
Wbx = (SbxEigVecs) * (SbxEigVals^(-1/2));	% Wbx is a (p x cx) matrix which unitizes Sbx

clear index eigVecs eigVals maxEigVal zeroEigIndx
clear PhibX artSbx SbxEigVecs SbxEigVals

%% Diagolalize the between-class scatter matrix (Sb) for Y

artSby = (PhibY') * (PhibY);	% Artificial Sby (artSby) is a (c x c) matrix
[eigVecs,eigVals] = eig(artSby);
eigVals = abs(diag(eigVals));

% Ignore zero eigenvalues
maxEigVal = max(eigVals);
zeroEigIndx = find(eigVals/maxEigVal<1e-6);
eigVals(zeroEigIndx) = [];
eigVecs(:,zeroEigIndx) = [];

% Sort in descending order
[~,index] = sort(eigVals,'descend');
eigVals = eigVals(index);
eigVecs = eigVecs(:,index);

% Calculate the actual eigenvectors for the between-class scatter matrix (Sby)
SbyEigVecs = (PhibY) * (eigVecs);

% Normalize to unit length to create orthonormal eigenvectors for Sby:
cy = length(eigVals);      % Rank of Sby
for i = 1:cy
    SbyEigVecs(:,i) = SbyEigVecs(:,i)/norm(SbyEigVecs(:,i));
end

% Unitize the between-class scatter matrix (Sby) for Y
SbyEigVals = diag(eigVals);                  % SbyEigVals is a (cy x cy) diagonal matrix
Wby = (SbyEigVecs) * (SbyEigVals^(-1/2));    % Wby is a (q x cy) matrix which unitizes Sby

clear index eigVecs eigVals maxEigVal zeroEigIndx
clear PhibY artSby SbyEigVecs SbyEigVals

%% Project data in a space, where the between-class scatter matrices are 
%  identity and the classes are separated

r = min(cx,cy);	% Maximum length of the desired feature vector

Wbx = Wbx(:,1:r);
Wby = Wby(:,1:r);

Xp = Wbx' * X;  % Transform X (pxn) to Xprime (rxn)
Yp = Wby' * Y;  % Transform Y (qxn) to Yprime (rxn)

%% Unitize the between-set covariance matrix (Sxy)
%  Note that Syx == Sxy'

Sxy = Xp * Yp';	% Between-set covariance matrix

[Wcx,S,Wcy] = svd(Sxy); % Singular Value Decomposition (SVD)

Wcx = Wcx * (S^(-1/2)); % Transformation matrix for Xp
Wcy = Wcy * (S^(-1/2)); % Transformation matrix for Yp

Xs = Wcx' * Xp;	% Transform Xprime to XStar
Ys = Wcy' * Yp;	% Transform Yprime to YStar

Ax = (Wcx') * (Wbx');	% Final transformation Matrix of size (rxp) for X
Ay = (Wcy') * (Wby');	% Final transformation Matrix of size (rxq) for Y

