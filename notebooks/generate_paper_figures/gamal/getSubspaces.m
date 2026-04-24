%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subspaces
% Copyright (C) 2015 Gamaleldin F. Elsayed and John P. Cunningham
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gamaleldin F. Elsayed
%
% getSubspaces.m
%
%     Inputs:
%           DataStruct(1).A = contains sample data to identify its subspace;
%           DataStruct(1).dim = dimensionality of this subspace;
%     Optional Inputs:
%           DataStruct(1).bias = weight of this data relative to the
%           others.
%     Outputs:
%           QSubspaces(1).Q = orthonormal vectors of the identified
%           subspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [QSubspaces] = getSubspaces(DataStruct)

numSubspaces = length(DataStruct);
totalDim = 0;
bias = ones(numSubspaces,1);
if isfield(DataStruct, 'bias')
    bias = [DataStruct.bias];
end
bias = bias./sum(bias);
Q = [];

for j = 1:numSubspaces
    Cj = cov(DataStruct(j).A);
    d = DataStruct(j).dim;
    [U, S, V] = svd(Cj);
    S = diag(S);
    covM(:,:,j) = Cj;
    dim(j) = d;
    totalDim = totalDim+d;
    Q = [Q V(:,end-d+1:end)];
    normFact(j) = bias(j)./sum(S(1:d));
end

[Q1,~, Q2] = svd(Q, 'econ');
Q = Q1*Q2';

stp = 0.1;
maxIter = 50000;
f = [];
for i = 1:maxIter
    [f(i,:), gradQ] = subspacesObj(Q, covM, dim, normFact);
    G = gradQ - Q*gradQ'*Q; % project gradient

    Qprev = Q;
    
    Q = Q + stp*G;
    [Q1,~, Q2] = svd(Q, 'econ');
    Q = Q1*Q2';
    
    
    convStatus(i) = norm(Q-Qprev,'fro')^2./sum(dim);
    
    if i>100
        if ~(sum(convStatus(i-100:i)>=1e-10)>0)
            break;
        end
    end
end
[f(i,:), ~, QSubspaces] = subspacesObj(Q, covM, dim, normFact);


%
% rank the dimensions in each subspaces from top to low variance
for j = 1:numSubspaces
    Aj = DataStruct(j).A;
    Qj = QSubspaces(j).Q;
   
    Aj_proj = bsxfun(@minus, Aj, mean(Aj))*Qj;
    [~, ~, V] = svd(cov(Aj_proj));
    
    QSubspaces(j).Q = Qj * V; % rank top to low variance

end

end % end main function


% objective function to be minimized
function [f, gradQ, QSubspaces] = subspacesObj(Q, C, dim, normFact)
numSubspaces = length(dim);
gradQ = nan(size(Q));
dim = [0;dim(:)];
f = nan(numSubspaces,1);
QSubspaces = struct([]);
for j = 1:numSubspaces
    Qj = Q(:,sum(dim(1:j))+1:sum(dim(1:j))+dim(j+1));
    Cj = C(:,:,j);
    normFactj = normFact(j);
    gradQ(:,sum(dim(1:j))+1:sum(dim(1:j))+dim(j+1)) = Cj*Qj*normFactj;
    f(j) = normFactj*trace(Qj'*Cj*Qj);
    QSubspaces(j).Q = Qj;
end
end