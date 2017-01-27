function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

Sigma = 0;
for i = 1:m
  hyp = sigmoid(theta' * X(i,:)');
  Sigma += -y(i) * log(hyp) - (1 - y(i)) * log(1 - hyp);
endfor
J = Sigma/m;
J += lambda/(2*m) * sum(theta(2:size(theta)).^2);

for j = 1:size(theta)
  Sigma = 0;
  for i = 1:m
    hyp = sigmoid(theta' * X(i,:)');
    Sigma += (hyp - y(i)) * X(i,j);
  endfor
  if j==1
    grad(j) = Sigma/m;
  else
    grad(j) = Sigma/m + lambda/m * theta(j);
  endif
endfor


% =============================================================

end
