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
%               derivatives of the cost w.r.t. each parameter in theta. Line 18

[J, grad] = costFunction(theta, X, y)	;	% get J and grad

% Update J
a = theta .* theta;
J = J + lambda / (2 * m) * sum(a(2:length(a)));

n = size(theta);

for i = 2 : n		% start from 2, not penalizing theta(1)
	grad(i) = grad(i) + lambda / m * theta(i);
end

% =============================================================

end
