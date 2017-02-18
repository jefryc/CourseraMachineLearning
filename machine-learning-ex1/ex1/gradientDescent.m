function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n = length(theta);
t = zeros(n, 1);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %		line 18
	for i = 1 : n
		A = X * theta;
		B = A - y;
		C = B .* X(:, i);
		x = alpha / m * sum(C);
		t(i) = theta(i) - x;
	end;
	theta = t;   % update theta
    % ===========================================================

    % Save the cost J in every iteration 
    J_history(iter) = computeCost(X, y, theta);
		
end

end