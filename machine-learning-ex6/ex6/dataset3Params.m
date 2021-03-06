function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C1 = [.01 .03 .1 .3 1 3 10 30];
sigma1 = [.01 .03 .1 .3 1 3 10 30];
m = 1;

C = 0;
sigma = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

for c = C1
	for s = sigma1
		model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s));
		predictions = svmPredict(model, Xval);
		m1 = mean(double(predictions ~= yval));
		if (m1 < m)
			m = m1;
			C = c;
			sigma = s;
		endif
	end
end

% =========================================================================

end
