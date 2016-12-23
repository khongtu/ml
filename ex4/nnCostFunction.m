function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%% Theta1 = [25 x 401]
%% Theat2 = [10 x 26]
%% X = [5000 x 400] without bias

a1 = [ones(m,1) X];		%% [m x 401]
z2 = a1 * Theta1'; 		%% [m x 401] x [401 x 25] = [m x 25]
a2 = sigmoid(z2);		%% [m x 25]
a2 = [ones(m,1) a2];	%% [m x 26]	
z3 = a2 * Theta2';		%% [m x 26] x [26 x 10] = [m x 10]
a3 = sigmoid(z3);		%% [m x 10]
	
for i=1:m
	delta_3 = zeros(num_labels, 1);
	for k=1:num_labels
		h = a3(i, k);
		t = y(i)==k;
		J = J + (-t * log(h) - (1 - t) * log(1 - h));	
		delta_3(k) = h - t;
	end
	tmpA2 = a2(i,:);
	Theta2_grad = Theta2_grad + (delta_3 * tmpA2);	
	delta_2 = (Theta2' * delta_3)' .* (tmpA2 .* (1 - tmpA2));
	Theta1_grad = Theta1_grad + ((delta_2(:,[2:end]))' * a1(i,:));
end

tmpTheta1 = Theta1;
tmpTheta1(:,1) = 0;
tmpTheta2 = Theta2;
tmpTheta2(:,1) = 0;

reg = lambda * (sum(sum(tmpTheta1 .^ 2)) + sum(sum(tmpTheta2 .^ 2))) / (2 * m);
J = J / m + reg;

Theta1_grad = Theta1_grad ./ m + (lambda / m) * tmpTheta1;
Theta2_grad = Theta2_grad ./ m + (lambda / m) * tmpTheta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
