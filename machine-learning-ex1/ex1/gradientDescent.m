function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
  
    delta = zeros(size(X,2),1); % initialize delta vector

    %for i=1:m,
      %hx = theta'*X'(:,i);
      %delta = delta + ((hx-y'(i))*X'(:,i));
    %endfor;
    hx = theta'*X';
    delta = sum(((hx-y').*X'),2);

    theta = theta - ((alpha/m)*delta);

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
