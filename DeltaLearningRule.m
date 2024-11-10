% Initialize parameters
C = 0.1;  % Learning rate
epochs = 100; % Number of iterations

% Input data (X) and Desired Output (D)
X = [10 2; 2 -5; 5 5]; % Inputs
D = [1 -1 -1; -1 1 -1; -1 -1 1]; % Desired outputs

% Add bias input to X
X = [X, -1*ones(size(X,1), 1)];

% Initialize weights matrix W
W = [1 -2 0; 0 -1 2; 1 3 -2];

% Training phase
for epoch = 1:epochs
    for i = 1:size(X,1)
        % Calculate net input (dot product of input and weights)
        V = X(i,:) * W;
        
        % Apply bipolar sigmoid activation function
        Y = (2 ./ (1 + exp(-V))) - 1;
        
        % Compute the error (difference between desired and actual output)
        E = D(i,:) - Y;
        
        % Calculate the derivative of the bipolar sigmoid function
        Y_derivative = 0.5 * (1 + Y) .* (1 - Y);
        
        % Update weights using Delta rule with the derivative of activation
        W = W + C * (X(i,:)' * (E .* Y_derivative));
    end
end

% Display final weights
disp('Final Weights:');
disp(W);

% Testing phase (Recall)
for i = 1:size(X,1)
    V = X(i,:) * W; % Calculate net input
    Y = (2 ./ (1 + exp(-V))) - 1; % Apply bipolar sigmoid activation function
    
    % Thresholding to get exact -1 or 1 outputs
    Y_thresholded = Y;
    Y_thresholded(Y >= 0) = 1;  % Set to 1 if Y is greater than or equal to 0
    Y_thresholded(Y < 0) = -1;   % Set to -1 if Y is less than 0
    
    fprintf('Input: [%d %d %d] -> Output: [%d %d %d]\n', X(i,1), X(i,2), X(i,3), Y_thresholded(1), Y_thresholded(2), Y_thresholded(3));
end