% Initialize variables
X = [10 2 -1; 2 -5 -1; 5 5 -1]; 
D = [1 -1 -1; -1 1 -1; -1 1 1];
W = [1 -2 0; 0 -1 2; 1 3 -1]; % Initialize weights

% Parameters
eta = 0.1; % Learning rate
max_epochs = 1000; % Maximum number of iterations

% Bipolar Binary Activation Function
activation_function = @(net) arrayfun(@(n) (n >= 0) * 2 - 1, net);

% Training loop
for epoch = 1:max_epochs
    total_error = 0;
    
    for i = 1:size(X, 1)  % Loop through each input
        % Calculate net input
        net = W * X(i, :)';
        
        % Apply bipolar binary activation function
        O = activation_function(net);
        
        % Calculate the error (difference between desired and actual output)
        error = D(i, :)' - O;
        
        % Update weights only if di is not equal to oi
        for j = 1:size(W, 1)
            if error(j) ~= 0
                delta_w = eta * error(j) * X(i, :);
                W(j, :) = W(j, :) + delta_w;
            end
        end
        
        % Accumulate the total error
        total_error = total_error + sum(error.^2);
    end
    
    % Break if the total error is 0 (perfect classification)
    if total_error == 0
        break;
    end
end

% Display final weights
disp('Final Weights:');
disp(W);

% Testing (Recall) Phase
disp('Testing (Recall) Phase:');
total_error = 0;
for i = 1:size(X, 1)
    net_test = W * X(i, :)'; % Calculate net input for each test example
    O_test = activation_function(net_test); % Apply bipolar binary activation function

    % Calculate the error for this input
    error = D(i, :)' - O_test;
    total_error = total_error + sum(error.^2); % Accumulate total error
    
    % Display the results
    fprintf('Input: [%d %d %d] -> Output: [%d %d %d] -> Desired Output: [%d %d %d]\n', ...
        X(i,1), X(i,2), X(i,3), O_test(1), O_test(2), O_test(3), D(i,1), D(i,2), D(i,3));
end

% Display total error after testing
fprintf('Total Error: %.4f\n', total_error);
