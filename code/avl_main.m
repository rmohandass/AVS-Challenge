%% Initialization
clear ; close all; clc

fprintf('Loading and Visualizing Data ...\n\n\n')

fprintf('Loading training data ...\n')
avs_train_data = load('X_Train_AVS.txt');

fprintf('Loading cross-validation data ...\n')
avs_cv_data = load('X_cv_AVS.txt');

X = avs_train_data(:,1:9);
y = avs_train_data(:,10);

Xval = avs_cv_data(:,1:9);
yval = avs_cv_data(:,10);

%lets add 2nd degree features
x_sq = X.^2;
x_cb = X.^3;
X = [X x_sq x_cb];

xval_sq = Xval.^2;
xval_cb = Xval.^3;
Xval = [Xval xval_sq xval_cb];


[m,n] = size(X);
% Normalize input
[X mu sigma] = featureNormalize(X);



% Initializing neural network
fprintf('\nInitializing Neural Network Parameters ...\n')

input_layer_size  = 27;
hidden_layer_size = 4;
num_labels = 1;

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


% Train neural network
fprintf('\nTraining Neural Network ...\n')
options = optimset('MaxIter', 50);
lambda = 1;

% debug_J  = nnCostFunction(initial_nn_params, input_layer_size, ...
                          % hidden_layer_size, num_labels, X, y, lambda);
						  
% checkNNGradients(lambda);

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
								   

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
				 
%Predict Training set accuracy
fprintf('\nPredicting traning output ...\n')
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%Predict cross-valiation set accuracy
fprintf('\nPredicting cv output ...\n')
Xval_norm = zeros(size(Xval));
%disp(mu);
%disp(sigma);
for i = 1:n
	Xval_norm(:,i) = (Xval(:,i) - mu(i))./sigma(i);
end
%disp(Xval_norm);
pred = predict(Theta1, Theta2, Xval_norm);

fprintf('\nCross-validation Set Accuracy: %f\n', mean(double(pred == yval)) * 100);
