function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');

%The below line is applicable only for logical arrays where number of classes > 2
%[dummy, p] = max(h2, [], 2);
idx = [1:m];
%disp([idx' h2]);

for i = 1:length(h2)
	p(i) = (h2(i) >= 0.5);
end
% =========================================================================

disp([idx' p]);
end
