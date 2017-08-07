% Author: Ben Wilson
% Date: Jan. 23, 2017
% Course: CS 545
% Assignment: Homework #1


% The following MATLAB script trains ten perceptrons to determine the value 
% of hand written digits.
num_epoch = 50;
eta = [0.001, 0.01, 0.1];
file_prefix = 'cs545_hw1'; % used to generate save file names

% Load training data
filename = 'mnist_train.csv';
trn_data = csvread(filename);

trn_target = trn_data(:, 1);        % Actual training digit value
trn_data = trn_data(:, 1:end)/255;  % Normalized grayscale values
trn_data(:, 1) = 1;                 % Set bias input

% Load test data
filename = 'mnist_test.csv';
test_data = csvread(filename);

test_target = test_data(:, 1);          % Actual test digit value
test_data = test_data(:, 1:end)/255;    % Normalized grayscale values
test_data(:, 1) = 1;                    % Set bias input

% Calulate size of training and test data sets
[num_trn_samp, num_inputs] = size(trn_data);
[num_test_samp, ~] = size(test_data);

% Run the training/testing algorithm for each eta value
for i = 1:length(eta)
    tic
    % Initialize weights with small random values
    w = (rand(10, num_inputs) - 0.5) / 10;
    
    % Initialize arrays for plotting results
    trn_a = nan(1, num_epoch+1);
    test_a = nan(1, num_epoch+1);
    x_a = 0:1:num_epoch;
    
    % Run initial test to examine the untrained perceptrons
    [~, trn_a(1)] = test(w, trn_data, trn_target);
    [~, test_a(1)] = test(w, test_data, test_target);
    fprintf('Training Accuracy = %.1f%% \nTest Accuracy = %.1f%% \nEpoch = %d\n\n',...
        trn_a(1), test_a(1), 0);
    
    for epoch = 1:num_epoch
        % Train perceptrons
        w = train(w, trn_data, trn_target, eta(i));
        
        % Test training data
        [~, trn_a(epoch+1)] = test(w, trn_data, trn_target);
        
        % Test testing data
        [conf_mat, test_a(epoch+1)] = test(w, test_data, test_target);
        
        % Plot the results
        clf
        subplot(2, 1, 1); plot(x_a, trn_a);
        hold on
        grid on
        subplot(2, 1, 1); plot(x_a, test_a); title('Accuracy vs Epochs');
        hold off
        subplot(2, 1, 2); imshow(gen_grayscale_img(w), [0 255]); title('Weights');
        
        % Print accuracy and epoch number to screen
        fprintf('Training Accuracy = %.1f%% \nTest Accuracy = %.1f%% \nEpoch = %d\n\n',...
            trn_a(epoch+1), test_a(epoch+1), epoch);
        pause(0.01) % The plot won't update unless there is a pause
    end
    
    % 
    disp(conf_mat)
    
    eta_str = strrep(num2str(eta(i)), '.', 'p');
    save([file_prefix, '_eta', eta_str, '.mat'])
    toc
end
% % Play audio to signal end of test
% sound(audioread('Wubalubadubdub.wav'), 44100)




function weights = train(weights, data, targets, eta)
% % %%%%%%%%%% Original Algorithm %%%%%%%%%%
% % [num_samp, num_inputs] = size(data);
% % output = zeros(1, 10);
% % for k = 1:num_samp
% %     for dig = 1:10 % Test each of the ten neurons
% %         output(dig) = dot(weights(dig, :), data(k, :));
% %     end
% %     [~, ind] = max(output);
% %     if targets(k) ~= ind-1
% %         % Update weights for corresponding output
% %         for i = 1:num_inputs
% %             delta = (eta * data(k, i));
% %             weights(ind, i) = weights(ind, i) - delta;
% %             weights(targets(k)+1, i) = weights(targets(k)+1, i) + delta;
% %         end
% %     end
% % end

% %%%%%%%%%% New Algorithm %%%%%%%%%%
[num_samp, ~] = size(data);
for k = 1:num_samp
    for dig = 1:10 % Test each of the ten neurons
        output = dot(weights(dig, :), data(k, :));
        if output > 0  % The output is predicted to be (dig-1)
            if targets(k) ~= dig-1   % Incorrect positive prediction
                
                % Update weights (turn down input weights)
                weights(dig, :) = weights(dig, :) - (eta .* data(k, :));
            end
        else        % The output is predicted as NOT (dig-1)
            if targets(k) == dig-1   % Incorrect negative prediction
                % Update weights (turn up input weights)
                weights(dig, :) = weights(dig, :) + (eta .* data(k, :));
            end
        end
    end
end

% % %%%%%%%%%% Random Permutaion Algorithm %%%%%%%%%%
% % [num_samp, num_inputs] = size(data);
% % rand_ind = randperm(num_samp);
% % for k = 1:num_samp
% %     for dig = 1:10 % Test each of the ten neurons
% %         output = dot(weights(dig, :), data(rand_ind(k), :));
% %         if output > 0  % The output is predicted to be (dig-1)
% %             if targets(rand_ind(k)) ~= dig-1   % Incorrect positive prediction
% % 
% %                 Update weights (turn down input weights)
% %                 for i = 1:num_inputs
% %                     weights(dig, i) = weights(dig, i) - (eta * data(rand_ind(k), i));
% %                 end
% %             end
% % 
% %         else        % The output is predicted as NOT (dig-1)
% %             if targets(rand_ind(k)) == dig-1   % Incorrect negative prediction
% % 
% %                 Update weights (turn up input weights)
% %                 for i = 1:num_inputs
% %                     weights(dig, i) = weights(dig, i) + (eta * data(rand_ind(k), i));
% %                 end
% %             end
% %         end
% %     end
% % end
end


function [confusion_matrix, accuracy] = test(weights, data, targets)
% Initialize a 10x10 confustion matrix with all zeros.
confusion_matrix = zeros(10);
[num_samp, ~] = size(data);
output = zeros(1, 10);
for k = 1:num_samp
    for dig = 1:10 % Test each of the ten neurons
        output(dig) = dot(weights(dig, :), data(k, :));
    end
    % Use the target and predicted values to increment the confusion matrix
    [~, ind] = max(output);
    confusion_matrix(targets(k)+1, ind) = confusion_matrix(targets(k)+1, ind) + 1;
end
accuracy = 100 * (trace(confusion_matrix) / num_samp);
end


function a = gen_grayscale_img(dig_weight)
% Weights are reshaped into 28x28 matrices and placed in a 2x5 arrangement
% to be displayed along side the accuracy plot
ar = resize_grayscale(dig_weight(1, 2:end));
zero = reshape(ar, 28, 28)';

ar = resize_grayscale(dig_weight(2, 2:end));
one = reshape(ar, 28, 28)';

ar = resize_grayscale(dig_weight(3, 2:end));
two = reshape(ar, 28, 28)';

ar = resize_grayscale(dig_weight(4, 2:end));
three= reshape(ar, 28, 28)';

ar = resize_grayscale(dig_weight(5, 2:end));
four = reshape(ar, 28, 28)';

ar = resize_grayscale(dig_weight(6, 2:end));
five = reshape(ar, 28, 28)';

ar = resize_grayscale(dig_weight(7, 2:end));
six = reshape(ar, 28, 28)';

ar = resize_grayscale(dig_weight(8, 2:end));
seven = reshape(ar, 28, 28)';

ar = resize_grayscale(dig_weight(9, 2:end));
eight = reshape(ar, 28, 28)';

ar = resize_grayscale(dig_weight(10, 2:end));
nine = reshape(ar, 28, 28)';


a1 = horzcat(zero, one, two, three, four);
a2 = horzcat(five, six, seven, eight, nine);
a = vertcat(a1, a2);
end


function a = resize_grayscale(a)
% Renormalizes array to values between 0-255
a = a - min(a);
a = a / max(a);
a = a * 255;
end
