num_epoch = 20;
eta = [0.001 0.005 0.01 0.05 0.1 0.5];
file_prefix = 'cs545_hw1_app_pairs';

% Load training data
filename = 'mnist_train.csv';
trn_data = csvread(filename);

trn_target = trn_data(:, 1);    % Actual training digit value
trn_data = trn_data(:, 1:end)/255;  % Normalized grayscale values
trn_data(:, 1) = 1;         % Set bias input

% Load test data
filename = 'mnist_test.csv';
test_data = csvread(filename);

test_target = test_data(:, 1);    % Actual test digit value
test_data = test_data(:, 1:end)/255;  % Normalized grayscale values
test_data(:, 1) = 1;         % Set bias input

% Calulate size of training and test data sets
[num_trn_samp, num_inputs] = size(trn_data);
[num_test_samp, ~] = size(test_data);
for i = 1:length(eta)
    tic
    % Initialize weights with small random values
    w = (rand(45, num_inputs) - 0.5) / 10;
    
    epoch = 0;
    trn_a = nan(1, num_epoch+1);
    test_a = nan(1, num_epoch+1);
    x_a = 0:1:num_epoch;
    
    % Run initial test to examine the untrained perceptrons
    [~, trn_a(1)] = test(w, trn_data, trn_target);
    [~, test_a(1)] = test(w, test_data, test_target);
    fprintf('Training Accuracy = %.1f%% \nTest Accuracy = %.1f%% \nEpoch = %d\n\n',...
        trn_a(1), test_a(1), 0);
    
    for epoch = 1:num_epoch
        w = train(w, trn_data, trn_target, eta(i));
        [~, trn_a(epoch+1)] = test(w, trn_data, trn_target);
        [conf_mat, test_a(epoch+1)] = test(w, test_data, test_target);
        
        clf
        plot(x_a, trn_a);
        hold on
        grid on
        plot(x_a, test_a); title('Accuracy vs Epochs');
        hold off
        axis([0, num_epoch, 80, 100])
        
        fprintf('Training Accuracy = %.1f%% \nTest Accuracy = %.1f%% \nEpoch = %d\n\n',...
            trn_a(epoch+1), test_a(epoch+1), epoch);
        pause(0.01)
    end
    disp(conf_mat)
    
    eta_str = strrep(num2str(eta(i)), '.', 'p');
    save([file_prefix, '_eta', eta_str, '.mat'])
    toc
end
sound(audioread('Wubalubadubdub.wav'), 44100)
% imshow(reshape(data(k, 2:end), 28, 28)'*255)

function weights = train(weights, data, targets, eta)

[num_samp, ~] = size(data);
rand_ind = randperm(num_samp);
p = 0;
for a = 1:9
    b = a;
    while b < 10
        b = b + 1;
        p = p + 1;
        
        
        for k = 1:num_samp
            if targets(rand_ind(k)) == a-1 || targets(rand_ind(k)) == b-1
                output = dot(weights(p, :), data(rand_ind(k), :));
                if output < 0 % result is a
                    if targets(rand_ind(k)) ~= a-1
                        weights(p, :) = weights(p, :) + (eta .* data(rand_ind(k), :));
                    end
                else % result is b
                    if targets(rand_ind(k)) ~= b-1
                        weights(p, :) = weights(p, :) - (eta .* data(rand_ind(k), :));
                    end
                end
            end
        end
    end
end
end

function [confusion_matrix, accuracy] = test(weights, data, targets)

confusion_matrix = zeros(10);
[num_samp, ~] = size(data);
for k = 1:num_samp
    p = 0;
    output = zeros(1, 10);
    for a = 1:9
        b = a;
        while b < 10
            b = b + 1;
            p = p + 1;
            
            activate = dot(weights(p, :), data(k, :));
            if activate < 0
                output(a) = output(a) + 1;
            else
                output(b) = output(b) + 1;
            end
        end
    end
    
    m = max(output);
    max_ind = find(output == m);
    ind = datasample(max_ind, 1);
    confusion_matrix(targets(k)+1, ind) = confusion_matrix(targets(k)+1, ind) + 1;
end

accuracy = 100 * (trace(confusion_matrix) / num_samp);
end
