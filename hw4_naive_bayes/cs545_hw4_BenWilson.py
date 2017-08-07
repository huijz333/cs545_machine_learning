# CS 545 - Machine Learning
# Homework #4
# Author - Ben Wilson
# Date - February 23, 2017
#

import numpy as np
from sklearn import linear_model

def process_spam_data(filename='spambase.data'):
    '''Process spambase.data file - computes std and mean for each feature'''
    # Load data from file
    data = np.loadtxt(filename, delimiter=',', unpack=False)
    n_samp, n_feature = data.shape

    # Split the data so that the positive and negative cases are roughly equal
    proportion = 0.0
    while proportion <= 0.98 or proportion >= 1.02:
        # Shuffle data
        np.random.shuffle(data)

        # Slice data in half to create training and testing sets
        test_data = data[:int(n_samp/2)]
        train_data = data[int(n_samp/2):]

        # Calculate the proportion of spam in each set. Exit if balanced
        n_spam_r = train_data[:,-1].sum()
        n_spam_t = test_data[:,-1].sum()
        proportion = 2*n_spam_r / (n_spam_r + n_spam_t)

    # Take out the truth column and change zero values to -1
    true_r = train_data[:,-1]
    train_data = np.delete(train_data, -1, 1)

    true_t = test_data[:,-1]
    test_data = np.delete(test_data, -1, 1)
    return train_data, true_r, test_data, true_t

def gen_conditional_prob(train_data, true_r):
    n_samp, n_feat = train_data.shape

    # Sort the positive and negative samples into two different groups
    train_pos = train_data[true_r==1]
    train_neg = train_data[true_r==0]

    # Compute the mean of the positive and negative sample features
    train_mu = np.empty((2,n_feat))
    train_mu[0,:] = np.mean(train_neg, axis=0)
    train_mu[1,:] = np.mean(train_pos, axis=0)

    # Compute standard deviation of the positive and negative sample features
    train_std = np.empty((2,n_feat))
    train_std[0,:] = np.std(train_neg, axis=0)
    train_std[1,:] = np.std(train_pos, axis=0)
    return train_mu, train_std 

def predict_nb(test_data, mu, std, prior_pos, prior_neg):
    # To avoid divide by zero warnings/errors, remove columns with zero std
    b = np.where(std[1,:] == 0) 
    mu = np.delete(train_mu, b, 1)
    std = np.delete(train_std, b, 1)
    test_data = np.delete(test_x, b, 1)

    # Compute constant values to improve computation speed
    n_samp, n_feat = test_data.shape
    log_root_2pi = np.log(2*np.pi)/(-2)
    log_prior_neg = np.log(prior_neg)*(-2) - np.log(2*np.pi)
    log_prior_pos = np.log(prior_pos)*(-2) - np.log(2*np.pi)
    var = np.power(std, 2)

    # Determine prediction for each instance
    prediction = np.empty(n_samp)
    for i in range(n_samp):
        # Compute the conditional probabilities for the negative cases
        log_prob_n = np.power((test_data[i,:]-mu[0,:]),2) / var[0,:]
        log_prob_n += np.log(var[0,:])
        # Add log of prior probibility to the sum of the conditionals
        neg = log_prior_neg + np.sum(log_prob_n)

        # Compute the conditional probabilities for the positive cases
        log_prob_p = np.power((test_data[i,:]-mu[1,:]),2) / var[1,:]
        log_prob_p += np.log(var[1,:])
        # Add log of prior probibility to the sum of the conditionals
        pos = log_prior_pos + np.sum(log_prob_p)

        # Return smaller value -- This comes from a simplification due to the 
        # natural log function. See report for details.
        if pos <= neg:  prediction[i] = 1
        else:           prediction[i] = 0
    return prediction

def eval_prediction(predict, actual):
    '''Determine the accuracy, precision, recall, and fpr of the prediction'''
    # I'll admit this is sloppy and needs improvement, but this creates a 
    # confusion matrix and generates accuracy, precision and recall 
    conf_mat = np.zeros([2, 2])
    for i in range(len(predict)):
        if predict[i] > actual[i]:  # False Positive
            conf_mat[1, 0] += 1
        elif predict[i] < actual[i]:# False Negative
            conf_mat[0, 1] += 1
        elif predict[i] == 1:       # True Positive
            conf_mat[0, 0] += 1
        else:                       # True Negative
            conf_mat[1, 1] += 1

    accuracy = np.trace(conf_mat) / np.sum(conf_mat)
    precision = np.sum(conf_mat[0, 0]) / np.sum(conf_mat[:,0])
    recall = np.sum(conf_mat[0, 0]) / np.sum(conf_mat[0,:])
    return accuracy, precision, recall, conf_mat 

# ----------------------------------------------------------------------------
# ----------------------------- Experiment #1 --------------------------------
# ----------------------------------------------------------------------------

# Collect and split data into normalized training and testing sets
train_x, train_t, test_x, test_t = process_spam_data()

# Calculate the conditional probabilities
train_mu, train_std = gen_conditional_prob(train_x, train_t)

# Calculate prior
prior_pos = np.sum(train_t) / len(train_t)  # Probability of spam
prior_neg = 1 - prior_pos                   # Probability of NOT spam

# Compute Naive Bayes prediction
prediction = predict_nb(test_x, train_mu, train_std, prior_pos, prior_neg)

# Calculate accuracy, precision, and recall
accu, prec, recall, conf_mat = eval_prediction(prediction, test_t)
print("------------------------------")
print("-------- Experiment 1 --------")
print("------------------------------")
print("Accuracy = " + str(accu))
print("Precision = " + str(prec))
print("Recall = " + str(recall))
print("Confusion Matrix:")
print(conf_mat)
print('')

# ----------------------------------------------------------------------------
# ----------------------------- Experiment #2 --------------------------------
# ----------------------------------------------------------------------------

# # Collect and split data into normalized training and testing sets
# train_x, train_t, test_x, test_t = process_spam_data()

# Create Logistic Regression Classifier
logit = linear_model.LogisticRegression(penalty='l2', C=1)
logit.fit(train_x, train_t)
prediction = logit.predict(test_x)

# Calculate accuracy, precision, and recall
accu, prec, recall, conf_mat = eval_prediction(prediction, test_t)
print("------------------------------")
print("-------- Experiment 2 --------")
print("------------------------------")
print("Accuracy = " + str(accu))
print("Precision = " + str(prec))
print("Recall = " + str(recall))
print("Confusion Matrix:")
print(conf_mat)
