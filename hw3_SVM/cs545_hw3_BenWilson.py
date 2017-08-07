# CS 545 - Machine Learning
# Homework #3
# Author - Ben Wilson
# Date - February 14, 2017
#
# The following program processes the spambase data set, trains an SVM, 
# generates a ROC curve, and tries various methods of selecting features
# for training.

import matplotlib.pyplot as plt
import numpy as np
from random import shuffle, sample
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn import svm

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
    true_r[true_r==0] = -1
    train_data = np.delete(train_data, -1, 1)

    true_t = test_data[:,-1]
    true_t[true_t==0] = -1
    test_data = np.delete(test_data, -1, 1)

    # Calculate the mean and standard deviation of the training data
    train_mean = np.mean(train_data, axis=0)
    train_std = np.std(train_data, axis=0)

    # # Use scikitlearn to scale the data
    # train_data = scale(train_data)

    # Subtract training mean from and divide by training standard deviation
    train_data = np.subtract(train_data, train_mean)
    train_data = np.divide(train_data, train_std)

    # Subtract the training mean from test data and divide by training std
    test_data = np.subtract(test_data, train_mean)
    test_data = np.divide(test_data, train_std)
    return train_data, true_r, test_data, true_t

def eval_prediction(predict, actual):
    '''Determine the accuracy, precision, recall, and fpr of the prediction'''
    # I'll admit this is sloppy and needs improvement, but this creates a 
    # confusion matrix and generates accuracy, precision, trp, and fpr
    conf_mat = np.zeros([2, 2])
    for i in range(len(predict)):
        if predict[i] > actual[i]:
            conf_mat[1, 0] += 1
        elif predict[i] < actual[i]:
            conf_mat[0, 1] += 1
        else:
            if predict[i] == 1:
                conf_mat[0, 0] += 1
            else:
                conf_mat[1, 1] += 1

    accuracy = np.trace(conf_mat) / np.sum(conf_mat)
    precision = np.sum(conf_mat[0, 0]) / np.sum(conf_mat[:,0])
    recall = np.sum(conf_mat[0, 0]) / np.sum(conf_mat[0,:])
    fpr = np.sum(conf_mat[1, 0]) / np.sum(conf_mat[1,:])
    return accuracy, precision, recall, fpr 

def gen_roc_curve(actual, score, res=200):
    '''This function generates values for plotting a ROC curve'''

    # Create an array to shift the scores along a threshold - init to zero
    score_shift = score[0,:] - np.min(score)

    # Apply the signum and remove zeros
    predict = np.sign(score_shift)
    predict = [1 if j==0 else j for j in predict]

    # Create arrays to hold the fpr and tpr
    fpr = np.zeros(res+1)
    tpr = np.zeros(res+1)

    # Create threshold steps based on the res argument
    step = (np.max(score) - np.min(score)) / res

    # Shift the score matrix through the full range of scores
    for i in range(res):
        _, _, tpr[i], fpr[i] = eval_prediction(predict, actual)
        score_shift -= step
        predict = np.sign(score_shift)

    return fpr, tpr

# ----------------------------------------------------------------------------
# ----------------------------- Experiment #1 --------------------------------
# ----------------------------------------------------------------------------

# Collect and split data into normalized training and testing sets
train_x, train_t, test_x, test_t = process_spam_data()

# Train an SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(train_x, train_t)

# Test SVM using test data
prediction = clf.predict(test_x)

# Calculate accuracy, precision, and recall
accu, prec, recall, fpr = eval_prediction(prediction, test_t)
print("Accuracy = " + str(accu))
print("Precision = " + str(prec))
print("Recall = " + str(recall))
print("False Positive Rate = " + str(fpr))

# Calculate weights and compute scores
w = np.dot(clf.dual_coef_, clf.support_vectors_)
score = np.dot(test_x, np.transpose(w))

# Generate false and true positive rates
fpr, tpr = gen_roc_curve(test_t, np.transpose(score))

# Plot results
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve)')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve\nSpambase SVM Classification')
plt.legend(loc="lower right")
plt.grid(True)
# plt.show()

plt.savefig('cs545_hw3_exp_1.png')
plt.clf()

# ----------------------------------------------------------------------------
# ----------------------------- Experiment #2 --------------------------------
# ----------------------------------------------------------------------------

# Collect and split data into normalized training and testing sets
train_x, train_t, test_x, test_t = process_spam_data()

# Train an SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(train_x, train_t)

# Calculate weights and compute scores
weights = np.dot(clf.dual_coef_, clf.support_vectors_)

# Get the absolute values of the weights and sort largest first
ind = np.fliplr(np.argsort(np.abs(weights)))
n_samp, n_feat = test_x.shape

# Initialize arrays to hold accuracy and number of features 
accu = np.zeros(n_feat-1)
feat = [i+1 for i in range(1, n_feat)]

# Initialize matrices to hold training and testing data
train_feat_x = train_x[:,ind[:,0]]
test_feat_x = test_x[:,ind[:,0]] 
for i in range(1, n_feat):
    # Add a new row to the training data
    new_col = train_x[:,ind[:,i]]
    train_feat_x = np.concatenate((train_feat_x, new_col), axis=1)

    # Add a new row to the test data
    new_col = test_x[:,ind[:,i]]
    test_feat_x = np.concatenate((test_feat_x, new_col), axis=1)

    # Train an SVM classifier
    clf = svm.SVC(kernel='linear')
    clf.fit(train_feat_x, train_t)

    prediction = clf.predict(test_feat_x)
    accu[i-1], _, _, _ = eval_prediction(prediction, test_t)

# Plot results
plt.figure()
lw = 2
plt.plot(feat, accu, color='darkorange', lw=lw, label='Accuracy Curve')
plt.ylim([0.6, 1.05])
plt.xlim([1, n_feat+1]) 
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Features Trained\nSorted by Largest Weights')
plt.legend(loc="lower right")
plt.grid(True)
# plt.show()

plt.savefig('cs545_hw3_exp_2.png')
plt.clf()

# ----------------------------------------------------------------------------
# ----------------------------- Experiment #3 --------------------------------
# ----------------------------------------------------------------------------

# Collect and split data into normalized training and testing sets
train_x, train_t, test_x, test_t = process_spam_data()

# Generate random list to loop through features
n_samp, n_feat = test_x.shape
temp = np.empty((1,n_feat))
temp[0,:] = sample(range(123456789), n_feat)
ind = np.fliplr(np.argsort(temp))

# Initialize arrays to hold accuracy and number of features 
accu = np.zeros(n_feat-1)
feat = [i+1 for i in range(1, n_feat)]

# Initialize matrices to hold training and testing data
train_feat_x = train_x[:,ind[:,0]]
test_feat_x = test_x[:,ind[:,0]] 
for i in range(1, n_feat):
    # Add a new row to the training data
    new_col = train_x[:,ind[:,i]]
    train_feat_x = np.concatenate((train_feat_x, new_col), axis=1)

    # Add a new row to the test data
    new_col = test_x[:,ind[:,i]]
    test_feat_x = np.concatenate((test_feat_x, new_col), axis=1)

    # Train an SVM classifier
    clf = svm.SVC(kernel='linear')
    clf.fit(train_feat_x, train_t)

    prediction = clf.predict(test_feat_x)
    accu[i-1], _, _, _ = eval_prediction(prediction, test_t)

# Plot results
plt.figure()
lw = 2
plt.plot(feat, accu, color='darkorange', lw=lw, label='Accuracy Curve')
plt.ylim([0.6, 1.05])
plt.xlim([1, n_feat+1]) 
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Features Trained\nRandomly Added Features')
plt.legend(loc="lower right")
plt.grid(True)
# plt.show()

plt.savefig('cs545_hw3_exp_3.png')
plt.clf()

