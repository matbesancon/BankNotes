# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 10:06:24 2016

@author: MATHIEU
@description: Buidling a GL model
"""

def prob_log(x,w):
    """
    probability of an observation belonging 
    to the class "one"
    given the predictors x and weights w
    """
    return np.exp(np.dot(x,w))/(np.exp(np.dot(x,w))+1)

def grad_log_like(X, y, w):
    """
    computes the gradient of the log-likelihood from predictors X,
    output y and weights w
    """
    return np.dot(X.T,y- np.apply_along_axis(lambda x: prob_log(x,w),1,X)).reshape((len(w),))
    
def learn_weights(df):
    """
    computes and updates the weights until convergence
    given the features and outcome in a data frame
    """
    X = np.c_[np.ones(len(df)),np.array(df.iloc[:,:df.shape[1]-1])]
    y = np.array(df["class"])
    niter = 0
    error = .0001
    w = np.zeros((df.shape[1],))
    w0 = w+5
    alpha = .3
    while sum(abs(w0-w))>error and niter < 10000:
        niter+=1
        w0 = w
        w = w + alpha*min(.1,(3/(niter**.5+1)))*(grad_log_like(X,y,w))
    if niter==10000:
        print("Maximum iterations reached")
    return w


def predict_outcome(df,w):
    """
    takes in a test data set and computed weights
    returns a vector of predicted output, the confusion matrix 
    and the total errors
    """
    confusion_matrix = np.zeros((2,2))
    p = []
    for line in df.values:
        x = np.append(1,line[0:df.shape[1]-1])
        p.append(prob_log(x,w))
        if (prob_log(x,w)>.5) and line[3]:
            confusion_matrix[1,1]+=1
        elif (prob_log(x,w)<.5) and line[3]:
            confusion_matrix[1,0]+=1
        elif (prob_log(x,w)<.5) and not line[3]:
            confusion_matrix[0,0]+=1
        else:
            confusion_matrix[0,1]+=1
    return p, confusion_matrix, len(df)-sum(np.diag(confusion_matrix))
    
error = []
weights = []
for test in range(100):
    trainIndex = np.random.rand(len(data0)) < 0.85
    data_train = data1[trainIndex]
    data_test = data1[~trainIndex]
    weights.append(learn_weights(data_train))
    error.append(predict_outcome(data_test,weights[-1])[2])
    
plt.grid()
plt.plot(error)
plt.plot(error,'r.')
plt.xlabel("Test")
plt.ylabel("Errors")
plt.title("Misclassified points")
plt.savefig("figures/GLM_errors.png")
plt.show()

# np.mean(error)/len(data1)
# 0.0035349854227405245




