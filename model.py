"""
Support Vector Machine from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standardize_features
from sklearn.preprocessing import StandardScaler

def standardize_features(x):
    S=StandardScaler()
    scaled=S.fit_transform(x)
    return scaled

# Step 2 - initialize_parameters
import numpy as np

def initialize_parameters(n_features):
    w = np.zeros(n_features)
    b = 0.0

    return {'w': w, 'b': b}

# Step 3 - compute_scores
import numpy as np

def compute_scores(x, params):
    result = []

    for i in range(len(x)):
        result.append(x[i] @ params['w'] + params['b'])

    return np.array(result)

# Step 4 - predict_from_scores
import numpy as np

def predict_from_scores(scores):
    result=[]
    for i in range(len(scores)):
        if scores[i] >= 0:
            result.append(1)
        else:
            result.append(-1)
    return np.array(result)

# Step 5 - hinge_loss_example
def hinge_loss_example(score, y):
    return max(0, 1 - y * score)

# Step 6 - svm_objective
import numpy as np

def svm_objective(x, y, params, reg_lambda):
    scores = x @ params['w'] + params['b']
    
    hinge = np.maximum(0, 1 - y * scores)
    
    loss = np.mean(hinge)
    
    regularization = reg_lambda * np.dot(params['w'], params['w'])
    
    return loss + regularization

# Step 7 - compute_gradients
import numpy as np

def compute_gradients(x, y, params, reg_lambda):
    scores = compute_scores(x, params)

    margins = 1 - y * scores

    active = margins > 0

    n = len(y)

    dw = -np.sum(y[active, None] * x[active], axis=0) / n
    dw = dw + 2 * reg_lambda * params['w']

    db = -np.sum(y[active]) / n

    return {'dw': dw, 'db': float(db)}

# Step 8 - apply_update
def apply_update(params, grads, learning_rate):
    # TODO: return a new params dict after one gradient-descent step on 'w' and 'b'.
    new={}
    new['w']=params['w']-learning_rate*grads['dw']
    new['b']=params['b']-learning_rate*grads['db']
    return new

# Step 9 - train_svm
def train_svm(x, y, learning_rate, reg_lambda, n_epochs):
    params = initialize_parameters(x.shape[1])
    for _ in range(n_epochs):
        grads = compute_gradients(x, y, params, reg_lambda)
        params = apply_update(params, grads, learning_rate)
    return params

# Step 10 - predict_labels
import numpy as np

def predict_labels(x, params):
    scores = x @ params['w'] + params['b']
    return np.where(scores >= 0, 1, -1)

# Step 11 - accuracy_score
import numpy as np

def accuracy_score(y_pred, y_true):
    correct = (y_pred == y_true)
    accuracy = np.mean(correct)
    return accuracy

