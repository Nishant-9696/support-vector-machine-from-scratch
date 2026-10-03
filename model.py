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

# Step 7 - compute_gradients (not yet solved)
# TODO: implement

# Step 8 - apply_update (not yet solved)
# TODO: implement

# Step 9 - train_svm (not yet solved)
# TODO: implement

# Step 10 - predict_labels (not yet solved)
# TODO: implement

# Step 11 - accuracy_score (not yet solved)
# TODO: implement

