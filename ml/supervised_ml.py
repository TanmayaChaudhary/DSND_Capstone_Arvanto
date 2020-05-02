import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
from sklearn.preprocessing import FunctionTransformer, Imputer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import learning_curve



def make_pipeline(ct, model):
    '''
    Creates pipeline with  two steps: column transformer (ct) introduced in preprocessing step and classifier (model).
    
    Input:
        ct: object type that implements the “fit” and “transform” methods
        model: object type that implements the “fit” and “predict” methods
        
    Output:
        pipeline: object type with "fit" and "predict" methods 
    '''
    
    pipeline = Pipeline([
                        ('transform', ct), 
                        ('classifier', model)
                        ])
    return pipeline

def randomize(X, y):
    '''
    Returns randomized DataFrame X and  Series y
    
    Input:
        X: DataFrame
        y: Series
        
    Output:
        X2: randomized DataFrame X
        y2: randomized pandas Series y2
    '''
    X2 = pd.concat([X,y], axis=1)
    X2 = X2.sample(frac=1, random_state=42)
    y2 = X2["RESPONSE"]
    X2.drop(["RESPONSE"], axis=1)
    return X2, y2

def draw_learning_curves(X, y, estimator, num_trainings):
    '''
    Draw learning curve that shows the validation and training auc_score of an estimator 
    for varying numbers of training samples.
    
    Input:
        X: array like sample
        y: array like target relative to X2 sample
        estimator: object type that implements the “fit” and “predict” methods
        num_trainings (int): number of training samples to plot
        
    Output:
        None
    '''
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=None, scoring = 'roc_auc', train_sizes=np.linspace(.1, 1.0, num_trainings))

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    print("Roc_auc train score = {}".format(train_scores_mean[-1].round(2)))
    print("Roc_auc validation score = {}".format(test_scores_mean[-1].round(2)))
    plt.grid()

    plt.title("Learning Curves")
    plt.xlabel("% of training set")
    plt.ylabel("Score")

    plt.plot(np.linspace(.1, 1.0, num_trainings)*100, train_scores_mean, 'o-', color="g",
             label="Training score")
    plt.plot(np.linspace(.1, 1.0, num_trainings)*100, test_scores_mean, 'o-', color="y",
             label="Cross-validation score")

    plt.yticks(np.arange(0.45, 1.02, 0.05))
    plt.xticks(np.arange(0., 100.05, 10))
    plt.legend(loc="best")
    print("")
    plt.show()


