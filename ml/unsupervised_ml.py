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



def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components 
    
    Input:
        pca - the result of instantian of PCA in scikit learn
            
    Output:
            None
    '''
    num_components=len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
 
    plt.figure(figsize=(15, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')

def map_component_to_features(pca, component, column_names):
    '''
    Map weights for the component number to corresponding feature names
    
    Input:
        pca : object of pca class
        component (int): number of component to map to feature
        column_names (list(str)): column names of DataFrame before PCA transformation
    
    Output:
        df_features (DataFrame): DataFrame with feature weight sorted by feature name
    
    '''
    weights_array = pca.components_[component]
    df_features = pd.DataFrame(weights_array, index=column_names, columns=['weight'])
    return  df_features.sort_values(by='weight',ascending=False).round(2)


def select_attributes_by_type(df):
    '''
    Return names of attributes that needs log transformation, binary attributes, 
    categorical and numerical attributes
    
    Input:
        df (DataFrame): input azdias DataFrame
        
    Output:
        log_transfor_attributes, binary_attributes, categorical_attributes, numerical_attributes: list of names 
    '''
    skew_threshold = 1.0
    
    #choose numeric continuous attribtues
    att_type = pd.read_csv('./data/attribute_types.csv', sep=',')
    continuous_attributes = att_type[(att_type["type"]=="numeric") & 
                                     (att_type["action"]=="keep")]["attribute"].values
    
    np.append(continuous_attributes, "EINGEFUEGT_AM")
    
    #plot continuous attributes distribution
    print("Continuous attributes distribution", continuous_attributes)
    log_transform_attributes = []
    for att in continuous_attributes:
        skew = df[att].skew()
        print("{} skew_value = {}.".format(att,skew))
        if abs(skew) > skew_threshold:
            log_transform_attributes.append(att)
        ax = df[att].plot(kind="hist", bins=40)
        ax.set_title(att)
        plt.show()
        plt.close()
    #print(log_transform_attributes)    
    #df[continuous_attributes].describe()
    
    #categorical attributes that need to be re-encoded (one hot encoded) 
    categorical_attributes = list(np.intersect1d(att_type[att_type["action"].isin(["onehot"])]
                                             ["attribute"].values, df.columns))
    
    #binary attributes
    binary_attributes = []
    for column in df:
        if len(df[column].value_counts())==2:
            binary_attributes.append(column)
    
    #all
    numerical_attributes = df.columns[~df.columns.isin(binary_attributes+
                                                       log_transform_attributes+categorical_attributes)]
    
    return log_transform_attributes, binary_attributes, categorical_attributes, numerical_attributes


def get_kmeans_score(data, center, batch_size=20000):
    '''
    returns the kmeans score regarding SSE for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the SSE score for the kmeans model fit to the data
    '''
    #instantiate kmeans
    kmeans = MiniBatchKMeans(n_clusters=center, batch_size=batch_size, random_state=42).fit(data)

    # Then fit the model to your data using the fit method
    model = kmeans.fit(data)

    # Obtain a score related to the model fit
    score = np.abs(model.score(data))

    return score

def get_clusters_attributes(cluster_pipeline, num_attributes, log_attributes, column_names):
    '''
    Function transforms clusters centers, by performin pca inverse transform,
    reverse scale for numerical(num_attributes) and atrributes with logarithmic transformation (log_attributes),
    and exponential transformation for log_attributes
    
    Input:
        cluster_pipeline: object of cluster_pipeline
        num_attributes: list of numerical attributes which were rescaled
        log_attributes: list of attributes wich had logarithmic transformation and also were rescaled
        column_names: names of all columns after Column Transformer operation
        
    Output:
        cluster_centers_df (DataFrame): DataFrame of cluster_centers with their attributes values
        
    '''

    pca_components = cluster_pipeline.named_steps['pca']
    kmeans = cluster_pipeline.named_steps['kmeans']
    transformer =  cluster_pipeline.named_steps['transform']

    cluster_centers = pca_components.inverse_transform(kmeans.cluster_centers_)
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=column_names)

    num_scale = transformer.named_transformers_['num'].named_steps['num_scale']
    log_scale = transformer.named_transformers_['log'].named_steps['log_scale']
    log_transform = transformer.named_transformers_['log'].named_steps['log_transform']

    cluster_centers_df[num_attributes] = num_scale.inverse_transform(cluster_centers_df[num_attributes])
    cluster_centers_df[log_attributes] = log_scale.inverse_transform(cluster_centers_df[log_attributes])
    cluster_centers_df[log_attributes] = log_transform.inverse_transform(cluster_centers_df[log_attributes])
    #cluster_centers_df[log_attributes] = cluster_centers_df[log_attributes].apply(lambda x: np.expm1(x))

    return cluster_centers_df  


