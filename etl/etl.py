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



class DummiesTransformer(BaseEstimator, TransformerMixin):
    '''
    A Dataframe transformer that provide dummy variable encoding
    
    '''    
    def transform(self, X, **transformparams):
        '''
        Returns a dummy variable encoded version of a DataFrame
        
        Inputs:
            X :pandas DataFrame
        
        Returns:
            X_transformed: pandas DataFrame with new encoding    
        '''
        # check that we have a DataFrame with same column names as the one we fit
        if set(self._columns) != set(X.columns):
            raise ValueError('Passed DataFrame has different columns than fit DataFrame')
        elif len(self._columns) != len(X.columns):
            raise ValueError('Passed DataFrame has different number of columns than fit DataFrame')
          
        # create separate array for new encoded categoricals
        X_cat = np.empty((len(X), self._total_cat_cols), dtype='int')
        i = 0
        for col in self._columns:
            vals = self._cat_cols[col]
            for val in vals:
                X_cat[:, i] = X[col] == val
                i += 1
                
        return pd.DataFrame(data=X_cat, columns=self._feature_names)

    
    def fit(self, X, y=None, **fitparams):
        # Assumes X is a DataFrame
        self._columns = X.columns.values
        self._feature_names = []
        self._total_cat_cols = 0
        # Create a dictionary mapping categorical column to unique values
        self._cat_cols = {}
        for col in self._columns: 
            vals = X[col].value_counts().index.values
            vals = np.sort(vals)
            col_names = [col + '_' + str(int(val)) for val in vals]
            self._feature_names = np.append(self._feature_names, col_names)
            self._cat_cols[col] = vals
            self._total_cat_cols += len(vals)
        return self
    
    def get_feature_names(self):
        return self._feature_names


# 1. 'DIAS Attributs - Values 2017.xlsx' file was converted to csv file 'DIAS_Attributes_Values_2017.csv'. 
# 2. create_missing_code_dict function is created to create dictionary of missings keys for attributes


def get_missing_by_column(df):
    '''
    Calculates number of Nan in each column and returns an array of total number of missing values
    
    Input:
        df (DataFrame): Dataset for which columns the missing values will be calculated
        
    Output:
        n_missing_array: array of number of missing valeus
    '''       
    n_missing_array = []
    for col in df.columns:
        #count Nan in azdias
        try:
            n_missing_array.append(df[col].isnull().value_counts()[1])
        except:
            n_missing_array.append(0)   
            
    return n_missing_array


def create_missing_key_dict(attr_value_file):
    '''
    Read DIAS_Attributes_Values_2017.csv and parse missing and unkonwn keys for all attributes
    
    Input:
        attr_value_file (str): path to DIAS_Attributes_Values_2017.csv file
    
    Output:
        missing_keys_dict (dict): dictionary of attributes with values being an array of missing keys
    '''
    attr_values = pd.read_csv(attr_value_file, sep=',')
    missing_keys = attr_values[attr_values["Meaning"].isin(["unknown","unknown / no main age detectable"])]#, 
#                                                            "no transaction known", "no transactions known"])]

    missing_keys_dict = {}
    for _, row in missing_keys.iterrows():
        key = row["Attribute"]
        missing_keys_dict[key] = row["Value"].split(",")

        #Treat D13_*_RZ values as correponding D13_*
        #Replace D13_*_RZ in dictionary by D13_*
        #if "_RZ" in key:
        #    new_key = key.replace("_RZ", "")
        #    missing_keys_dict[new_key] = missing_keys_dict.pop(key)
    
    no_transaction_attributes = ['D19_BANKEN_DIREKT_RZ', 'D19_BANKEN_GROSS_RZ', 'D19_BANKEN_LOKAL_RZ', 'D19_BANKEN_REST_RZ',
                                 'D19_BEKLEIDUNG_GEH_RZ', 'D19_BEKLEIDUNG_REST_RZ', 'D19_BILDUNG_RZ', 'D19_BIO_OEKO_RZ',
                                 'D19_BUCH_RZ', 'D19_DIGIT_SERV_RZ', 'D19_DROGERIEARTIKEL_RZ', 'D19_ENERGIE_RZ', 
                                 'D19_FREIZEIT_RZ', 'D19_GARTEN_RZ', 'D19_HANDWERK_RZ', 'D19_HAUS_DEKO_RZ', 'D19_KINDERARTIKEL_RZ'
                                 'D19_KOSMETIK_RZ', 'D19_LEBENSMITTEL_RZ', 'D19_LOTTO_RZ', 'D19_NAHRUNGSERGAENZUNG_RZ', 
                                 'D19_RATGEBER_RZ', 'D19_SAMMELARTIKEL_RZ', 'D19_SCHUHE_RZ', 'D19_SONSTIGE_RZ', 
                                 'D19_TECHNIK_RZ', 'D19_TELKO_MOBILE_RZ', 'D19_TELKO_REST_RZ', 'D19_TIERARTIKEL_RZ',
                                 'D19_VERSAND_REST_RZ', 'D19_VERSICHERUNGEN_RZ', 'D19_VOLLSORTIMENT_RZ', 'D19_WEIN_FEINKOST_RZ']

    #[missing_keys_dict[key]=0 for key in no_transaction_attributes]
    
    #Treat D13_*_RZ values as correponding D13_*
    #Replace D13_*_RZ in dictionary by D13_*
    for key in no_transaction_attributes:
        new_key = key.replace("_RZ", "")
        missing_keys_dict[new_key] = ['0']        

    missing_keys_dict["CAMEO_INTL_2015"] = ['XX']
    missing_keys_dict["CAMEO_DEUG_2015"] = ['X','XX']
    missing_keys_dict["CAMEO_DEU_2015"] =['XX']
    
    missing_keys_dict["GEBURTSJAHR"] = ['0']
    
    #add information about missing codes from similar attributes
    missing_keys_dict["CAMEO_INTL_2015"] == missing_keys_dict["CAMEO_DEUINTL_2015"]
    #missing_keys_dict["D19_BANKEN_REST"] == missing_keys_dict["D19_BANKEN_LOKAL"]
    missing_keys_dict["KBA13_CCM_1401_2500"] = missing_keys_dict["KBA13_CCM_1400_2500"]
    missing_keys_dict["KBA13_BAUMAX"] = missing_keys_dict["KBA05_BAUMAX"]
    missing_keys_dict["KBA13_ANTG1"] =  missing_keys_dict["KBA05_ANTG1"]
    missing_keys_dict["KBA13_ANTG2"] =  missing_keys_dict["KBA05_ANTG2"]
    missing_keys_dict["KBA13_ANTG3"] =  missing_keys_dict["KBA05_ANTG3"]
    missing_keys_dict["KBA13_ANTG4"] =  missing_keys_dict["KBA05_ANTG4"]    
    
    #missing_keys_dict["D19_BUCH_CD"] =  missing_keys_dict["D19_BUCH"]  
    
    missing_keys_dict["KOMBIALTER"] = ['9']
    #missing_keys_dict["D19_VERSI_DATUM"] =  missing_keys_dict["D19_VERSAND_DATUM"]
    #missing_keys_dict["D19_VERSI_OFFLINE_DATUM"] =  missing_keys_dict["D19_VERSAND_OFFLINE_DATUM"]    
    #missing_keys_dict["D19_VERSI_ONLINE_DATUM"] =  missing_keys_dict["D19_VERSAND_ONLINE_DATUM"]    
    
    return missing_keys_dict



def get_missing_by_column(df):
    '''
    Calculates number of Nan in each column and returns an array of total number of missing values
    
    Input:
        df (DataFrame): Dataset for which columns the missing values will be calculated
        
    Output:
        n_missing_array: array of number of missing valeus
    '''       
    n_missing_array = []
    for col in df.columns:
        #count Nan in azdias
        try:
            n_missing_array.append(df[col].isnull().value_counts()[1])
        except:
            n_missing_array.append(0)   
            
    return n_missing_array



def convert_keys_to_nan(df, keys_dict):
    '''
    Replaces given keys from keys_dict to np.nan in df inplace
    
    Input:
        df (DataFrame): Dataset for which keys to np.nan need to be converted
        keys_dict: dictionary of attributes with keys that needs to be converted to np.nan
        
    Output:
        None
    '''
    for attribute in keys_dict:
        if attribute in df.columns:       
            keys_array = keys_dict[attribute]
            #print(attribute, keys_array)
            for key in keys_array:
                if key == 'X' or key == 'XX':
                    key = str(key)
                else:
                    key = int(key)
                    #df[attribute].value_counts()
                df[attribute].replace(key, np.NaN, inplace=True)

        else:
            print("Attribute {} is not available in DataFrame.".format(attribute))

def create_missing_info_df(df, ini_missing=None):
    '''
    Create missing info data frame with each row being an attribute, and columns representing
    number of initially missing rows ("ini_missing"), 
    missing after np.nan encoding ("final_missing", and percentage of missing values ("percent_missing")
    DataFrame is sorted by "percent missing" from highest to lowest.
    
    Input:
        df (DataFrame)
        ini_missing: array of initially missing values, default None
    
    Output:
        sorted_missing_info (DataFrame): created dataframe of number of missing rows in each attribute
    '''
    
    missing_info = pd.DataFrame(data=df.columns, columns=["Attribute"])
    
    if ini_missing !=None:
        missing_info["ini_missing"] = ini_missing
        
    missing_info['final_missing'] = get_missing_by_column(df)

    total = df.shape[0]
    #calculate percent of total missing value by attribute
    missing_info['percent_missing'] = missing_info['final_missing']/total*100

    #DataFrame of missing attributes sorted from lowest to highest
    sorted_missing_info = missing_info.sort_values(by='final_missing', ascending=False)
    
    return sorted_missing_info

def plot_attribute_distribution(df, row_names, attribute, n):
    '''
    Plot Distribution of attribute n first rows from sorted DataFrame
    
    Input:
        df (DataFrame): Sorted dataset by given attribute
        row_names (str): name of clolumn with names
        attribute (str): name of attribute which distribution will be plotted
        n: number of rows that will be plotted
        
    Output:
        None
    '''
    ax = df[:n].plot(x = row_names ,y = attribute,  kind='barh', figsize=(5,15))
    ax.invert_yaxis()
    ax.set_xlabel(attribute, size='large')
    ax.set_ylabel(row_names, size='large');
    ax.set_title('Distribution of {}'.format(attribute), size='large')


def engineer_PRAEGENDE_JUGENDJAHRE(df):
    '''
    Engineer two new attributes from PRAEGENDE_JUGENDJAHRE: MOVEMENT and GENERATION_DECADE

    PRAEGENDE_JUGENDJAHRE initial encoding
    Dominating movement of person's youth (avantgarde vs. mainstream; east vs. west)
    - -1: unknown
    -  0: unknown
    -  1: 40s - war years (Mainstream, E+W)
    -  2: 40s - reconstruction years (Avantgarde, E+W)
    -  3: 50s - economic miracle (Mainstream, E+W)
    -  4: 50s - milk bar / Individualisation (Avantgarde, E+W)
    -  5: 60s - economic miracle (Mainstream, E+W)
    -  6: 60s - generation 68 / student protestors (Avantgarde, W)
    -  7: 60s - opponents to the building of the Wall (Avantgarde, E)
    -  8: 70s - family orientation (Mainstream, E+W)
    -  9: 70s - peace movement (Avantgarde, E+W)
    - 10: 80s - Generation Golf (Mainstream, W)
    - 11: 80s - ecological awareness (Avantgarde, W)
    - 12: 80s - FDJ / communist party youth organisation (Mainstream, E)
    - 13: 80s - Swords into ploughshares (Avantgarde, E)
    - 14: 90s - digital media kids (Mainstream, E+W)
    - 15: 90s - ecological awareness (Avantgarde, E+W)
    
    Final encooding:
    
    "MOVEMENT": 
    - 1: Mainstream
    - 2: Avantgarde

    “GENERATION_DECADE”:
    - 4: 40s
    - 5: 50s
    - 6: 60s
    - 7: 70s
    - 8: 80s
    - 9: 90s

    Input:
        df (DataFrame): dataframe that has PRAEGENDE_JUGENDJAHRE attribute

    Output:
        df_new (DataFrame): dataframe with new attributes MOVEMENT and GENERATION, PRAEGENDE_JUGENDJAHRE attribute 
                            is removed
    
    '''

    #create new binary attribute MOVEMENT with values Avantgarde (0) vs Mainstream (1)
    df['MOVEMENT'] = df['PRAEGENDE_JUGENDJAHRE']
    df['MOVEMENT'].replace([-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
                           [np.nan,np.nan,1,2,1,2,1,2,2,1,2,1,2,1,2,1,2], inplace=True) 

    #create new ordinal attribute GENERATION_DECADE with values 40s, 50s 60s ... encoded as 4, 5, 6 ...
    df['GENERATION_DECADE'] = df['PRAEGENDE_JUGENDJAHRE']
    df['GENERATION_DECADE'].replace([-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
                                    [np.nan,np.nan,4,4,5,5,6,6,6,7,7,8,8,8,8,9,9], inplace=True) 

    #delete 'PRAEGENDE_JUGENDJAHRE'
    df_new = df.drop(['PRAEGENDE_JUGENDJAHRE'], axis=1)
    
    return df_new

def engineer_CAMEO_INTL_2015(df):
    '''
    Engineer two new attributes from CAMEO_INTL_2015: WEALTH and LIFE_AGE

    CAMEO_INTL_2015 initial encoding
    German CAMEO: Wealth / Life Stage Typology, mapped to international code
    - -1: unknown
    - 11: Wealthy Households - Pre-Family Couples & Singles
    - 12: Wealthy Households - Young Couples With Children
    - 13: Wealthy Households - Families With School Age Children
    - 14: Wealthy Households - Older Families &  Mature Couples
    - 15: Wealthy Households - Elders In Retirement
    - 21: Prosperous Households - Pre-Family Couples & Singles
    - 22: Prosperous Households - Young Couples With Children
    - 23: Prosperous Households - Families With School Age Children
    - 24: Prosperous Households - Older Families & Mature Couples
    - 25: Prosperous Households - Elders In Retirement
    - 31: Comfortable Households - Pre-Family Couples & Singles
    - 32: Comfortable Households - Young Couples With Children
    - 33: Comfortable Households - Families With School Age Children
    - 34: Comfortable Households - Older Families & Mature Couples
    - 35: Comfortable Households - Elders In Retirement
    - 41: Less Affluent Households - Pre-Family Couples & Singles
    - 42: Less Affluent Households - Young Couples With Children
    - 43: Less Affluent Households - Families With School Age Children
    - 44: Less Affluent Households - Older Families & Mature Couples
    - 45: Less Affluent Households - Elders In Retirement
    - 51: Poorer Households - Pre-Family Couples & Singles
    - 52: Poorer Households - Young Couples With Children
    - 53: Poorer Households - Families With School Age Children
    - 54: Poorer Households - Older Families & Mature Couples
    - 55: Poorer Households - Elders In Retirement
    - XX: unknown

    Final encooding:
    
    "WEALTH"
    - 1: Wealthy Households
    - 2: Prosperous Households
    - 3: Comfortable Households
    - 4: Less Affluent Households
    - 5: Poorer Households

    "LIFE_AGE"
    - 1: Pre-Family Couples & Singles
    - 2: Young Couples With Children
    - 3: Families With School Age Children
    - 4: Older Families &  Mature Couples
    - 5: Elders In Retirement
    
    Input:
        df (DataFrame): dataframe that has CAMEO_INTL_2015 attribute

    Output:
        df_new (DataFrame): dataframe with new attributes WEALRH and LIFE_AGE; CAMEO_INTL_2015 attribute 
                            is removed
    
    '''
    #create new ordinal attribute WEALTH
    df['WEALTH'] = df['CAMEO_INTL_2015'].str[:1].astype(float)


    #create new ordinal attribute LIFE_AGE
    df['LIFE_AGE'] = df['CAMEO_INTL_2015'].str[1:2].astype(float)

    #delete 'CAMEO_INTL_2015'
    df_new = df.drop(['CAMEO_INTL_2015'], axis=1)
    
    return df_new
   

def engineer_WOHNLAGE(df):
    '''
    Engineer RURAL_NEIGHBORHOOD from WOHNLAGE attribute
    "WOHNLAGE" feature could be divided into “RURAL_NEIGHBORHOOD” and “QUALITY_NEIGHBORHOOD”.
    However, there are 24% of rural data that will have missing values in "QUALITY_NEIGHBORHOOD”
    feature, therefore only binary "RURAL_NEIGHBORHOOD" feature was created inplace of "WOHNLAGE"

    Initial encoding of WOHNLAGE:
    Neighborhood quality (or rural flag)
    - -1: unknown
    -  0: no score calculated
    -  1: very good neighborhood
    -  2: good neighborhood
    -  3: average neighborhood
    -  4: poor neighborhood
    -  5: very poor neighborhood
    -  7: rural neighborhood
    -  8: new building in rural neighborhood

    Final encooding:

    "RURAL_NEIGBORHOOD"
    - 0: Not Rural
    - 1: Rural

    Input:
        df (DataFrame): dataframe that has WOHNLAGE attribute

    Output:
        df_new (DataFrame): dataframe with new attribute RURAL_NEIGBORHOOD; WOHNLAGE attribute
                            is removed


'''

    #create new binary attribute RURAL_NEIGHBORHOOD with values Rural (1) vs NotRural(0)
    df['RURAL_NEIGHBORHOOD'] = df['WOHNLAGE']
    df['RURAL_NEIGHBORHOOD'].replace([-1,0,1,2,3,4,5,7,8], [np.nan,np.nan,0,0,0,0,0,1,1], inplace=True)

    #delete 'WOHNLAGE'
    df_new = df.drop(['WOHNLAGE'], axis=1)

    return df_new

def engineer_PLZ8_BAUMAX(df):
    '''
    Engineer PLZ8_BAUMAX_BUSINESS and PLZ8_BAUMAX_FAMILY attributes from PLZ8_BAUMAX attribute
   
    PLZ8_BAUMAX initial encoding:
    Most common building type within the PLZ8 region
    - -1: unknown
    -  0: unknown
    -  1: mainly 1-2 family homes
    -  2: mainly 3-5 family homes
    -  3: mainly 6-10 family homes
    -  4: mainly 10+ family homes
    -  5: mainly business buildings
    
    Final encoding:
    “PLZ8_BAUMAX_BUSINESS”
    - 0: Not Business
    - 1: Business

    “PLZ8_BAUMAX_FAMILY”
    - 0: 0 families
    - 1: mainly 1-2 family homes
    - 2: mainly 3-5 family homes
    - 3: mainly 6-10 family homes
    - 4: mainly 10+ family homes
    
    Input:
        df (DataFrame): dataframe that has PLZ8_BAUMAX attribute

    Output:
        df_new (DataFrame): dataframe with new attributes PLZ8_BAUMAX_BUSINESS and PLZ8_BAUMAX_FAMILY; PLZ8_BAUMAX attribute 
                            is removed
    
    '''

    #create new binary attribute PLZ8_BAUMAX_BUSINESS with values Business (1) vs Not Business(0)
    df['PLZ8_BAUMAX_BUSINESS'] = df['PLZ8_BAUMAX']
    df['PLZ8_BAUMAX_BUSINESS'].replace([1,2,3,4,5], [0,0,0,0,1], inplace=True) 

    #create new ordinal attribute PLZ8_BAUMAX_FAMILY with from 1 to 4 encoded as in data dictionary
    df['PLZ8_BAUMAX_FAMILY'] = df['PLZ8_BAUMAX']
    df['PLZ8_BAUMAX_FAMILY'].replace([5], [0], inplace=True) 

    #delete 'PLZ8_BAUMAX'
    df_new = df.drop(['PLZ8_BAUMAX'], axis=1)
    
    return df_new

def check_columns_numeric(df):
    '''
    Check whether all columns are of numeric dtype
    
    Input:
        df (DataFrame)
    Output:
        result (Bool): all numeric columns True or False
    '''
    
    result = True
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column])==False:
            print("{} is not numeric.".format(column))
            #print(df[column])
            result = False
            
    return result

def plot_distribution_comparison(df1, label1, df2, label2, attribute_list):
    '''
    Plot distribution of two datasets for 6 six random attributes from given attribute list

    Input:
        df1 (DataFrame): first dataset
        label1 (str): label for df1
        df2 (DataFrame): second dataset
        label2 (str): label for df2
        attribute_list: array of attributes

    Output:
        None
    '''

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 6))

    attribute_list = [attribute_list[random.randint(0, len(attribute_list)-1)] for i in range (6)]
    #print(attribute_list)
    for i, attribute in enumerate(attribute_list[:6]):
        #print(attribute)
        axes_0 = math.trunc(i/3)
        axes_1 = i%3
        ax = axes[axes_0,axes_1]

        df1[attribute].plot(ax = ax,alpha=0.7, kind='hist', label = label1)
        df2[attribute].plot(ax = ax,alpha=0.7, kind='hist', label = label2)
        ax.set_title(attribute)
        if axes_1 != 0:
            ax.get_yaxis().set_visible(False)
        else:
            ax.set_ylabel("Number of rows")
            ax.legend(loc="upper left")


def clean_data(df, test_data=False):
    '''
    Perform feature trimming, row dropping for a given DataFrame 
    
    Input:
        df (DataFrame)
        test_data (Bool): df is test data, so no rows should be dropped, default=False
    Output:
        cleaned_df (DataFrame): cleaned df DataFrame
    '''
    
    # convert missing value codes into NaNs, ... 
    print("Convert missing value codes into NaNs")
    missing_keys_dict = create_missing_key_dict('../Arvato-Capstone/DIAS_Attributes_Values_2017.csv')
    convert_keys_to_nan(df, missing_keys_dict)
    #drops columns with more than 30% of missing values
    print("Drop columns with more than 30% of missing values")
    deleted_columns_1 = ['ALTER_KIND4', 'TITEL_KZ', 'ALTER_KIND3', 'D19_BANKEN_LOKAL', 'ALTER_KIND2', 'D19_DIGIT_SERV', 
     'D19_BIO_OEKO', 'D19_TIERARTIKEL', 'D19_NAHRUNGSERGAENZUNG', 'D19_GARTEN', 'D19_LEBENSMITTEL', 
     'D19_WEIN_FEINKOST', 'D19_ENERGIE', 'D19_BANKEN_REST', 'D19_BILDUNG', 'ALTER_KIND1', 
     'D19_BEKLEIDUNG_GEH', 'D19_RATGEBER', 'D19_SAMMELARTIKEL', 'D19_FREIZEIT', 'D19_BANKEN_GROSS', 
     'D19_SCHUHE', 'D19_HANDWERK', 'D19_TELKO_REST', 'D19_DROGERIEARTIKEL', 'D19_LOTTO', 'D19_VERSAND_REST', 
     'D19_BANKEN_DIREKT', 'D19_TELKO_MOBILE', 'D19_HAUS_DEKO', 'D19_BEKLEIDUNG_REST', 'AGER_TYP', 
     'D19_VERSICHERUNGEN', 'EXTSEL992', 'D19_TECHNIK', 'D19_VOLLSORTIMENT', 'KK_KUNDENTYP', 'D19_SONSTIGE', 
     'KBA05_BAUMAX','GEBURTSJAHR', 'ALTER_HH']
    
#    df_cleaned = df.drop(dropped_columns,axis=1)
    
    #obtain dropped attributes from attribute_types.csv
    att_type = pd.read_csv('./data/attribute_types.csv', sep=',')
    deleted_columns_2 = att_type[att_type["action"].isin(["drop"])]["attribute"].values

    print("Drop columns indicated in attribute_types.csv file")

    deleted_columns = list(deleted_columns_1) + list(deleted_columns_2)
    df_cleaned = df.drop(deleted_columns,axis=1)
    

    #remove rows with more than 25 missing attributes if it not a testing data, skip this step if it is test data
    df_cleaned['n_missing'] = df_cleaned.isnull().sum(axis=1)
    #ax = df_cleaned["n_missing"].plot(kind='hist', bins=40)
    
    try:
        df_cleaned = df_cleaned.drop(['PRODUCT_GROUP','CUSTOMER_GROUP','ONLINE_PURCHASE'], axis=1)
    except:
        pass
    
    if not test_data:
        print("Remove rows with more than 25 missing attributes")
        df_cleaned = df_cleaned[df_cleaned["n_missing"]<= 25].drop("n_missing", axis=1)
    else:
        df_cleaned = df_cleaned.drop("n_missing", axis=1)
    
   
    #O -> 0, W -> 1
    print("Reencode OST_WEST_KZ attribute")
    df_cleaned['OST_WEST_KZ'].replace(['O','W'], [0, 1], inplace=True)
    
    #change EINGEFUEGT_AM to year format
    print("Change EINGEFUEGT_AM to year")
    df_cleaned["EINGEFUEGT_AM"] = pd.to_datetime(df_cleaned["EINGEFUEGT_AM"], format='%Y/%m/%d %H:%M')
    df_cleaned["EINGEFUEGT_AM"] = df_cleaned["EINGEFUEGT_AM"].dt.year
    
    #print(df_cleaned.shape)
    #engineer PRAEGENDE_JUGENDJAHRE
    print("Engineer PRAEGENDE_JUGENDJAHRE")
    df_cleaned = engineer_PRAEGENDE_JUGENDJAHRE(df_cleaned)
    
    #engineer CAMEO_INTL_2015
    print("Engineer CAMEO_INTL_2015")
    df_cleaned = engineer_CAMEO_INTL_2015(df_cleaned)
    
    #engineer_WOHNLAGE
    print("Engineer WOHNLAGE")
    df_cleaned = engineer_WOHNLAGE(df_cleaned)
    
    #engineer_PLZ8_BAUMAX
    print("Engineer PLZ8_BAUMAX")
    df_cleaned = engineer_PLZ8_BAUMAX(df_cleaned)  
    
    #change object type of CAMEO_DEUG_2015 to numeric type
    df_cleaned["CAMEO_DEUG_2015"] = pd.to_numeric(df_cleaned["CAMEO_DEUG_2015"])
        
    return df_cleaned


def tukey_rule(df, column_name):
    '''
    Function that uses the Tukey rule to detect outliers in a dataframe column
    and then removes that entire row from the data frame

    Input:
        df (DataFrame): DataFrame
        column_name (str): column name base on wich outliers will be analyzed and removed
    Output:
        df_reduced (DataFrame): new DataFrame with the outliers eliminated

    '''

    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)

    IQR = Q3 - Q1

    max_value = Q3 + 1.5 * IQR
    min_value = Q1 - 1.5 * IQR

    df_reduced = df[~((df[column_name] > max_value) | (df[column_name] < min_value))]



    #data_frame[(data_frame[column_name] < max_value) & (data_frame[column_name] > min_value)]

    return df_reduced

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


