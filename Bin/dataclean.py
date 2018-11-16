import sys
import os
import openml as oml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import random
import pprint

from IPython.core.display import display, HTML
from matplotlib.ticker import FormatStrFormatter
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from scipy.io import savemat
from bayesian.bin import abda

from sklearn.preprocessing import Imputer
from fancyimpute import KNN, IterativeImputer, MatrixFactorization
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import roc_auc_score
import time
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn import svm

from pandas.plotting import parallel_coordinates

plt.rcParams['figure.dpi']= 120
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8 

print("Current working directory")
print(os.getcwd()) # get current working directory

# hide print       
class NoStdStreams(object):
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()


##### Show important features #####

def build_forest(X, y):
    forest = Pipeline([('Imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
                       ('classifiers', RandomForestClassifier(n_estimators=100, random_state=0))])
    forest.fit(X,y)
    
    importances = forest.steps[1][1].feature_importances_
    indices = np.argsort(importances)[::-1]
    return importances, indices

def plot_feature_importances(dataset_name, features, importances, indices):
    a = 0.8
    f_sub = []
    max_features = 20

    for f in range(min(len(features), max_features)): 
            f_sub.append(f)

    # Create a figure of given size
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    # Set title
    ttl = dataset_name

    df = pd.DataFrame(importances[indices[f_sub]][::-1])
    df.plot(kind='barh', ax=ax, alpha=a, legend=False, edgecolor='w', 
            title=ttl, color = [plt.cm.viridis(np.arange(len(df))*10)])

    # Remove grid lines and plot frame
    ax.grid(False)
    ax.set_frame_on(False)

    # Customize title
    ax.set_title(ax.get_title(), fontsize=14, alpha=a, ha='left', x=0, y=1.0)
    plt.subplots_adjust(top=0.9)

    # Customize x tick lables
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.locator_params(axis='x', tight=True, nbins=5)

    # Customize y tick labels
    yticks = np.array(features)[indices[f_sub]][::-1]
    ax.set_yticklabels(yticks, fontsize=8, alpha=a)
    ax.yaxis.set_tick_params(pad=2) 
    ax.yaxis.set_ticks_position('none') 
    ax.set_ylim(ax.get_ylim()[0]-0.5, ax.get_ylim()[1]+0.5) 

    # Set x axis text
    xlab = 'Feature importance'
    ax.set_xlabel(xlab, fontsize=10, alpha=a)
    ax.xaxis.set_label_coords(0.5, -0.1)

    # Set y axis text
    ylab = 'Feature'
    ax.set_ylabel(ylab, fontsize=10, alpha=a)
    plt.show()

def show_important_features(X, y, data_name, features):
    display(HTML('<h2>Important Features</h2>'))
    importances, indices = build_forest(X, y)
    plot_feature_importances(data_name, features, importances, indices)
###################################

##### Show statistical information #####

def show_statistical_info(Xy):
    display(HTML('<h2>Statistical Information</h2>'))
    df = pd.DataFrame(Xy)
    display(df.describe())
########################################

#####  Discover Data Types #####

def infer_feature_type(feature):
    """
    Infer data types for the given feature using simple logic
    Possible data types to infer: boolean, date, float, integer, string
    Feature that is not either a boolean, a date, a float or an integer, is classified as a string
    """
    data_type = ""
    types = ["date","float64","int64","string"]
    weights = [0,0,0,0] #Weights corresponding to the data types
    feature_len = len(feature)
    
    indices_number = int(0.1 * feature_len) #Number of different values to check in a feature
    indices = random.sample(range(0,feature_len), min(indices_number,feature_len)) #Array of random indices
    
    # If the feature only contains two different unique values, then infer it as boolean
    if len(pd.unique(feature)) == 2:
        data_type = "bool"
    else:
        for i in indices:
            try:
                if (len(feature[i]) <= 10 
                    and (((feature[i][2:3]=='-' or feature[i][2:3]=='/') 
                    and (feature[i][5:6]=='-' or feature[i][5:6]=='/')) 
                    or ((feature[i][4:5]=='-' or feature[i][4:5]=='/')
                    and (feature[i][7:8]=='-' or feature[i][7:8]=='/')))):
                    weights[0] += 1 #Date
                else:
                    weights[3] += 1 #String
            except (TypeError,ValueError,IndexError):
                try:
                    int(feature[i]) # numeric
                    if ('.' in str(feature[i])):
                        if isinstance(feature[i], np.float64):
                            if feature[i].is_integer():
                                weights[2] += 1 # Integer
                            else:
                                weights[1] += 1 # Float
                        else:
                            weights[1] += 1 # Float
                    else:
                        weights[2] += 1 #Integer
                except (TypeError,ValueError,IndexError):
                    weights[3] += 1 #String
#         print ("Date: {}, Float64: {}, Int64: {}, String: {}".format(weights[0],weights[1],weights[2],weights[3])) #For debugging purposes
        data_type = types[weights.index(max(weights))]
        
    return data_type

def discover_type_heuristic(data):
    """
    Automatically infer data types for each feature in the dataframe using simple logic
    ---
    input:

    data : numpy array or dataframe (numeric data needs to be 64 bit)
        
    """
#     df = pd.DataFrame(data)
#     print(df)
    result = []
    if isinstance(data, np.ndarray):
        # convert float32 to float64
        data = np.array(data, dtype='float64')
        df = pd.DataFrame(data)
    else:
        df = data

    for column in df.columns:        
        #print ("Trying to automatically infer the data type of the",column,"feature...") #For debugging purposes
        type_inferred = infer_feature_type(df[column])
        result.append(type_inferred)
        #print ("Result:",inferredType) #For debugging purposes
    return result

def generate_mat(Xy, extra_cardinality=1):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = imp_mean.fit_transform(Xy)
    simple_types = discover_type_heuristic(data)
    # map simple types to meta types
    # 1: real (w positive: all real | positive | interval)
    # 2: real (w/o positive: all real | interval)
    # 3: binary data
    # 4: discrete (non-binary: categorical | ordinal | count)
    # note: for now, the implemented bayesian method version by isabel can distinguish between real, postive real, categorical and count
    # the binary data should be mapped to meta type 4 discrete instead of meta type 3 due to the limited implemented version. This may change
    # if the extended version has been implemented by isabel.
    meta_types = [] 
    for i in range(len(simple_types)):
#         print(simple_types[i])
        if simple_types[i] == "bool":
            meta_types.append(4) # may change in the future
        elif simple_types[i] == "int64" or simple_types[i] == "float64":
            if (len(np.unique(data[:,i])) < 0.02 * len(data[:,i]) and len(np.unique(data[:,i])) < 50):
                meta_types.append(4)
            else:
                if (data[:,i] > 0).all():
                    meta_types.append(1)
                else:
                    meta_types.append(2)
        else:
            meta_types.append(1)
    discrete_cardinality = [] # number of unique for discrete feature, 1 for others
    for i in range(len(meta_types)):
        if (meta_types[i] == 4):
            discrete_cardinality.append(int(np.max(data[:,i])) + extra_cardinality) 
        else:
            discrete_cardinality.append(1)
    data_dict = {'X': data,
                 'T': np.asarray(meta_types),
                 'R': np.asarray(discrete_cardinality)}
    # pprint.pprint(data_dict)
    savemat('bayesian/data/temp/data.mat', data_dict, oned_as='row')

def discover_type_bayesian(Xy):
    """
    Retrieve the key with the higher value from 'weights'
    The retrieved key is the statisical type of the corresponding feature
    
    ---
    input:

    Xy : numpy array
    
    """
    statistical_types = []
    generate_mat(Xy)
#     with HiddenPrints():
    with NoStdStreams():
        print("This will not be printed")
        weights = abda.main(seed=1337, dataset='bayesian/data/temp/data.mat', exp_id=None, args_output='./exp/temp/', args_miss=None, verbose=1,
         args_col_split_threshold=0.8, args_min_inst_slice=500, args_leaf_type='pm',
         args_type_param_map='spicky-prior-1', args_param_init='default', args_param_weight_init='uniform',
         args_n_iters=5, args_burn_in=4000, args_w_unif_prior=100, args_save_samples=1,
         args_ll_history=1, args_omega_prior='uniform', args_plot_iter=10, args_omega_unif_prior=10,
         args_leaf_omega_unif_prior=0.1, args_cat_unif_prior=1);
    for i in range(len(weights)):
#         print(max(weights[i], key=weights[i].get))
        statistical_types.append(str(max(weights[i], key=weights[i].get)))
    return statistical_types

def discover_types(Xy):
    """
    Discover types for numpy array
    
    """
    display(HTML('<h2>Discover Data Types</h2>'))
    
    display(HTML('<h4>Simple Data Types</h4>'))
    print(discover_type_heuristic(Xy))
    display(HTML('<h4>Statistical Data Types</h4>'))
    print(discover_type_bayesian(Xy))

################################

##### Duplicated Rows ######
def clean_duplicated_rows(dataframe):
    """
    Clean duplicated rows
    """
    display(HTML('<h2>Duplicated Rows</h2>'))
    
    print("Identifying Duplicated Rows ...")
    
    # Mark duplicated rows with True
    mark = dataframe.duplicated(keep=False)
    #print(mark)
    
    # Duplicated rows detected then set duplicated 'True' 
    duplicated = False
    for bool in mark:
        if bool:
            duplicated = True
    
    # Show duplicated rows if there are
    # Otherwise show no duplicated rows
    if duplicated:
        print("")
        display(HTML('<bold>Duplicated rows are detected.</bold>'))
        print("")
        print(dataframe[mark])
        print("")
        ans = input('Do you want to drop the duplicated rows? [y/n]')
        if ans == 'y':
            dataframe_no_duplicate = drop_duplicated_rows(dataframe)
            print("")
            print("Duplicated rows are dropped.")
            print("")
            return dataframe_no_duplicate
        else:
            print("")
            print("Duplicated rows are kept.")
            print("")
            return dataframe
    else:
        print("")
        display(HTML('<bold>No duplicated rows detected.</bold>'))
        print("")
        return dataframe

def drop_duplicated_rows(dataframe):
    """
    Drop duplicatd rows
    """
    # Drop the duplicates
    dataframe_no_duplicate = dataframe.drop_duplicates()
    #print(dataframe_no_duplicate)
    return dataframe_no_duplicate

############################

##### Inconsistent Column Names #####
def unify_name_consistency(names):
    """
    Unify inconsistent column names
    """
    display(HTML('<h2>Inconsitent Column Names</h2>'))
    print("")
    print("Column names")
    print("============")
    print(names)
    print("")
    
    if all(col[0].isupper() for col in names):
        print("Column names are consistent")
    elif all(col[0].islower() for col in names):
        print("Column names are consistent")
    else:
        display(HTML('<bold>Column names are not consistent</bold>'))
        ans = input("Do you want to unify the captalization? [y/n]")
        if ans == 'y':
            print('Select upper case or lower case [u/l]')
            ans = input()
            if ans == 'l':
                names = map(str.lower, names)
                print("Capitalizaiton unified.")
                print(names)
            else:
                names = map(str.upper, names)
                print("Capitalization unified.")
                print(names)
        else:
            pass
        
    return names
#####################################

##### Missing Values #####
def identify_missing(df=None):
    """
    Detect missing values
    
    ---
    input:
    
    df : The input DataFrame
    
    """
    
    display(HTML('<h4>Identify Missing Data ...</h4>'))
    na_values=['n/a', 'na', '--', '?']
    print('The default setting of missing characters is {}'.format(na_values))
    time.sleep(0.05)
    ans = input('Do you want to add extra character? [y/n]')
    if ans == 'y':
        char = input('Input the character to be identified as missing: ')
        na_values.append(char)
        print('New missing character added!')
        print(na_values)
    else:
        pass
    for value in na_values:
        df = df.replace(value, np.nan)
    
    # flag indicates whether any missing value is detected
    flag = df.isnull().values.any()
    if flag:
        print("")
        display(HTML('<bold>Missing values detected!</bold>'))
        print("")
        # number of missing in each feature
        print("Number of missing in each feature")
#         for col in df.columns:
#             print(df[col].isnull().sum())
        print(df.isnull().sum())
        print("")
        print("Records containing missing values:")
        display(df[df.isnull().any(axis=1)].head())
        print("")
#         # number of missing in each row
#         for i in df.isnull().sum(axis=1):
#             print(i)
        # identify missing mechanism
        identify_missing_mechanism(df)
        # visualize missing data
        visualize_missing(df)
    else:
        print("No missing value detected")
    return flag

def identify_missing_mechanism(df=None):
    """
    Try to guess missing mechanism of the dataset
    MCAR: Missing completely at ramdom
    MAR: Missing at random
    MNAR: Missing not at random (not available here, normally involes field expert)
    
    ---
    input:
    
    df: input DataFrame
    
    """
    # Pearson correlation coefficient between every 2 features
#     print("")
#     print("Missing correlation (Pearson correlation coefficient) between every 2 features")
#     display(df.isnull().corr())
    df2 = df.iloc[:, :-1].copy()
    missing_columns = df2.columns[df2.isnull().any(axis=0)] # columns containing missing values
    # relace nan as true, otherwise false for features containing missing values
    df2[df2.columns[df2.isnull().any(axis=0)]] = df2[df2.columns[df2.isnull().any(axis=0)]].isnull() 
    df2[missing_columns] = df2[missing_columns].astype(int) # replace true as 1, false as 0
    df_missing_corr = df2.corr()[missing_columns] # compute correlations between features containing missing values and other features
    print("Missing correlation between features containing missing values and other features")
    display(df_missing_corr)
    flag_mar = False
    # test if there is some correlation of a value being missed in feature and the value of any other of the features
    for col in df_missing_corr:
        list_high_corr =[]
        list_high_corr = list_high_corr + (df_missing_corr[col].index[df_missing_corr[col]>0.6].tolist())
        list_high_corr.remove(int(col))
#         print(list_high_corr)
        if list_high_corr:
            flag_mar = True
    if flag_mar:
        display(HTML('<bold>Missing mechanism is probably missing at random</bold>'))
    else:
        display(HTML('<bold>Missing mechanism is probably missing completely at random</bold>'))
#     tri_lower_no_diag = np.tril(df.isnull().corr(), k=-1)
#     # if any 2 features highly missing correlated
#     if (tri_lower_no_diag > 0.6).any() or (tri_lower_no_diag < -0.6).any():
#         display(HTML('<bold>Missing mechanism is highly possible to be missing at random</bold>'))
#     elif (tri_lower_no_diag > -0.2).all() and (tri_lower_no_diag < 0.2).all():
#         display(HTML('<bold>Missing mechanism is highly possible to be missing completely at random</bold>'))
#     else:
#         display(HTML('<bold>Missing mechanism is hard to guess</bold>'))

def visualize_missing(df=None):
    """
    Visualize dataset containing missing values
    
    ---
    input:
    
    df : The input DataFrame where values like '?', 'na' have already been replace by np.nan
    
    """
    print("")
    display(HTML('<h4>Visualize Missing Data ...</h4>'))
    print("")
    msno.matrix(df, figsize=(6, 4), fontsize=12)
    msno.bar(df, figsize=(6, 4), fontsize=12)
    msno.heatmap(df, figsize=(6, 4), fontsize=12)
    plt.show()

def missing_preprocess(features, df=None):
    """
    detect and drop empty rows
    detect features and instances with extreme large proportion of missing data
    """
    
    # number of missing in each row
#     print(df.isnull().sum(axis=1))

    # number of missing in each feature
#     print(df.isnull().sum())
    
    # number of instances
    num_instances = df.shape[0]
    # number of features
    num_features = df.shape[1]
    
    # detect empty rows
    if any(df.isnull().sum(axis=1) == num_features):
        print(df[df.isnull().sum(axis=1) == num_features])
        print("Above empty rows are detected and removed \n")
        df = df.dropna(how='all') # remove empty rows
    
    large_missing_cols=[] # list of columns with extreme large proportion of missing data
    for col in df.columns[:-1]: # exclude target class
        if df[col].isnull().sum() > 0.9 * num_instances:
            large_missing_cols.append(col)
    if large_missing_cols:
        print("Feature {} has extreme large proportion of missing data".format(large_missing_cols))
        ans = input('Do you want to delete the above features? [y/n]')
        if ans == 'y':
            df.drop(large_missing_cols,1, inplace=True)
        else:
            pass
    features_new = [features[i] for i in df.columns.values[:-1]]
    return df, features_new

def compute_imputation_score(Xy):
    """
    Compute score of the imputation by applying simple classifiers
    The following simple learners are evaluated:
    Naive Bayes Learner;
    Linear Discriminant Learner;
    One Nearest Neighbor Learner;
    Decision Node Learner;
    ---
    input:
    
    Xy: complete numpy array
    
    """
    X = Xy[:,:-1]
#     print(X.dtype)
    y = Xy[:,-1]
    y = y.astype('int')
#     print(y.dtype)
    
    scores =[]
    naive_bayes = GaussianNB()
    decision_node = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=1, random_state=0)
    linear_discriminant_analysis = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    one_nearest_neighbor =  KNeighborsClassifier(n_neighbors = 1)
    classifiers = [naive_bayes, decision_node, linear_discriminant_analysis, one_nearest_neighbor]
    for classifier in classifiers:
        # compute accuracy score for each simple classifier
        score = np.mean(cross_val_score(classifier, X, y, cv=5, scoring='accuracy', n_jobs=-1))
#         print("Score of {} is {}".format(classifier, score))
        scores.append(score)
        
    imputation_score = np.mean(scores)
    return imputation_score

def deal_mcar(df):
    """
    Deal with missing data with missing completely at random pattern
    
    ---
    input:
    
    df: input DataFrame
    """
    # number of instances
    num_instances = df.shape[0]
    
    # number of rows containing missing
    num_missing_instances = df.isnull().sum(axis=1).astype(bool).sum()
    
    # missing percentage
    missing_percentage = num_missing_instances / num_instances
    print("Missing percentage is {}".format(missing_percentage))
    
    if missing_percentage < 0.05:
        recommend = 'list deletion'
    else:
        Xy_incomplete = df.values
        # mean
        Xy_filled_mean = Imputer(missing_values=np.nan, strategy='mean').fit_transform(Xy_incomplete)
        score_mean = compute_imputation_score(Xy_filled_mean)
        print("Imputation score of mean is {}".format(score_mean))
        # mode
        Xy_filled_mode = Imputer(missing_values=np.nan, strategy='most_frequent').fit_transform(Xy_incomplete)
        score_mode = compute_imputation_score(Xy_filled_mode)
        print("Imputation score of mode is {}".format(score_mode))
        # knn
        with NoStdStreams():
            Xy_filled_knn = KNN().fit_transform(Xy_incomplete);
        score_knn = compute_imputation_score(Xy_filled_knn)
        print("Imputation score of knn is {}".format(score_knn))
        # matrix factorization
        with NoStdStreams():
            Xy_filled_mf = MatrixFactorization().fit_transform(Xy_incomplete);
        score_mf = compute_imputation_score(Xy_filled_mf)
        print("Imputation score of matrix factorization is {}".format(score_knn))
        # multiple imputation
        with NoStdStreams():
            Xy_filled_ii = IterativeImputer().fit_transform(Xy_incomplete)
        score_ii = compute_imputation_score(Xy_filled_ii)
        print("Imputation score of multiple imputation is {}".format(score_ii))
        
        score_dict = {'mean': score_mean, 'mode': score_mode, 'knn': score_knn,
                      'matrix factorization': score_mf, 'multiple imputation': score_ii}
        print("Imputation method with the highest socre is {}".format(max(score_dict, key=score_dict.get)))
        recommend = max(score_dict, key=score_dict.get)
    return recommend

def deal_mar(df):
    """
    Deal with missing data with missing at random pattern
    
    ---
    input:
    
    df: input DataFrame
    """
    
    Xy_incomplete = df.values

    # knn
    with NoStdStreams():
        Xy_filled_knn = KNN().fit_transform(Xy_incomplete);
    score_knn = compute_imputation_score(Xy_filled_knn)
    print("Imputation score of knn is {}".format(score_knn))
    # matrix factorization
    with NoStdStreams():
        Xy_filled_mf = MatrixFactorization().fit_transform(Xy_incomplete);
    score_mf = compute_imputation_score(Xy_filled_mf)
    print("Imputation score of matrix factorization is {}".format(score_knn))
    # multiple imputation
    with NoStdStreams():
        Xy_filled_ii = IterativeImputer().fit_transform(Xy_incomplete)
    score_ii = compute_imputation_score(Xy_filled_ii)
    print("Imputation score of multiple imputation is {}".format(score_ii))

    score_dict = {'knn': score_knn,
                  'matrix factorization': score_mf, 'multiple imputation': score_ii}
    print("Imputation method with the highest socre is {}".format(max(score_dict, key=score_dict.get)))
    recommend = max(score_dict, key=score_dict.get)
    return recommend

def deal_mnar(df):
    """
    Deal with missing data with missing at random pattern
    
    ---
    input:
    
    df: input DataFrame
    """
    recommend = 'multiple imputation'
    return recommend

def clean_missing(df,features):
    display(HTML('<h4>Clean Missing Data ...</h4>'))
#     Xy = np.concatenate((X,y.reshape((y.shape[0],1))), axis=1)
    df_preprocessed, features_new = missing_preprocess(df, features)
    print("")
    print("Choose the missing mechanism [a/b/c/d]:")
    print("a.MCAR b.MAR c.MNAR d.Skip")
    ans = input()
    if ans == 'a':
        recommend = deal_mcar(df_preprocessed)
    elif ans == 'b':
        recommend = deal_mar(df_preprocessed)
    elif ans == 'c':
        recommend = deal_mnar(df_preprocessed)
    else:
        print("Default MAR")
        recommend = deal_mar(df_preprocessed)
    print("")
    display(HTML('<bold>Recommended Approach!</bold>'))
    print("The recommended approach is {}".format(recommend))
    time.sleep(0.05)
    ans = input("Do you want to apply the recommended approach? [y/n]")
    print("")
    if ans == 'y':
        if recommend == 'mean':
            print("Applying mean imputation ...")
            Xy_filled = Imputer(missing_values=np.nan, strategy='mean').fit_transform(df_preprocessed.values)
            print("Missing values cleaned!")
        elif recommend == 'mode':
            print("Applying mode imputation ...")
            Xy_filled = Imputer(missing_values=np.nan, strategy='most_frequent').fit_transform(df_preprocessed.values)
            print("Missing values cleaned!")
        elif recommend == 'knn':
            print("Applying knn imputation ...")
            with NoStdStreams():
                Xy_filled = KNN().fit_transform(df_preprocessed.values);
            print("Missing values cleaned!")
        elif recommend == 'matrix factorization':
            print("Applying matrix factorization ...")
            with NoStdStreams():
                Xy_filled = MatrixFactorization().fit_transform(df_preprocessed.values);
            print("Missing values cleaned!")
        elif recommend == 'multiple imputation':
            print("Applying multiple imputation ...")
            with NoStdStreams():
                Xy_filled = IterativeImputer().fit_transform(df_preprocessed.values)
            print("Missing values cleaned!")
        else:
            print("Error: Approach not available!")
        
    else:
        print("")
        print("Choose the approach you want to apply [a/b/c/d/e/skip]:")
        print("a.Mean b.Mode c.K Nearest Neighbor d.Matrix Factorization e. Multiple Imputation")
        ans = input()
        print("")
        if ans == 'a':
            print("Applying mean imputation ...")
            Xy_filled = Imputer(missing_values=np.nan, strategy='mean').fit_transform(df_preprocessed.values)
        elif ans == 'b':
            print("Applying mode imputation ...")
            Xy_filled = Imputer(missing_values=np.nan, strategy='most_frequent').fit_transform(df_preprocessed.values)
        elif ans == 'c':
            print("Applying knn imputation ...")
            with NoStdStreams():
                Xy_filled = KNN().fit_transform(df_preprocessed.values);
        elif ans == 'd':
            print("Applying matrix factorization ...")
            with NoStdStreams():
                Xy_filled = MatrixFactorization().fit_transform(df_preprocessed.values);
        elif ans == 'e':
            print("Applying multiple imputation ...")
            with NoStdStreams():
                Xy_filled = IterativeImputer().fit_transform(df_preprocessed.values)
        else:
            print("Applying default method mean imputation ...")
            Xy_filled = Imputer(missing_values=np.nan, strategy='mean').fit_transform(df_preprocessed.values)
            
        print("Missing values cleaned!")
    return features_new, Xy_filled

def handle_missing(features, df):
    """
    Handle missing values
    """
    
    display(HTML('<h2>Missing values</h2>'))
    flag = identify_missing(df)
    if flag:
        features_new, Xy_filled = clean_missing(features, df)
    return features_new, Xy_filled
##########################

##### Outliers #####
def compute_clustering_metafeatures(X):
    X = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X)
    kmeans_model = KMeans(n_clusters=2, random_state=1).fit(X)
    labels = kmeans_model.labels_
    silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
    calinski_harabaz_score = metrics.calinski_harabaz_score(X, labels)
    davies_bouldin_score = metrics.davies_bouldin_score(X, labels)
    return silhouette_score, calinski_harabaz_score, davies_bouldin_score
def pipeline(X, y, estimator):
#     if scipy.sparse.issparse(X) == True: # Check if X is sparse array
#         X = X.toarray()
    start_time_pipeline = time.process_time()
    pipe = Pipeline([('Imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
                     ('classifiers', estimator)])
    
    # to prevent sklearn cross_val_score failure due to label imabalance
#     pred = cross_val_predict(pipe, X, y, cv=5)
#     score = roc_auc_score(y, pred)
    score = np.mean(cross_val_score(pipe, X, y, cv=10, scoring='accuracy', n_jobs=-1))
    time_pipeline = time.process_time() - start_time_pipeline
    return score, time_pipeline

def compute_metafeatures(X, y):
    """
    The following landmarking features are computed:
        Naive Bayes Learner;
        Linear Discriminant Learner;
        One Nearest Neighbor Learner;
        Decision Node Learner;
        Randomly Chosen Node Learner
    """
    
    # if regression
    if len(np.unique(y)) > 100 or len(np.unique(y)) > 0.1*y.shape[0]:
        print("regression")
        print("meta features cannot be extracted as the target is not categorical")
    # if classification
    else:
#         print("classification")
        metafeatures_clf = {}
        # compute clustering performance metafeatures
        metafeatures_clf['silhouette'], metafeatures_clf['calinski_harabaz'], metafeatures_clf['davies_bouldin'] = compute_clustering_metafeatures(X)
    
        # compute landmarking metafeatures
        metafeatures_clf['naive_bayes'], metafeatures_clf['naive_bayes_time'] = pipeline(X, y, GaussianNB()) 
        metafeatures_clf['linear_discriminant_analysis'], metafeatures_clf['linear_discriminant_analysis_time'] = pipeline(X, y, LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')) 
        metafeatures_clf['one_nearest_neighbor'], metafeatures_clf['one_nearest_neighbor_time'] = pipeline(X, y, KNeighborsClassifier(n_neighbors = 1)) 
        metafeatures_clf['decision_node'], metafeatures_clf['decision_node_time'] = pipeline(X, y, DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=1, random_state=0)) 
        metafeatures_clf['random_node'], metafeatures_clf['random_node_time'] = pipeline(X, y, DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=1, random_state=0))
        metafeatures = list(metafeatures_clf.values())
    
    return metafeatures

def predict_best_anomaly_algorithm(X, y):
    """
    Predict best anomaly detection algorithm
    0: isolation forest;
    1: lof
    2: one svm
    """
    # extract metafeatures of this dataset
    mf = np.array(compute_metafeatures(X, y))
    mf = mf.reshape(-1, mf.shape[0])
    mf.shape
    
    # load meta learner
    metalearner = joblib.load('metalearner_rf.pkl')
    best_anomaly_algorithm = metalearner.predict(mf)
    if best_anomaly_algorithm[0] == 0:
        print("The recommended approach is isolation forest.")
    elif best_anomaly_algorithm[0] == 1:
        print("The recommended approach is local outlier factor.")
    elif best_anomaly_algorithm[0] == 2:
        print("The recommended approach is one class support vector machine.")
    else:
        print("Error in prediction.")
    ans = input("Do you want to apply the recommended outlier detection approach? [y/n]")
    if ans == "y":
        best = best_anomaly_algorithm[0]
    else:
        print("Choose the approach you want to apply [a/b/c/d]: ")
        print("a.Isolation Forest b.Local Outlier Factor c.One Class SVM d.Skip")
        ans = input()
        if ans == 'b':
            best = 1
            print("Applying local outlier factor ...")
        elif ans == 'c':
            best = 2
            print("Applying one class svm ...")
        else:
            best = 0
            print("Applying isolation forest ...")
    return best

def highlight_outlier(data):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_outlier = ~(np.abs(data - data.mean()) < (3 * data.std()))
    return ['background-color: yellow' if v else '' for v in is_outlier]


def identify_outliers(df,algorithm=0, detailed=False):
    """
    Identify outliers in multi dimension for openml dataset where dataset has already been parsed
    """

#     df_exclude_target = df[df.columns.difference([target])] # exclude target from data
    df_exclude_target = df.iloc[:,:-1]
    df_numeric = df_exclude_target.select_dtypes(include=[np.number]) # keep only numeric type features
    total_length = len(df_numeric) # total length of the dataframe, used for computing contamination later
#     print(total_length)
    
    outliers_count = np.zeros(len(df_numeric.columns)) # number of outliers of each feature
    dict_outliers = {}
    flag = False
    df_union = pd.DataFrame()
    for i, col in enumerate(df_numeric.columns):
#         if(df_numeric[col].dtype in [np.number]): # bug! to be figured out 

        # first detect outliers in each column
        # keep only the ones that are out of +3 to -3 standard deviations in the column 'Data'.
        dict_outliers[col] = df_numeric[~(np.abs(df_numeric[col]-df_numeric[col].mean())<(3*df_numeric[col].std()))] # ~ means the other way around
        # combine all the rows containing outliers in one feature
        df_union = df_union.combine_first(dict_outliers[col])
#             print(dict_outliers[col])
        if len(dict_outliers[col]) != 0:
            outliers_count[i] = len(dict_outliers[col])
            flag = True
            if detailed:                    
                print("There are {} outliers in variable {}".format(len(dict_outliers[col]), col))
                print(dict_outliers[col][col])
                print("")
        else:
            if detailed:
                print("No outliers are detected in variable {}".format(col))
                print("")
    
    # boxplot: show outliers in each feature
    # feature scaling
    ss = StandardScaler()
    df_scaled = ss.fit_transform(df_numeric)
    df_scaled = pd.DataFrame(df_scaled, columns=df_numeric.columns)
    df_scaled.head()
    # draw box plot for numeric variables
    fig = plt.figure(figsize=(6, 4))
    fig.subplots_adjust(top=0.93, wspace=0)
    ax = sns.boxplot(data=df_scaled, palette="Set1")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.show()
    
    # Two options to estimate the propotion of outliers
    # One is to take the number of outliers in the feature containing most outliers
    # The other is to take the length of the union of rows containing outliers in any feature
#     print(outliers_count)
#     print(df_union)
#     max_outliers = max(outliers_count)
    max_outliers = len(df_union) 
#     print("max outliers number is {}".format(max_outliers))
#     if flag:
#         print("Outliers detected")
#         print("")
#     else:
#         print("No outliers detected")
#         print("")
#     plt.show()
    contamination = max_outliers / total_length
    X = np.asarray(df_numeric)
    # to be extended later, for now isolation forest
    if algorithm == 2:
        clf = svm.OneClassSVM(nu=0.95 * contamination + 0.05)
        clf.fit(X)
        y_pred = clf.predict(X)
    elif algorithm == 1:
        clf = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        y_pred = clf.fit_predict(X)
    else:
        clf = IsolationForest(contamination = contamination)
        clf.fit(X)
        y_pred = clf.predict(X)
#     print(y_pred)
    outlier_index, = np.where(y_pred == -1)
    df_outliers = df_numeric.iloc[outlier_index.tolist()]
#     print(outlier_index)
    if algorithm == 1:
        anomaly_score = y_pred # decision function only available for novelty detection for in lof
    else:
        anomaly_score = clf.decision_function(X) # p_pred: The anomaly score of the input samples. The lower, The more abnormal.
    anomaly_score = pd.DataFrame(anomaly_score, columns=['anomaly_score'])
    df_with_anomaly_score = pd.concat([df, anomaly_score], axis=1)
    df_with_anomaly_score
    df_sorted = df_with_anomaly_score.sort_values(by='anomaly_score')
    cm = sns.diverging_palette(10, 220, sep=80, n=7, as_cmap=True)
    df_styled = df_sorted.style.background_gradient(cmap=cm, subset=['anomaly_score']).apply(highlight_outlier, subset=df_sorted.columns[:-1])
#     print("*********************************************")
#     print("Outliers detected in multi dimensional space:")
#     print("*********************************************")

#     print(df_numeric.iloc[outlier_index.tolist()])
    df_pred = pd.DataFrame(y_pred, columns=['pred'])
    display(df_styled)
    return df_scaled, df_styled, df_outliers, df_pred, outliers_count

def visualize_outliers_scatter(df, df_pred):
    """
    Select out the two features most likely to have outliers
    and show them in a scatter plot
    """
    df_exclude_target = df.iloc[:,:-1]
    df_numeric = df_exclude_target.select_dtypes(include=[np.number]) # keep only numeric type features
    total_length = len(df_numeric) # total length of the dataframe, used for computing contamination later
 

    outliers_count = np.zeros(len(df_numeric.columns)) # number of outliers of each feature
    dict_outliers = {}
    flag = False
    df_union = pd.DataFrame()
    for i, col in enumerate(df_numeric.columns):
    #         if(df_numeric[col].dtype in [np.number]): # bug! to be figured out 
        if(df_numeric[col].dtype == np.float64 or df_numeric[col].dtype == np.int64): 
            # first detect outliers in each column
            # keep only the ones that are out of +3 to -3 standard deviations in the column 'Data'.
            dict_outliers[col] = df_numeric[~(np.abs(df_numeric[col]-df_numeric[col].mean())<(3*df_numeric[col].std()))] # ~ means the other way around
            # combine all the rows containing outliers in one feature
            df_union = df_union.combine_first(dict_outliers[col])
    #             print(dict_outliers[col])
            if len(dict_outliers[col]) != 0:
                outliers_count[i] = len(dict_outliers[col])

#     print(outliers_count)

    # get the index of the two features most likely to have outliers
    index_max2 = outliers_count.argsort()[-2:][::-1] # [::-1] reverse

    df_max2=df_numeric.iloc[:,index_max2]
#     print(df_max2.head())

    # draw scatter plot
    df_max2_anomaly=df_max2.loc[df_pred['pred'] == -1]
    df_max2_normal=df_max2.loc[df_pred['pred'] == 1]
    graph = sns.jointplot(x = df_max2[df_max2.columns.values[0]], 
                          y = df_max2[df_max2.columns.values[1]], 
                          color = 'skyblue', 
                          size = 4,
                          space = 0,
                          s = 5)
    graph.x = df_max2_anomaly[df_max2.columns.values[0]]
    graph.y = df_max2_anomaly[df_max2.columns.values[1]]
    graph.plot_joint(plt.scatter,c='firebrick', s=5)
    plt.show()

def visualize_outliers_parallel_coordinates(df_scaled, df_pred):
    from pandas.plotting import parallel_coordinates
    df_final = pd.concat([df_scaled, df_pred], axis=1)
#     print(df_final.head())
    fig = plt.figure(figsize=(6, 4))
    title = fig.suptitle("Parallel Coordinates", fontsize=18)
    fig.subplots_adjust(top=0.93, wspace=0)
    pc = parallel_coordinates(df_final, 'pred', color=('skyblue', 'firebrick'), sort_labels=True)
    pc.set_xticklabels(pc.get_xticklabels(), rotation=40, ha="right")
    _=plt.legend(['Normal', 'Anomaly'])
    plt.show()

def drop_outliers(df, df_outliers):
    df_no_outliers = df.drop(df_outliers.index.values)
    return df_no_outliers

def handle_outlier(Xy):
    display(HTML('<h2>Outliers</h2>'))
    display(HTML('<h4>Recommend Algorithm ... </h4>'))
    X = Xy[:,:-1]
    y = Xy[:,-1]
    best = predict_best_anomaly_algorithm(X, y)
    df = pd.DataFrame(Xy)
    display(HTML('<h4>Visualize Outliers ... </h4>'))
    df_scaled, df_styled, df_outliers, df_pred, outliers_count = identify_outliers(df, algorithm=best)
#     print("Count for normal data and outliers")
#     print(df_pred['pred'].value_counts())
    visualize_outliers_scatter(df, df_pred)
    visualize_outliers_parallel_coordinates(df_scaled, df_pred)
    print("")
    display(HTML('<h4>Drop Outliers ... </h4>'))
    time.sleep(0.05)
    ans = input("Do you want to drop outliers? [y/n]")
    if ans == 'y':
        print("Outliers are dropped.")
        df_no_outliers = drop_outliers(df, df_outliers)
        return df_no_outliers
    else:
        print("Outliers are kept.")
        return df

####################

##### Main #####
def autoclean(openml_id):
    data = oml.datasets.get_dataset(openml_id)
    X, y, features = data.get_data(target=data.default_target_attribute, return_attribute_names=True)
    Xy = data.get_data()
    show_important_features(X, y, data.name, features)
    show_statistical_info(Xy)
    discover_types(Xy)
    df_no_duplicate = clean_duplicated_rows(pd.DataFrame(Xy))
    features = unify_name_consistency(features)
    features_new, Xy_filled = handle_missing(features, df_no_duplicate)
    df_handled = handle_outlier(Xy_filled)
    return df_handled