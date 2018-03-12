"""
Automatic data cleaning
"""

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections

def get_data(file):
    """
    load dataset into a DataFrame
    """
    
    # Read the file into a DataFrame: df
    dataframe = pd.read_csv(file)
    return dataframe

def show_data(dataframe, complete=False):
    """
    Print the data frame
    """
    if complete:
        # Print complete df
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(dataframe)
    else:
        # Print part of df
        print(dataframe)    

def report_data(dataframe, detailed=False):
    """
    Show a concise summary
    """
    variable_type = {}
    unique_count = {}
    missing_observation = {}
    for col in dataframe.columns:
        # Type of each variable
        variable_type[col] = dataframe[col].dtypes
        # Unique count of each variable
        unique_count[col] = dataframe[col].nunique()
        # Missing pecentage
        missing_observation[col] = (len(dataframe.index)-dataframe[col].count()) / len(dataframe.index)
        #print("# uniques values of {:<10} is {:>4}".format(col, data[col].nunique()))
        #print(dataframe[col].value_counts(dropna=False).head())   
    #print(variable_type)  
    #print(unique_count)
    #print(missing_observation)
    
    # Merge multiple dictionaries with the same key
    # A general solution that will handle an arbitrary amount of dictionaries,
    # with cases when keys are in only some of the dictionaries
    dict_summary = collections.defaultdict(list)
    for dict in (unique_count, variable_type, missing_observation):
        for key, value in dict.items():
            dict_summary[key].append(value)
    #print(dict_summary)
    
    # dictionary to tabular data structure
    df = pd.DataFrame(dict_summary,
                      index=["# unique values", "Variable type", "Missing_observation"],
                      columns = dataframe.columns)
    # print summary table
    print("*************")
    print("Summary Table")
    print("*************")
    # tranpose table
    print(df.T)
    print("")
    
    
    # if boolean tailed = true, show more details
    if detailed:
        print("Data Describing")
        print("====================")
        print(dataframe.describe())
        print("")
        print("Additional Information")
        print("================")
        dataframe.info()
        print("")
        print("Unique values for each variable")
        print("===============================")
        for col in dataframe.columns:
            print("# uniques values of {:<10} is {:>4}".format(col, dataframe[col].nunique()))
            print(dataframe[col].value_counts(dropna=False))
            print("")
            
def visualize_data(df):
    """
    Visualize the distribution of each variable
    Histograms for numeric data
    Bar charts for categorical data
    """
    # Show distribution by histograms and bar charts
    for col in df.columns:
        # plot histograms for numeric data
        if(df[col].dtype == np.float64 or df[col].dtype == np.int64):
            plt.figure()
            plt.title("{}".format(col))
            df[col].plot(kind='hist')
            #plt.hist(df[col].dropna())
        # plot bar charts for categorical data
        else:
            #df[col].plot('hist')
            #plt.show()
            #print(df[col].value_counts(dropna=False))
            plt.figure()
            plt.title("{}".format(col))
            plt.ylabel("Count")
            df[col].value_counts(dropna=False).plot(kind='bar')
    plt.show()

def convert_type(dataframe, col=None, type=None):
    """
    Converting data type of a specific variable
    """
    dataframe[col] = dataframe[col].astype(type)
    return dataframe

def identify_missing_values(dataframe, detailed=False):
    missing_count = {}
    flag = False
    for col in dataframe.columns:
        # Missing count
        missing_count[col] = len(dataframe.index)-dataframe[col].count()
        if missing_count[col] != 0:
            flag = True
            if detailed:
                print("There are {} missing values for variable {}".format(missing_count[col],col))
        else:
            if detailed:
                print("No missing values detected for variable {}".format(col))
    if flag:
        print("")
        print("Missing value detected")
    else:
        print("")
        print("No missing value detected")
        
def drop_missing_values(dataframe):
    """
    Drop rows containing missing values
    """
    dataframe_missing_dropped = dataframe.dropna()
    return dataframe_missing_dropped

def fill_missing_values(dataframe, fillingvalue = None, col = None):
    """
    Fill missing values with specific value
    If no value is specified,
    fill numeric type varible with mean();
    fill category type varible with 'missing'
    """
    if fillingvalue != None:
        if col != None:
            dataframe[col] = dataframe[col].fillna(fillingvalue)
        else:
            dataframe = dataframe.fillna(fillingvalue)
    else:
        for col in dataframe.columns:
            if(dataframe[col].dtype == np.float64 or dataframe[col].dtype == np.int64):
                mean_value=dataframe[col].mean()
                dataframe[col] = dataframe[col].fillna(mean_value)
            else:
                dataframe[col] = dataframe[col].fillna('missing')
    print("")
    print("Missing values filled")
    #print(dataframe)
    return dataframe            
            
    

def identify_outliers(df, variable=None, groupby=None, detailed=False):
    """
    Indentify outliers by show box plot
    """
    
    # Draw box plot for numeric variable
    # Outliers can be shown
    df.boxplot(column=variable, by=groupby)
    plt.draw()
    
    df_outliers = {}
    flag = False
    for col in df.columns:
        if(df[col].dtype == np.float64 or df[col].dtype == np.int64):
            #keep only the ones that are out of +3 to -3 standard deviations in the column 'Data'.
            df_outliers[col] = df[~(np.abs(df[col]-df[col].mean())<(3*df[col].std()))]
            if len(df_outliers[col]) != 0:
                flag = True
                if detailed:                    
                    print("There are {} outliers in variable {}".format(len(df_outliers[col]), col))
                    print(df_outliers[col])
                    print("")
                    #print(len(df_outliers))
            else:
                if detailed:
                    print("No outliers are detected in variable {}".format(col))
                    print("")
    
    if flag:
        print("Outliers detected")
        print("")
    else:
        print("No outliers detected")
        print("")
    plt.show()

def identify_duplicated_rows(dataframe):
    """
    Identify duplicated rows
    """
    
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
        print("Duplicated rows detected")
        print("========================")
        print(dataframe[mark])
        print("")
    else:
        print("No duplicated rows")
        print("")

def clean_duplicated_rows(dataframe):
    """
    Drop with duplicatd rows
    """
    # Drop the duplicates
    dataframe_no_duplicate = dataframe.drop_duplicates()
    #print(dataframe_no_duplicate)
    return dataframe_no_duplicate

def identify_name_consistency(dataframe):
    """
    Identify if column names are consistent
    """
    print("")
    print("Column names")
    print("============")
    print(dataframe.columns)
    print("")
    
    if all(col[0].isupper() for col in dataframe.columns):
        print("Column names consistent")
    elif all(col[0].islower() for col in dataframe.columns):
        print("Column names consistent")
    else:
        print("Column names not consistent")

def diagnose_data(dataframe):
    """
    Deal with common data problems
    like missing data, duplicate rows and so on
    """
    print(dataframe)
    
    # Identify inconsistent column names
    print("")
    print("***********************************")
    print("Detecting inconsistent column names")
    print("***********************************")
    print("")
    identify_name_consistency(dataframe)
    
    # Identify the duplicates
    print("")
    print("*************************")
    print("Detecting duplicated rows")
    print("*************************")
    print("")
    identify_duplicated_rows(dataframe)

    # Identify potential outliers
    print("")
    print("******************")
    print("Detecting outliers")
    print("******************")
    print("")
    identify_outliers(dataframe,detailed=True)
    
    # Identify missing values
    print("")
    print("************************")
    print("Detecting missing values")
    print("************************")
    print("")
    identify_missing_values(dataframe, detailed=True)
    
df = get_data('tips.csv')
##show_data(df, complete = False)
##report_data(df, detailed = True)
##visualize_data(df)
##diagnose_data(df)




