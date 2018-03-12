"""
Tests for automatic data cleaning
"""
import datacleaner as dc
import pandas as pd
import numpy as np

def main_test():
    """
    Tests for read file, show data, report data and visualize data
    Always run first
    """
    # General test
    df = dc.get_data('tips.csv')
    dc.show_data(df, False)
    #dc.visualize_data(df)
    dc.diagnose_data(df)
    dc.report_data(df, False)

def typeconvert_test():
    """
    Test for converting data type
    """
    # The type for 'sex' in 'tips.csv'
    # should be category instead of object
    df1 = dc.get_data('tips.csv')
    dc.report_data(df1, False)
    df1 = dc.convert_type(df1, 'sex', 'category')
    dc.report_data(df1, False)

def duplicated_test():
    """
    Test for dropping duplicated rows
    """
    # Generate a dataframe with duplicated rows   
    d = {'col1': [1, 2, 2, 5], 'col2': [3, 4, 4, 7],
         'col3': [3, 9, 9, 8], 'col4': [2, 13, 13, 28]}
    df2 = dc.pd.DataFrame(data=d)
    print("Original data")
    print("=============")
    print(df2)
    print("")
    dc.identify_duplicated_rows(df2)
    df2 = dc.clean_duplicated_rows(df2)
    print("After dropping duplicated rows")
    print("==============================")
    print(df2)

def outlier_test():
    """
    Test for detecting outliers
    """
    # Generate a random dataframe
    df3 = pd.DataFrame({'Data':np.random.normal(size=200)})
    # Create a few outliers (3 of them, at index locations 10, 55, 80)
    #df3.iloc[[10, 55, 80]] = 40.
    print(df3)
    dc.identify_outliers(df3)

def drop_missing_test():
    """
    Test for dropping missing values
    """
    df = get_data('airquality.csv')
    df.info()
    identify_missing_values(df)
    df2 = drop_missing_values(df)
    df2.info()

def filling_missing_test():
    """
    Test for filling missing values
    """
    df = dc.get_data('airquality.csv')
    print("Orignial data")
    print(df.head())
    dc.identify_missing_values(df)
    df = dc.fill_missing_values(df, "missing", "Ozone")
    print(df.head())
    
def consistency_test():
    """
    Test for detecting inconsitent column names
    """
    df = dc.get_data('airquality.csv')
    df = df.rename(index=str, columns={"Ozone": "ozone"})
    print(df)
    dc.identify_name_consistency(df)
    
##typeconvert_test()
##duplicated_test()
##outlier_test()
##consistency_test()
##drop_missing_test()
filling_missing_test()
