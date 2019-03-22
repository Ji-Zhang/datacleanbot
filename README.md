# datacleanbot: Automated Data Cleaning Tool
Research project about automated data cleaning. 
The main goal is to develop a Python tool ``datacleanbot`` such that:
    Given a random parsed raw dataset representing a supervised learning problem, the Python tool is capable of automatically identifying the potential issues and reporting the results and recommendations to the end-user in an effective way.
    
``datacleanbot`` is equipped with the following capabilities:
* Present an overview report of the given dataset
    * The most important features
    * Statistical information (e.g., mean, max, min)
    * **Data types of features**
* Clean common data problems in the raw dataset
    * Duplicated records
    * Inconsistent column names
    * **Missing values**
    * **Outliers**

The three aspects ``datacleanbot`` meaningfully automates are marked in bold.

The user's guide can be found at [datacleanbot](https://automatic-data-cleaning.readthedocs.io/en/latest/index.html).
