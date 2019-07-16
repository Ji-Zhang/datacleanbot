---
title: 'datacleanbot: automated, data-driven tool to help users clean data effectively'
tags:
  - data clean
  - Python
  - machine learning
  - data type discovery
  - missing value detection
  - outlier detection
authors:
  - name: Ji Zhang
    orcid: 0000-0002-4913-518X
    affiliation: 1
affiliations:
 - name: Eindhoven University of Technology
   index: 1
date: 16 July 2019
bibliography: paper.bib
---

# Summary

Data in real life almost never come in a clean way, and poor data quality may severely affect the effectiveness of learning algorithms [@Sessions_theeffects]. Consequently, raw data need to be cleaned before being able to proceed with training or running machine learning models.

``datacleanbot`` is a Python package which can offer automated, data-driven support to help users clean data effectively and smoothly. Given a random parsed raw dataset representing a supervised learning problem, ``datacleanbot`` is capable of automatically identifying the potential issues and reporting the results and recommendations to the end-user in an effective way. To be noticed, ``datacleanbot`` is aimed for supervised learning tasks and data need to be parsed as numeric format beforehand.

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

**Data types of features**: ``datacleanbot`` detects both basic data types (bool, date, float, integer and string) and statistical data types (real-valued, positive real-valued, count and categorical). ``datacleanbot`` detects basic data types by applying some logical rules. For statistical data types, ``datacleanbot`` utlizes the Bayesian model abda [@vergari2018automatic].

**Missing values**: ``datacleanbot`` identifies characters 'n/a', 'na', '--', '?' as missing. Users can add extra characters to be identified as missing. After the missing values being detected, ``datacleanbot`` presents the missing data in effective visualizations with the help of missingno [@bilogur2018missingno]. Afterwards, ``datacleanbot`` recommends the appropriate approach to delete or impute missing values according to the missing mechanism of the given dataset.

**Outliers**: A meta-learner is trained beforehand to predict the optimal outlier detection algorithm for the given dataset. Then outliers are reported to users in various visualizations.
Users can choose whether or not to drop the outliers.

``datacleanbot`` has a strong connection to OpenML[@vanschoren2014openml], a platform where people can easily share data, experiments and machine learning models. Users can easily acquire data from OpenML and clean these data with the assistance of ``datacleanbot``. 



# Acknowledgements

Many thanks to Dr. Joaquin Vanschoren for his dedication and guidance throughout this project in my master study.

# References

