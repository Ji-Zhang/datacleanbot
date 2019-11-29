[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# datacleanbot
Automated Data Cleaning Tool.
The main goal is to develop a Python tool ``datacleanbot`` such that:
    Given a random parsed raw dataset representing a supervised learning problem, the Python tool is capable of automatically identifying the potential issues and reporting the results and recommendations to the end-user in an effective way.

## Install

```sh
$ pip install datacleanbot
```

## QuickStart

### Install OpenML (version 0.9.0):

OpenML is used to easily import datasets and share models and experiments.

```sh
$ pip install openml
```

For Windows, you need to have C++ Compiler installed.

### Acquire data from OpenML:

    >>> import openml as oml
    >>> data = oml.datasets.get_dataset(id) # id: openml dataset id
    >>> X, y, categorical_indicator, features = data.get_data(target=data.default_target_attribute, dataset_format='array')
    >>> Xy = np.concatenate((X,y.reshape((y.shape[0],1))), axis=1)

### Autoclean data with datacleanbot:

    >>> import datacleanbot.dataclean as dc
    >>> Xy = dc.autoclean(Xy, data.name, features)


## Description

``datacleanbot`` is equipped with the following capabilities:
* Present an overview report of the given dataset
    * The most important features
    * Statistical information (e.g., mean, max, min)
    * Data types of features
* Clean common data problems in the raw dataset
    * Duplicated records
    * Inconsistent column names
    * **Missing values**
    * **Outliers**

The two aspects ``datacleanbot`` meaningfully automates are marked in bold.

## User's Guide

The user's guide can be found at [datacleanbot](https://datacleanbot.readthedocs.io/en/latest/).
