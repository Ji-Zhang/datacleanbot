Usage
=====

Acquire Data
------------
The first step is to acquire data from `OpenML <https://www.openml.org/>`_.

.. code-block:: python
   
   import openml as oml
   import datacleanbot.dataclean as dc
   import numpy as np

   data = oml.datasets.get_dataset(id) # id: openml dataset id
   X, y, categorical_indicator, features = data.get_data(target=data.default_target_attribute, dataset_format='array')
   Xy = np.concatenate((X,y.reshape((y.shape[0],1))), axis=1)




Show Impotant Features
----------------------
``datacleanbot`` computes the most important features of
the given dataset using random forest and present the
15 most useful features to the user.

.. code-block:: python

   dc.show_important_features(X, y, data.name, features)

Unify Column Names
------------------

Inconsistent capitalization of column names can be detected and
reported to the user. Users can decide whether to unify them or 
not. The capitalization can be unified to either upper case or
lower case.

.. code-block:: python

   dc.unify_name_consistency(features)

Show Statistical Inforamtion
----------------------------
``datacleanbot`` can present the statistical information to
help users gain a better understanding of the data 
distribution.

.. code-block:: python

   dc.show_statistical_info(Xy)

Discover Data Types
-------------------

``datacleanbot`` can discover feature data types.
Basic data types discovered are 'datetime', 'float', 'integer',
'bool' and 'string'.
``datacleanbot`` also can discover statistical data types (real, positive real, 
categorical and count) using `Bayesian Model abda <https://arxiv.org/abs/1807.09306/>`_.

.. code-block:: python

   dc.discover_types(Xy)

Clean Duplicated Rows
---------------------

``datacleanbot`` detects the duplicated records and reports them to users.

.. code-block:: python

   dc.clean_duplicated_rows(Xy)

Handle Missing Values
---------------------

``datacleanbot`` identifies characters 'n/a', 'na', '--' and '?' as missing values.
Users can add extra characters to be considered as missing. After the missing
values being detected, ``datacleanbot`` will present the missing values in effective
visualizations to help users identify the missing mechanism. Afterwards, ``datacleanbot``
recommends the appropriate approach to clean missing values according the missing
mechanism.

.. code-block:: python

   features, Xy = dc.handle_missing(features, Xy)

Outlier Detection
-----------------

A meta-learner is trained beforehand to recommend the outlier detection algorithm
according to the meta features og the given dataset. Users can apply the
recommended algorithm or any other available algorithm to detect outliers.
After the detection, outliers will be present to users in effective visualizations
and users can choose to drop them or not.

.. code-block:: python

   Xy = dc.handle_outlier(features, Xy)