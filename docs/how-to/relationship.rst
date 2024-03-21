.. _how_to_relationship:

=======================
How-to Guide: Relationships
=======================

.. note::
    For now, numeric feature vs numeric target is studied as a linear relationship, but it leaves room for improvement.

Overall summary
---------------

The `cc_tk.relationship` submodule provides functions that allow you to study the relationship between variables.
It is particularly useful for feature selection when you have a lot of variables and you want to understand which ones are the most statistically significant to discriminate the target variable.
Target variable can be either numeric or categorical.


Let's consider the following example:

.. code-block:: python

    import cc_tk.relationship import RelationshipSummary
    from sklearn.datasets import load_iris

    # Create a dataframe
    X, y = load_iris(return_X_y=True, as_frame=True)

    # Artificially create a categorical variable
    X["sepal_length_cat"] = X["sepal length (cm)"] > 5.5

    # Study the relationship between X and y
    rs = RelationshipSummary(X, y.astype(object))

    rs.numeric_significance

This will show the relationship between numeric features of X and y and assign them a confidence score based on the statistical tests.
You can also use the `categorical_significance` attribute to study the relationship between categorical features of X and y.


Single variable relationship
----------------------------

You may also want to use directly the underlying functions to study the relationship between a single variable and the target variable.

.. warning::

    Be careful as when you use these functions you should be aware of the type of both feature variable and target variable.

.. code-block:: python

    import cc_tk.relationship import (
        significance_numeric_categorical, significance_categorical_categorical
    )
    from sklearn.datasets import load_iris

    # Create a dataframe
    X, y = load_iris(return_X_y=True, as_frame=True)

    # Artificially create a categorical variable
    X["sepal_length_cat"] = X["sepal length (cm)"] > 5.5

    # Study the relationship specific features and y
    significance_sepal_length_num = significance_numeric_categorical(X["sepal length (cm)"], y)
    significance_sepal_length_cat = significance_categorical_categorical(X["sepal_length_cat"], y)


.. admonition:: Future work

    I am planning to add more features to the `cc_tk.relationship` submodule.
    Already planned features are:
    - a scikit-learn transformer that will allow you to select the most significant features based on the relationship with the target variable
    - a parametrization of significance tests to allow the user to choose the most appropriate test for their data
