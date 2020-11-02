# Databricks notebook source
# MAGIC %md #### Problem - Tutorial 3: Regression
# 
# <table>
#   <tr><td>
#     <img src="https://brokeassstuart.com/wp-content/pictsnShit/2019/07/inside-airbnb-1263x560.jpg"
#          alt="Bank Note " width="600">
#   </td></tr>
#   <tr><td></td></tr>
#   <tr><td>
#   <img src="https://miro.medium.com/max/593/1*pfmeGgGM5sxmLBQ5IQfQew.png"
#          alt="Matrix" width="600">
#   <tr><td></td></tr>
#   <tr><td>Can we predict AirBnB prices in SF ...</td></tr>
# </table>
# 
# source: Databricks Learning Academy MLflow Course
# 
# Refactored code to modularize it
# 
# While iterating or build models, data scientists will often create a base line model to see how the model performs.
# And then iterate with experiments, changing or altering parameters to ascertain how the new parameters or
# hyper-parameters move the metrics closer to their confidence level.
# 
# This is our base line model using RandomForestRegressor model to predict AirBnb house prices in SF.
# Given 22 features, can we predict what the next house price will be?
# 
# We will compute standard evalution metrics and log them.
# 
# Aim of this module is:
# 
# 1. Introduce tracking ML experiments in MLflow
# 2. Log a base experiment and explore the results in the UI
# 3. Record parameters, metrics, and a model
# 
# Some Resources:
# * https://mlflow.org/docs/latest/python_api/mlflow.html
# * https://www.saedsayad.com/decision_tree_reg.htm
# * https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
# * https://towardsdatascience.com/understanding-random-forest-58381e0602d2
# * https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# * https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e
# * https://seaborn.pydata.org/tutorial/regression.html

# COMMAND ----------

# MAGIC %run ./setup/class_setup

# COMMAND ----------

# load the data
dataset = Utils.load_data("https://github.com/dmatrix/tutorials/raw/master/mlflow/labs/data/airbnb-cleaned-mlflow.csv")
dataset.head()

# COMMAND ----------

dataset.describe()

# COMMAND ----------

# To try different experiment runs, each with its own instance of model with the supplied parameters, 
# add more parameters to this dictionary list to experiment different parameters and their
# effects on evaluation metrics.
params_list = [ {"n_estimators": 75,"max_depth": 6, "random_state": 42}]
# run these experiments, each with its own instance of model with the supplied parameters.
for params in params_list:
  rfr = RFFExperimentModel.new_instance(params)  
  experiment = "Experiment with {} trees".format(params['n_estimators'])
  (experimentID, runID) = rfr.mlflow_run(dataset, r_name=experiment)
  print("MLflow Run completed with run_id {} and experiment_id {}".format(runID, experimentID))
  print("-" * 100)

# COMMAND ----------

# MAGIC %md #### Lab/Homework for Some Experimental runs
# 
#  1. Consult [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) documentation to see what hyperparameters you can specify
#   * Change or add parameters, such as depth of the tree
#  2. Change or alter the range of runs and increments of n_estimators
#  3. Use [scikit-learn cross validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html) to see any difference in metrics

# COMMAND ----------

