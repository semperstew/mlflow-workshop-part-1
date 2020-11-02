# Databricks notebook source
# MAGIC %md #### Problem - Tutorial 2: Classification
# 
# Given a set of features or attributes of a bank note, can we predict whether it's authentic or fake?
# Four attributes contribute to this classification:
# 1. variance or transformed image
# 2. skeweness
# 3. entropy
# 4. curtosis
# 
# <table>
#   <tr><td>
#     <img src="https://raw.githubusercontent.com/dmatrix/tutorials/master/mlflow/images/bank_note.png"
#          alt="Bank Note " width="600">
#   </td></tr>
#   <tr><td>
#     <img src="https://databricks.com/wp-content/uploads/2020/04/matrix_morpheus.png"
#          alt="Bank Note " width="600">
#   </td></tr>
# </table>

# COMMAND ----------

# MAGIC %md #### Solution:
# 
# We are going to use Random Forest Classification to make the prediction, and measure on the accuracy.
# The closer to 1.0 is the accuracy the better is our confidence in its prediction.
# 
# This example is borrowed from these source, modified and modularized for this tutorial [source-1](https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/)
# and data [source-2](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)
# 
# Aim of this Lab:
# 
# 1. Understand MLflow Tracking API
# 2. How to use the MLflow Tracking API
# 3. Use the MLflow API to experiment few Runs
# 4. Interpret and observer runs via the MLflow UI
# 
# Some resources:
# * [MLflow Docs](https://mlflow.org/docs/latest/python_api/mlflow.html)
# * [All about confufsion matrix](https://devopedia.org/confusion-matrix)
# * [More on classification matrix](https://joshlawman.com/metrics-classification-report-breakdown-precision-recall-f1/)
# * [How to plot AUC and ROC](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)

# COMMAND ----------

# MAGIC %md Define all the classes and bring them into scope

# COMMAND ----------

# MAGIC %run ./setup/class_setup

# COMMAND ----------

# load and print dataset
dataset = Utils.load_data("https://github.com/dmatrix/tutorials/raw/master/mlflow/labs/data/bill_authentication.csv")
dataset.head(5)

# COMMAND ----------

# iterate over several runs with different parameters
# TODO in the Lab (change these parameters, n_estimators and random_state
# with each iteration.
# Does that change the metrics and accuracy?
# start with n=10, step by 10 up to X <=40
for n in range(10, 40, 10):
  params = {"n_estimators": n, "random_state": 42}
  rfr = RFCModel.new_instance(params)
  (experimentID, runID) = rfr.mlflow_run(dataset)
  print("MLflow Run completed with run_id {} and experiment_id {}".format(runID, experimentID))
  print("-" * 100)

# COMMAND ----------

# MAGIC %md #### Lab/Homework for Some Experimental run
# 1. Consult [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) documentation
#   * Change or add parameters, such as depth of the tree or random_state: 42 etc.
# 2. Change or alter the range of runs and increments of n_estimators
# 3. Check in MLfow UI if the metrics are affected
# 4. Try [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instead of [RandomForestClassifer](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and examine if there is a difference in the metrics
# 5. Is the ROC for Randorm Forest better than LogisticRegression
