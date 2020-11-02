# Databricks notebook source
# MAGIC %md #### Problem Tutorial 1: Regression Model
# 
# We want to predict the gas consumption (in millions of gallons/year) in 48 of the US states
# based on some key features. 
# 
# These features are 
#  * petrol tax (in cents); 
#  * per capital income (in US dollars);
#  * paved highway (in miles); and
#  * population of people with driving licences
# 
# <table>
#   <tr><td>
#     <img src="https://informedinfrastructure.com/wp-content/uploads/2012/06/traffic-jam.jpg"
#          alt="Bank Note " width="600">
#   </td></tr>
#   <tr><td></td></tr>
#   <tr><td>
#   <img src="https://miro.medium.com/max/593/1*pfmeGgGM5sxmLBQ5IQfQew.png"
#          alt="Matrix" width="600">
#   <tr><td></td></tr>
#   <tr><td>And seems like a bad consumption problem to have ...</td></tr>
# </table>
#   
# #### Solution:
# 
# Since this is a regression problem where the value is a range of numbers, we can use the
# common Random Forest Algorithm in Scikit-Learn. Most regression models are evaluated with
# four [standard evalution metrics](https://medium.com/usf-msds/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4): 
# 
# * Mean Absolute Error (MAE)
# * Mean Squared Error (MSE)
# * Root Mean Squared Error (RSME)
# * R-squared (r2)
# 
# This example is borrowed from this [source](https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/) and modified and modularized for this tutorial
# 
# Aim of this this:
# 
# 1. Understand MLflow Tracking API
# 2. How to use the MLflow Tracking API
# 3. Use the MLflow API to experiment several Runs
# 4. Interpret and observe runs via the MLflow UI
# 
# Some Resources:
# * https://mlflow.org/docs/latest/python_api/mlflow.html
# * https://www.saedsayad.com/decision_tree_reg.htm
# * https://towardsdatascience.com/understanding-random-forest-58381e0602d2
# * https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# * https://towardsdatascience.com/regression-an-explanation-of-regression-metrics-and-what-can-go-wrong-a39a9793d914
# * https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/

# COMMAND ----------

# MAGIC %md Define all the classes and bring them into scope

# COMMAND ----------

# MAGIC %run ./setup/class_setup

# COMMAND ----------

# MAGIC %md ### Load the Dataset

# COMMAND ----------

# load and print dataset
dataset = Utils.load_data("https://github.com/dmatrix/tutorials/raw/master/mlflow/labs/data/petrol_consumption.csv")
dataset.head(5)

# COMMAND ----------

# MAGIC %md Get descriptive statistics for the features

# COMMAND ----------

dataset.describe()

# COMMAND ----------

# iterate over several runs with different parameters, such as number of trees. 
# For expermientation, try max_depth and consult the documentation what tunning parameters
# may affect a better outcome.
max_depth = 0
for n in range (20, 250, 50):
  max_depth = max_depth + 2
  params = {"n_estimators": n, "max_depth": max_depth}
  rfr = RFRModel.new_instance(params)
  (experimentID, runID) = rfr.mlflow_run(dataset)
  print("MLflow Run completed with run_id {} and experiment_id {}".format(runID, experimentID))
  print("-" * 100)

# COMMAND ----------

# MAGIC %md **Note**:
# 
# With 20 trees, the root mean squared error is `64.93`, which is greater than 10 percent of the average petrol consumption i.e., `576.77`. 
# This may sugggest that we have not used enough estimators (trees).

# COMMAND ----------

# MAGIC %md ### Let's Explorethe MLflow  UI
# 
# * Add Notes & Tags
# * Compare Runs pick two best runs
# * Annotate with descriptions and tags
# * Evaluate the best run

# COMMAND ----------

# MAGIC %md #### Homework Assignment. Try different runs with:
# 1. Change the [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) to a [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html?highlight=linearregression#sklearn.linear_model.LinearRegression)
#     * compare the evaluation metrics and ascertain which one is better
# 2. Change or add parameters, such as depth of the tree or random_state: 42 etc.
# 3. Change or alter the range of runs and increments of n_estimators
# 4. Check in MLfow UI if the metrics are affected
# 5. Convert your machine learning model code from work, use MLflow APIs to track your experiment
# 6. Explore the [MLflow GitHub Examples](https://mlflow.org/docs/latest/tutorials-and-examples/index.html)
