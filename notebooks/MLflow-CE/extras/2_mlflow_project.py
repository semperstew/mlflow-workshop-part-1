# Databricks notebook source
import mlflow
from mlflow import projects

# COMMAND ----------

print(mlflow.__version__)

# COMMAND ----------

# MAGIC %md #### Configure databricks CLI

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
dbutils.fs.put("file:///root/.databrickscfg","[DEFAULT]\nhost=https://community.cloud.databricks.com\ntoken = "+token,overwrite=True)

# COMMAND ----------

# MAGIC %md Use MLflow Fluent API

# COMMAND ----------

res_sub = mlflow.run("https://github.com/mlflow/mlflow-example", parameters={"alpha": 0.6, "l1_ratio": 0.1})
print(f"status={res_sub.get_status()}")
print(f"run_id={res_sub.run_id}")

# COMMAND ----------

# MAGIC %md Use MLflow Projects API

# COMMAND ----------

import mlflow
res_sub = projects.run("https://github.com/dmatrix/mlflow-workshop-project-expamle-1", parameters={'batch_size': 5,'epochs': 1000})
print(f"status={res_sub.get_status()}")
print(f"run_id={res_sub.run_id}")
