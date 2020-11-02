# Databricks notebook source
# MAGIC %md Randorm Forest class defintions for:
#  * `RFRBaseModel`
#  * `RFFExperimentModel`

# COMMAND ----------

import os
import numpy as np
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from  mlflow.tracking import MlflowClient

# COMMAND ----------

class RFRBaseModel():

    def __init__(self, params={}):
        """
        Construtor for the RandomForestRegressor
        :param params: dictionary to RandomForestRegressor
        """
        self._params = params
        self._rf = RandomForestRegressor(**params)

    @classmethod
    def new_instance(cls, params={}):
        return cls(params)

    @property 
    def model(self):
        """
        Getter for the model
        :return: return the model
        """
        return self._rf

    @property
    def params(self):
      """
      Getter for model parameters
      returns: Dictionary of model parameters
      """
      return self._params
    
    def mlflow_run(self, df, r_name="Lab-3: Baseline RF Model"):
        """
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run
        :param df: pandas dataFrame
        :param r_name: Name of the experiment as logged by MLflow
        :return: Tuple of MLflow experimentID, runID
        """
        with mlflow.start_run(run_name=r_name) as run:
            X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)
            self._rf.fit(X_train, y_train)
            predictions = self._rf.predict(X_test)

            # Log model and parameters
            mlflow.sklearn.log_model(self.model, "random-forest-model")
            mlflow.log_params(self.params)

            # Create metrics
            mae = metrics.mean_absolute_error(y_test, predictions)
            mse = metrics.mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(y_test, predictions)

            # Log metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            runID = run.info.run_uuid
            experimentID = run.info.experiment_id

            # print some data
            print("-" * 100)
            print("Inside MLflow {} Run with run_id {} and experiment_id {}".format(r_name, runID, experimentID))
            print("Estimator trees        :", self.params["n_estimators"])
            print('Mean Absolute Error    :', mae)
            print('Mean Squared Error     :', mse)
            print('Root Mean Squared Error:', rmse)
            print('R2                     :', r2)

            return (experimentID, runID)

# COMMAND ----------

displayHTML("""
<div>Declared RandomForestModel Base Class with methods:</div>
<li>Declared <b style="color:green">model()</b> returns existing instance of Random Forest Model</li>
<li>Declared <b style="color:green">new_instance(params={...})</b> returns a new instance of RandomForestClassifierModel</li> 
<li>Declared <b style="color:green"> mlflow_run(DataFrame, run_name="name")</b> returns experiment_ID, run_ID</li>

<br/>
""")

# COMMAND ----------

class RFFExperimentModel(RFRBaseModel):
    """
    Constructor for the Experimental RandomForestRegressor.
    """
    def __int__(self, params):
        """
        Call the superclass initializer
        :param params: parameters for the RandomForestRegressor instance
        :return: None
        """
        super(RFRBaseModel, self).__init__(params)

    def mlflow_run(self, df, r_name="Lab-4:RF Experiment Model"):
        """
        Override the base class mlflow_run for this epxerimental runs
        This method trains the model, evaluates, computes the metrics, logs
        all the relevant metrics, artifacts, and models.
        :param df: pandas dataFrame
        :param r_name: name of the experiment run
        :return:  MLflow Tuple (ExperimentID, runID)
        """

        with mlflow.start_run(run_name=r_name) as run:
            # get experimentalID and runID
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id
            
            # split train/test and train the model
            X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)
            self._rf.fit(X_train, y_train)
            predictions = self._rf.predict(X_test)
            # create an Actual vs Predicted DataFrame
            df_preds = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': predictions.flatten()})

            # Log model and parameters
            mlflow.sklearn.log_model(self.model, "random-forest-model")

            # Note we are logging as a dictionary of all params instead of logging each parameter
            mlflow.log_params(self.params)


            # Create metrics
            mse = metrics.mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = metrics.mean_absolute_error(y_test, predictions)
            r2 = metrics.r2_score(y_test, predictions)

            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            
            # Log prediciton/actual values in file as a feature artifact
            temp_file_name = Utils.get_temporary_directory_path("predicted-actual-", ".csv")
            temp_name = temp_file_name.name
            try:
                df_preds.to_csv(temp_name, index=False)
                mlflow.log_artifact(temp_name, "predicted-actual-files")
            finally:
                temp_file_name.close()  # Delete the temp file

            # Create feature importance and save them as artifact
            # This allows us to remove least important features from the dataset
            # with each iteration if they don't have any effect on the predictive power of
            # the prediction.
            importance = pd.DataFrame(list(zip(df.columns, self._rf.feature_importances_)),
                                      columns=["Feature", "Importance"]
                                      ).sort_values("Importance", ascending=False)

            # Log importance file as feature artifact
            temp_file_name = Utils.get_temporary_directory_path("feature-importance-", ".csv")
            temp_name = temp_file_name.name
            try:
                importance.to_csv(temp_name, index=False)
                mlflow.log_artifact(temp_name, "feature-importance-files")
            finally:
                temp_file_name.close()  # Delete the temp file

            # Create residual plots and image directory
            # Residuals R = observed value - predicted value
            (plt, fig, ax) = Utils.plot_residual_graphs(predictions, y_test, "Predicted values for Price ($)", "Residual",
                                                  "Residual Plot")

            # Log residuals images
            temp_file_name = Utils.get_temporary_directory_path("residuals-", ".png")
            temp_name = temp_file_name.name
            try:
                fig.savefig(temp_name)
                mlflow.log_artifact(temp_name, "residuals-plots")
            finally:
                temp_file_name.close()  # Delete the temp file

            print("-" * 100)
            print("Inside MLflow {} Run with run_id {} and experiment_id {}".format(r_name, runID, experimentID))
            print("  mse: {}".format(mse))
            print(" rmse: {}".format(rmse))
            print("  mae: {}".format(mae))
            print("  R2 : {}".format(r2))

            return (experimentID, runID)

# COMMAND ----------

displayHTML("""
<div>Declared RFFExperimentModel Extended Class with methods:</div>
<li>Declared <b style="color:green">model()</b> returns existing instance of Random Forest Model</li>
<li>Declared <b style="color:green">new_instance(params={...})</b> returns a new instance of RandomForestClassifierModel</li> 
<li>Declared <b style="color:green"> mlflow_run(DataFrame, run_name="name")</b> returns experiment_ID, run_ID</li>

<br/>
""")
