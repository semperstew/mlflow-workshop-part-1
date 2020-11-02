# Databricks notebook source
# MAGIC %md
# 
# All classes used for this tutorial are defined here. 

# COMMAND ----------

import os
import numpy as np
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from  mlflow.tracking import MlflowClient


class RFRModel():
    """
    General class for Scikit-Learn RandomForestRegressor
    """
    # class wide variables common to all instances
    # that keep track of cumulative estimators and rsme
    # so we can plot the results
    rsme = []
    estimators = []

    def __init__(self, params={}):
        """
        Constructor for the RandomForestRegressor
        :param params: dictionary to RandomForestRegressor
        """
        self.rf = RandomForestRegressor(**params)
        self._params = params

    @classmethod
    def new_instance(cls, params={}):
        return cls(params)

    @property
    def model(self):
        """
        Getter to return the model created
        :return: handle or instance of the RandomForestReqgressor
        """
        return self.rf

    @property
    def params(self):
      """
      Getter for model parameters 
      """
      return self._params
      
    def mlflow_run(self, df, r_name="Lab-1:RF Petrol Regression Experiment"):
        """
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run
        :param df: pandas dataFrame
        :param r_name: Name of the run as logged by MLflow
        :return: MLflow Tuple (ExperimentID, runID)
        """

        with mlflow.start_run(run_name=r_name) as run:
            # get all feature independent attributes
            X = df.iloc[:, 0:4].values
            # get all the values of last columns, dependent variables,
            # which is what we want to predict as our values, the petrol consumption
            y = df.iloc[:, 4].values

            # create train and test data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # Feature Scaling, though for RF is not necessary.
            # z = (X - u)/ s, where u is the man, s the standard deviation
            # get the handle to the transformer
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            # train and predict
            self.rf.fit(X_train, y_train)
            y_pred = self.rf.predict(X_test)

            # Log model and params using the MLflow sklearn APIs
            mlflow.sklearn.log_model(self.model, "random-forest-reg-model")
            mlflow.log_params(self.params)

            # compute  metrics; r2 is a statistical measure of how well the
            # data fits the model: higher the value indicates better fit.
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            rsme = np.sqrt(mse)
            r2 = metrics.r2_score(y_test, y_pred)
            

            # Log metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rsme", rsme)
            mlflow.log_metric("r2", r2)

            # update global class instance variable with values
            self.rsme.append(rsme)
            self.estimators.append(self.params["n_estimators"])

            # plot graphs and save as artifacts
            (fig, ax) = Utils.plot_graphs(self.estimators, self.rsme, "Random Forest Estimators", "Root Mean Square", "Root Mean Square vs Estimators")

            # get current run and experiment id
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id

            # create temporary artifact file name and log artifact
            temp_file_name = Utils.get_temporary_directory_path("rsme_estimators-", ".png")
            temp_name = temp_file_name.name
            try:
                fig.savefig(temp_name)
                mlflow.log_artifact(temp_name, "rsme_estimators_plots")
            finally:
                temp_file_name.close()  # Delete the temp file

            # print some data
            print("-" * 100)
            print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
            print("Estimator trees        :", self.params["n_estimators"])
            print('Mean Absolute Error    :', mae)
            print('Mean Squared Error     :', mse)
            print('Root Mean Squared Error:', rsme)
            print('R2                     :', r2)
            
            return (experimentID, runID)

# COMMAND ----------

displayHTML("""
<div>Declared RandomForestRegress Model Class with methods:</div>
<li>Declared <b style="color:green">model()</b> returns existing instance of Random Forest Model</li>
<li>Declared <b style="color:green">new_instance(params={...}</b></li> returns a new instance of RandomForestClassifierModel</li> 
<li>Declared <b style="color:green"> mlflow_run(DataFrame, run_name="name")</b> returns experiment_ID, run_ID</li>

<br/>
""")

# COMMAND ----------

import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score

class RFCModel():

    def __init__(self, params={}):
        """
        Constructor for RandamForestClassifier
        :param params: parameters for the constructor such as no of estimators, depth of the tree, random_state etc
        """
        self.rf = RandomForestClassifier(**params)
        self._params = params

    @classmethod
    def new_instance(cls, params={}):
        return cls(params)

    @property
    def model(self):
        """
        Getter for the property the model
        :return: return the model
        """
        
        return self.rf
  
    @property 
    def params(self):
      """
      Getter for the property the model
        :return: return the model params
      """
      return self._params
    
    def mlflow_run(self, df, r_name="Lab-2:RF Bank Note Classification Experiment"):
        """
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run
        :param df: pandas dataFrame
        :param r_name: Name of the experiment as logged by MLflow
        :return: MLflow Tuple (ExperimentID, runID)
        """

        with mlflow.start_run(run_name=r_name) as run:
            # get all rows and columns but the last column, which is our class
            X = df.iloc[:, 0:4].values
            # get all observed values in the last columns, which is what we want to predict
            y = df.iloc[:, 4].values

            # create train and test data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # Feature Scaling
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            # train and predict
            self.rf.fit(X_train, y_train)
            y_pred = self.rf.predict(X_test)

            # Log model and params using the MLflow sklearn APIs
            mlflow.sklearn.log_model(self.model, "random-forest-class-model")
            mlflow.log_params(self.params)

            # compute evaluation metrics
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test,y_pred)
            # ROC = summary of all confusion matrices that each
            # threshold produces
            roc = metrics.roc_auc_score(y_test, y_pred)

            # get confusion matrix values
            true_positive = conf_matrix[0][0]
            true_negative = conf_matrix[1][1]
            false_positive = conf_matrix[0][1]
            false_negative = conf_matrix[1][0]

            # get classification matrics as a dictionary
            class_report = classification_report(y_test,y_pred, output_dict=True)
            recall_0 = class_report['0']['recall']
            f1_score_0 = class_report['0']['f1-score']
            recall_1 = class_report['1']['recall']
            f1_score_1 = class_report['1']['f1-score']

            # log metrics
            mlflow.log_metric("accuracy_score", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("true_positive", true_positive)
            mlflow.log_metric("true_negative", true_negative)
            mlflow.log_metric("false_positive", false_positive)
            mlflow.log_metric("false_negative", false_negative)
            mlflow.log_metric("recall_0", recall_0)
            mlflow.log_metric("f1_score_0", f1_score_0)
            mlflow.log_metric("recall_1", recall_1)
            mlflow.log_metric("f1_score_1", f1_score_1)
            mlflow.log_metric("roc", roc)

            # get current run and experiment id
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id

            # create confusion matrix images
            (plt, fig, ax) = Utils.plot_confusion_matrix(y_test, y_pred, y, title="Bank Note Classification Confusion Matrix")

            # create temporary artifact file name and log artifact
            temp_file_name = Utils.get_temporary_directory_path("confusion_matrix-", ".png")
            temp_name = temp_file_name.name
            try:
                fig.savefig(temp_name)
                mlflow.log_artifact(temp_name, "confusion_matrix_plots")
            finally:
                temp_file_name.close()  # Delete the temp file

            # print some data
            print("-" * 100)
            print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
            print("Estimators trees:", self.params["n_estimators"])
            print(conf_matrix)
            print(classification_report(y_test,y_pred))
            print("Accuracy Score:", acc)
            print("Precision     :", precision)
            print("ROC           :", roc)

            return (experimentID, runID)


# COMMAND ----------

displayHTML("""
<div>Declared RandomForestClassifierModel Class with methods:</div>
<li>Declared <b style="color:green">model()</b> returns existing instance of Random Forest Model</li>
<li>Declared <b style="color:green">params()</b> returns existing Random Forest Model's init parameters.</li>
<li>Declared <b style="color:green">new_instance(params={...})</b> returns a new instance of RandomForestClassifierModel</li> 
<li>Declared <b style="color:green"> mlflow_run(DataFrame, run_name="name")</b> returns experiment_ID, run_ID</li>

<br/>
""")

# COMMAND ----------


class RFRBaseModel():

    def __init__(self, params={}):
        """
        Construtor for the RandomForestRegressor
        :param params: dictionary to RandomForestRegressor
        """
        self._params = params
        self.rf = RandomForestRegressor(**params)

    @classmethod
    def new_instance(cls, params={}):
        return cls(params)

    @property 
    def model(self):
        """
        Getter for the model
        :return: return the model
        """
        return self.rf

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
            self.rf.fit(X_train, y_train)
            predictions = self.rf.predict(X_test)

            # Log model and parameters
            mlflow.sklearn.log_model(self.model, "random-forest-model")
            mlflow.log_params(self.params)

            # Create metrics
            mae = metrics.mean_absolute_error(y_test, predictions)
            mse = metrics.mean_squared_error(y_test, predictions)
            rsme = np.sqrt(mse)
            r2 = metrics.r2_score(y_test, predictions)

            # Log metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rsme", rsme)
            mlflow.log_metric("r2", r2)

            runID = run.info.run_uuid
            experimentID = run.info.experiment_id

            # print some data
            print("-" * 100)
            print("Inside MLflow {} Run with run_id {} and experiment_id {}".format(r_name, runID, experimentID))
            print("Estimator trees        :", self.params["n_estimators"])
            print('Mean Absolute Error    :', mae)
            print('Mean Squared Error     :', mse)
            print('Root Mean Squared Error:', rsme)
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
            self.rf.fit(X_train, y_train)
            predictions = self.rf.predict(X_test)
            # create an Actual vs Predicted DataFrame
            df_preds = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': predictions.flatten()})

            # Log model and parameters
            mlflow.sklearn.log_model(self.rf, "random-forest-model")

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
            mlflow.log_metric("rsme", rmse)
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
            importance = pd.DataFrame(list(zip(df.columns, self.rf.feature_importances_)),
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
