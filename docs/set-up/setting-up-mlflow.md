# Setting Up MLflow
In this section, we we go over setting up MLflow and preparing the platform for experiment tracking
<hr>

# Introduction
This project utilizes MLflow extensively for experiment tracking, running models and comparing models from various runs. MLflow provides a straightforward work environment to perform various experiments, while tracking all the outputs on a dashboard.

# Installation
MLflow is packaged into a pip package and can be installed via the following command
```shell
$ pip install mlflow
```

Check that MLflow installed properly by running
```shell
$ mlflow --version
# output: mlflow, version 1.18.0
```
MLflow provides various Machine Learning functionalities. For this project we primarily used MLFlow to track experiments between runs and to compare the model results across changes in parameters and data transformations. MLflow provides an intuitive dashboard that can be used to view experiment results. It also exposes a python API which can be extended or used as-is in code. The [`ModelMetrics`](../api-docs/model_metrics) class leverages the MLflow Python API to provide various functionalities in terms of logging model output. To start the MLFlow dashboard, run:
```shell
$ mlflow ui
```
!!! note "Port Number"
    MLflow will start on port 5000. However, an optional port number can be specified as follows: `mlflow ui -p <PORT>`

Lastly, we use MLflow in this project to generate the pickled models which are used in making predictions in the production environment. The model pickle file is logged to MLflow after every experiment run, along with an automatically generated conda environment from which we can run the MLfow model.

# Common Errors and How to Fix them
**`Port In Use`** Errors

Rerun the `mlfow ui` command and specify a unique port using the `-p` command

**`Artifacts Not Logged to MLflow`**

Sometimes some artifacts may not be logged to MLFlow during an experiment run. Make sure that the `artifact_path` is properly set in the code. For example, to log a text file to mlfow in the `output` path, run
```python
mlflow.log_artifact("some_text_file.txt", artifact_path="output")
```

**`Logging Value Exceeds 100 items`**

Sometimes when trying to log a long list to MLflow, you may get a limit exceeded error that MLflow throws when the list of things to log is greater than 100. In this case, we suggest either breaking down that list into smaller chunks and logging those individually, or logging the list as an artifact in MLflow in which case you won't need to split up the data.

# Conclusion
MLflow provides an extensive API that allows the user to run multiple experiments and compare results across runs to find the optimal model. In this project we wrap the MLflow API in the `ModelMetrics` class to provide various useful functions that are able to log multiple features and metrics allowing for easy experimentation.
