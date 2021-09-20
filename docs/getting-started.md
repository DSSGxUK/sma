# Quickstart
Get started quickly with modelling and predictions
<hr>

![End-to-end Process for SMA Project](../assets/images/end-to-end-process.png)

# Introduction
In this quickstart guide we will cover the following:

1. Cloning the project repository and installing project dependencies
2. Preparing the data for feature engineering and modelling
3. Training the relevance model and the sanction gravity model
4. Logging model metrics to MLflow and tracking experiments
5. Viewing MLflow results and comparing experiments 
6. Using the model to make predictions 

## Step 1 - Cloning the repository and installing dependencies
Clone the repository by running the following command, then `cd` into the project root and install the project dependencies.
```shell
# clone the repo into ComplaintsModelSMA/
$ git clone https://github.com/inteligenciaambientalsma/ComplaintModelSMA
# cd into the project root
$ cd ComplaintModelSMA/
# install project dependencies
$ pip install -r requirements.txt
```

(Note: if the pip install gives any problems, please install the `es_core_news_sm` and `es_core_news_lg` independently and then run the rest of the requirements. As a last resort, install all packages independently).

## Step 2 - Prepare data
!!! note "You will need access to the SMA project `csv` files or the database"
    This step requires access to the SMA project `.csv` files or access to the SMA databases.
    You can get access to these files by contacting the SMA personnel.
- Run the `ETLer.R` script to download the data from the SMA database into csv files on local disk. `ETLer.R` has two mandatory arguments:

1. The location where you would like the data to be saved (make sure the folder exists, and there is a folder called `processed_r` inside it). 
2. The second input is the run-type and can be "full", "partial", or "filtered". "full" downloads all complaints, "partial" samples 1000 random complaints, and "filtered" only downlads the complaints that don't have 2 predictions registered in the db.

See an example to set up a folder structure and running `ETLer.R` in the example below:

```shell
$ mkdir ../project_data
$ mkdir ../project_data/processed_r
$ Rscript ETLer.R "../project_data" "full"
```
- The `ETLer.R` script will store the output data in the specified folder (on the VM it's `/home/<USERNAME>/data`). This output will consist of CSV files, where each CSV file has the name of a table from the SMA database. To view all the files in the data folder, run:
```shell
$ ls # on linux
OR
$ dir # on windows
```
The output should look like this:
```shell
Detalle_DenunciaColumnasExtraI.csv   Detalle_DenunciaEfectoMedioAmbi.csv
Detalle_DenunciaGeorreferencia.csv   Detalle_DenunciaHecho.csv
Detalle_DenunciaMateriaAmbienta.csv  Detalle_DenunciaPoblacionAfecta.csv
Detalle_DenunciaProcesoFiscaliz.csv  Detalle_DenunciaProcesoSancion.csv
Detalle_DenunciaUnidadFiscaliza.csv  Detalle_ProcesoFiscalizacionUni.csv
Detalle_ProcesoSancionHechoInst.csv  Detalle_ProcesoSancionProcesoFi.csv
Detalle_ProcesoSancionUnidadFis.csv  Detallle_DenunciaImpactoSalud.csv
ReporteProcesoFiscalizacion.csv      Resumen_Denuncia.csv
Resumen_ProcesoSancion.csv           Resumen_UnidadFiscalizable.csv
Variables_territoriales.csv          processed_r
```

- The function will also merge the CSV outputs from the ETL process into usable CSV files that can be passed into the feature engineering functions. These will be saved in the `processed_r` folder.
  For more details about how this works, see [Preparing the Data](../set-up/preparing-the-data) and [Feature Engineering](../set-up/feature-engineering).

## Step 3 - Training the models
The modelling scripts are stored in `src/models/` from the project root. To train the `relevance` model, run
```shell
$ cd src/models/
$ python relevance_test_train.py --data "/home/<USERNAME>/data/processed_r/" --output "RelevanceModel/run_one"
```
!!! caution "Careful with the relative position, on your local computer the path might be: `'../../../data_sma/processed_r/'`"
!!! caution "Model location needs to be unique, change `run_one` above **each time** you train the model"

To train the `sanction gravity` model, run:
```shell
$ python sanction_gravity_train_test.py --data "../../../data_sma/processed_r/" --output "SanctionModel/run_one"
```

### Explanation
In the step above, we run the respective model scripts, passing in two command line arguments:    
1. The **`--data`** argument which specifies the **data path** with the ETL output joint files (not the raw tables)    
2. The **`--output`** argument which specifies where the model `.pkl` file should be saved     

!!! caution "Training the model can take a long time"

## Step 4 - Model Metrics and MLFlow
The training process will automatically log the features used during training to MLflow. It will also log the feature importances to the MLflow dashboard, the pickled model file and the confusion matrix generated during the training. To view the MLflow experiments, run:
```shell
$ mlflow ui
```
The command above will start the MLflow UI. In your browser, visit http://localhost:5000 to view the details of the experiment run. The home page of the experiment displays the most recent experiment runs as shown below:

![MLflow UI dashboard with a few experiment runs](../assets/images/mlflow-dashboard.png)

## Step 5 - Comparing Experiments
Every time you run the training script, it starts a new experiment run and records the results in MLFlow. You can then compare the results from different experiments to see which model provides the best results.
**View an individual experiment by clicking on the experiment as indicated below:**
    ![View Experiment](../assets/images/view-experiment.png)

**The experiment will have the training hyperparameters logged in the "Parameters" section of MLflow as follows**
    ![Model Params](../assets/images/model-hyperparams.png)

**In the same way, the model metrics for the experiment will be logged under the "Metrics" section of MLflow**
    ![Model Metrics](../assets/images/model-metrics.png)

**The experiment also tracks various artifacts including the feature list and the feature importances as indicated below**
    ![Model Features](../assets/images/model-features.png)
    ![Model Artifacts](../assets/images/model-artifacts.png)

**We can compare experiments by selecting multiple experiments and clicking on the "Compare" button**
    ![Compare Experiments](../assets/images/select-multiple-for-compare.png)

**From the comparison, we can compare individual metrics by selecting a metric as indicated below**. In this case, we are comparing the model accuracies.
    ![Comparing Models](../assets/images/compare-accuracies.png)
    ![Comparisons](../assets/images/comparing-model-accuracies.png)
These comparisons allow us to decide the champion model which can then be used for making predictions

## Step 6 - Making Predictions
To make predictions, there are different steps in the relevance and sanction models. The sanctions model has a special script `sanction_gravity_exporter.py` that specificially creates a model pickle file, and trains the NLP assets on the full dataset, rather than just the training set. The Relevance model wasn't benefitted by adding this additional step and therefore just uses the NLP outputs from the training step. There are also two different expectations about model locations, so let's break up this section into each model. 

However, we probably don't want to make predictions for every point since that would be time-intensive. Therefore, let's run the ETLer again, this time limitting the number of datapoints that will be downloaded:

```shell
$ Rscript ETLer.R "../project_data" "partial"
```

### Relevance model

The relevance model predictor uses `relevance_model_saved.pkl` to make predictions. Therefore, prior to running the predictor, we have to specify which model we would like to run, requiring an explicit copy action from the model we would like to use:

```shell
$ cp RelevanceModel/run_one/model.pkl relevance_model_saved.pkl
```

Now we are ready to run the prediction script. We do still need to pass the **data path** to the file, via the input `data`, as shown below:

- *Make predictions with the relevance model*:
```shell
$ python relevance_prediction.py --data "../../../data_sma/processed_r/"
```

### Sanctions model

The sanctions model expects that the "explicit" step is running the outputter and therefore doesn't require a copy action (but care must be taken to ensure that the exporter script is using tthe same parameters/settings from the `test_train` script).

- *Make predictions with the sanctions gravity model*:
```shell
$ python sanction_gravity_exporter.py --data "../../../data_sma/processed_r/"
$ python sanction_gravity_prediction.py --data "../../../data_sma/processed_r/"
```

### Explanation
In the step above, we run the prediction script passing in various command line arguments:     
1. The **`--data`** path specifies the path where the new unseen data is stored        


# Next Steps
The above steps provide a quickstart to running the project. However, the rest of the documentation provides detailed steps into running the project and understanding the code base. To get a more in-depth overview of the models themselves, see [Relevance Model Walkthrough](../relevance-model/relevance-model-walkthrough) or [Sanctions Model Walkthrough](../sanctions-gravity-model/sanctions-gravity-model-walkthrough).

