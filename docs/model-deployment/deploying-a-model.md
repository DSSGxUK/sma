# Deploying a Model
This guide will walk through deploying a model after training
<hr>

# Introduction
After we build the model, we can deploy the model to production and use the model for making predictions on new data. The training step produces a pickled model which we can load into the predictions.

# Deploying A Model
The model deployment for the `Relevance Model` and the `Sanctions Gravity` model is identical, so the steps below should apply to either model. In this case, since we don't need real-time performance, we decided to not set up an API but instead to run the prediction scripts from the `crontab` (the task scheduler) as required.

# Copy the model and assets to Azure
Once the model training is completed, a pickle file will be generated by the model, as well as other assets, including the NLP outputs and one-hot-encoding assets. Copy and upload this file and the conda environment automatically generated along with the file to the Azure instance.
!!! note "Setup Azure Instance"
    Be sure to check that the Azure instance is properly configured as outlined in [Deployment Environment Setup](deployment-environment-setup)

# Basic operation

Once all the files are copied as per usual, the normal server process will trigger, as per `runner.bash`. These are the steps taken:

 1. Run a filtered ETL run (only downloads data that doesn't have a relevance AND sanction run)
 2. Runs the Prediction model on new data (inside it's conda environment)
 3. Runs the Sanction model on new data (inside it's conda environment)
 4. Uploads the new prediction data generated by the models to an Azure db

This file is triggered by the `crontab`, the contents of which can be seen in the base file `crontab.log`.

# Logging

There are three levels of logging.

1. The crontab contents themselves are logged daily to ensure that changes there aren't lost.
2. The runs on `runner.bash` themselves are logged to timeRun.txt. This file not only measures whether the overall process has been successful or not, but times it, in special lines denoted with `xxx`. This, combined with a trivial data-cleaning step can be used to set up a dashboard measuring run-times, and other uses.
3. Each individual step inside `runner.bash` has its own individual log file. Consult these log files to identify if any of the scripts within `runner.bash` have encountered errors. 

# Conclusion
The deployment is a rather straightforward process that does not involve much overhead. Once the model pickle files and associated assets are loaded into the Azure environment, the `runner.bash` script is able to make predictions on new data by loading the pickled file and using it for inference.