import mlflow
import argparse
from datetime import datetime
import pandas as pd
import os
import sys

data_preprocess = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/data/')
sys.path.append(data_preprocess)

from data_preprocess import merge_etl_data

features = ['ComplaintType', 'TF-IDF', 'LDA', 'EnvironmentalTopic', 'num_words', 'num_details',
            'facility_num_infractions',
            'natural_region', 'populated_district', 'FacilityEconomicSector',
            'month']

def transform_data(features, new_data, model_type="relevance"):
    model_type = model_type.strip().lower()
    try:
        if (model_type == "sanctions"):
            import sanction_gravity
            X = sanction_gravity.transform_raw_data_to_features(features, new_data, False)
            return X
        
        if(model_type == "relevance"):
            import three_class_relevance
            X = three_class_relevance.transform_raw_data_to_features(features, new_data, False)
            return X

    except ValueError as err:
        raise Exception(f"Classification type must be one of [relevance, sanction]. Got {model_type}") from err
    

def make_prediction(new_data, logged_model_path):
    loaded_model = mlflow.pyfunc.load_model(logged_model_path)
    return loaded_model.predict(new_data)


# RUN
# python predict_model.py --model_path "RelevanceModel/" --data "/files/data_merge/prod" --type relevance

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help = "specifies the model path")
    parser.add_argument("--data", help="Path of the data to use for classification")
    parser.add_argument("--type", help="specifies the classification type")
    args = parser.parse_args()

    # get the prediction data
    prediction_data = merge_etl_data(args.data)

    # get the model type
    model_type = args.type
    new_data = transform_data(features, prediction_data, model_type)

    complaint_ids = new_data["ComplaintId"]
    new_data.drop(["ComplaintId"], axis = 1, inplace=True)

    # get predictions
    predictions = make_prediction(new_data, logged_model_path=args.model_path)

    # These predictions can either be logged to a database or sent back via an API call
    # or saved as a csv file
    model_predictions_df = pd.DataFrame({
        "complaint_id": complaint_ids,
        "model_prediction": predictions,
        "model_type": model_type,
        "prediction_timestamp": datetime.now().isoformat(),
    })

    print(model_predictions_df)
    model_predictions_df.to_csv(f"{model_type}_model_predictions.csv")
