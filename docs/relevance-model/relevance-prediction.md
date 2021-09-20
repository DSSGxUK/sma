# Relevance Model – Prediction script
The `relevance_prediction.py` script can be called to take in the latest version of the trained model and output predictions for all complaints passed to it. The data processing steps will be similar to the steps used in the training stage, then the model will be loaded and used to make the predictions. This page is intended to explain how to use the prediction script and to outline the main steps it goes through.

## How to use the Relevance model prediction script
Inside the terminal, navigate to the directory where the `relevance_precition.py` file is located:
```shell
$ cd src/models
```
Inside the script, check that the paths to the data files, the `.csv` file containing the top TF-IDF words and the `.pkl` models for LDA, RAKE and the trained Relevance model are all correct. If they are not, modify these to the correct path. (For more information on these `.csv` and `.pkl` models, see the [walkthrough of the Relevance model](relevance-model-walkthrough.md).)

Then use the following command to run the script:
```shell
$ python relevance_prediction.py
```
This will ingest all the complaints from the data files passed in, process the data and use the pre-trained model to return a prediction for each complaint. The predictions are returned in the following form: `[ComplaintId, ModelType, Prediction, DateTime]`. 

- The `ComplaintId` is simply the ID of the complaint being predicted
- The `ModelType` will be `relevance` for the Relevance model or `sanctions` for the Sanction Gravity model
- The `Prediction` will be the class predicted by the model for that complaint
- The `DateTime` is the date and time of the prediction, which allows the user to know how recent the prediction is – if it was made a long time ago, perhaps the model has been updated in the meantime and it may be worth running the prediction again

## Steps of the prediction script
In this section, we will briefly run through the steps of the Relevance model's prediction script. Several steps are identical to the steps in the [Relevance model walkthrough](relevance-model-walkthrough.md) section, so further information on those steps can be found there.

### 1. Import the dependencies
The dependencies are imported similarly to the way they were for the model training stage.

### 2. Set the parameters and features of the model
The feature set and parameters are set in the same way as they were in the model training stage.

### 3. Load and normalize the data
The data is loaded and normalized to ascii in the same way as it is in the model training stage.

### 4. Remove the EndType
This step only applies if any of the complaints have already been assigned an `EndType`. This should not usually be the case, but it could happen if we want to try predicting older complaints which may already have been resolved.
```python
if 'EndType' in df.columns:
    df = df.drop(columns=['EndType'])
```

### 5. Concatenate the text
Concatenate the text from all the complaint details into a single complaint text per `ComplaintId` and add this as a column of the dataframe. This step is the same as for the model training stage.

### 6. One-hot encode the environmental topics
Get the one-hot encoded environmental topics for each of the complaints and add these columns to the dataframe. This step is the same as for the model training stage.

### 7. One-hot encode the event description data
One-hot encode the event description data and add these columns to the dataframe (if applicable). This step is the same as for the model training stage.

### 8. Apply feature transformations
Apply the desired feature transformations from the `feature_transformation.py` file and add these as columns of the dataframe. This step is the same as for the model training stage.

### 9. Apply the feature transformations based on geographical data
Apply feature engineering based on the geographical data provided by SMA and add these as columns of the dataframe (if applicable). This step is the same as for the model training stage.

### 10. One-hot encode the necessary features
We now need to one-hot encode the categorical variables which are not yet one-hot encoded. To do this, we use `pd.get_dummies()`, like we did in the training step. But now, we need to make sure that the resulting columns are the same as the ones we ended up with during the model training stage, as our model will not be able to deal with a different number of columns than it expects. For example, if the data we are predicting on does not have any rows containing a specific `FacilityRegion` that was present in the training dataset, the one-hot encoding step will result in one less column (the column for that specific region) appearing in the dataframe. Therefore, we need to add a column of zeros for that region into our dataframe, so that it looks the way our model expects it to be.
```python
# One-hot encode the necessary columns
X = pd.get_dummies(X)
# Import the names of all the columns expected by the model, saved during the training stage
with open('relevance_column_names.csv') as f:
      reader = csv.reader(f)
      for row in reader:
        full_column_names = row
# Remove any categories not present in the trained model, as the model will not know how to 
# deal with these
for col in X.columns:
    if col not in full_column_names:
        X.drop(col, axis=1, inplace=True)
# Add in the missiing columns which the model expects
for name in full_column_names:
    if name not in X.columns:
        # Add the column, with zeros in all the rows
        X[name] = pd.Series([0] * X.shape[0])
# Specify the order of the columns to be the same as in the model training stage
X = X[full_column_names]
```

### 11. Apply the text feature engineering steps
Apply the text feature engineering steps (LDA, TF-IDF and RAKE). This step is different than it was in the model training stage, as we are now using the pre-trained information learned from the training stage to make our predictions, rather than re-training these on the new data. To do this, we need to import the information we saved during the model training stage into our prediction script.

#### Preprocess the text to prepare for the text feature extraction
Clean up the text by removing stopwords, lemmatizing and stemming, just like we did in the model training stage.
```python
X.loc[:,'concat_text'] = X['concat_text'].apply(normalize_text).apply(lemmatize_and_stem_text)
```
#### TF-IDF
In the code below, replace `'relevance_tfidf_words.csv'` with the path to the `.csv` file where the top TF-IDF words are saved.
```python
if 'TF-IDF' in FEATURE_NAMES:
    with open('relevance_tfidf_words.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            tfidf_words = row
    X = tfidf_word_counts(X, tfidf_words)
```
#### RAKE
In the code below, replace `'relevance_rake.pkl'` with the path to the `.pkl` file where the fitted RAKE model is saved.
```python
if 'RAKE' in FEATURE_NAMES:
    rake_features = pickle.load(open('relevance_rake.pkl', 'rb'))
    X['rake_feature'] = rake_features.fit_transform(X['concat_text'])
```
#### LDA
In the code below, replace `'relevance_lda_model.pkl'` with the path to the `.pkl` file where the fitted LDA model is saved and`'relevance_id2word.dict'` with the path to the `.dict` file where the learned dictionary for topic modelling is saved.
```python
if 'LDA' in FEATURE_NAMES:
    # Load the fitted LDA model from the .pkl file
    lda_model = pickle.load(open('relevance_lda_model.pkl', 'rb'))
    # Load the id2word dictionary saved at the model training stage
    id2word = gensim.corpora.dictionary.Dictionary.load('relevance_lda_model.pkl')
    # Get the corpus for LDA (including bigrams)
    bigram_test = get_bigram_pred(X)
    corpus = [id2word.doc2bow(text) for text in bigram_test]
    # Get the LDA vectors
    vecs = []
    for i in range(len(X)):
        top_topics = lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(NUMBER_OF_LDA_TOPICS)]
        vecs.append(topic_vec)
    # Create a dataframe for the LDA information
    topics_df = pd.DataFrame(vecs, dtype = float).reset_index(drop=True)
    topics_df.columns = pd.Index(np.arange(1,len(topics_df.columns)+1).astype(str))
    for topic in topics_df.columns:
        topics_df = topics_df.rename(columns={topic:'LDA_Topic_' + str(topic)})
    X = X.reset_index(drop=True)
    X = pd.concat([X, topics_df], axis = 1)
    if 'Number' in X.columns:
        X = X.drop(['Number'], axis=1)
```

### 11. Run the predictions
Now that the data is fully prepared, we can run the predictions. This time, we are not training the model, but simply loading the model we want to use and feeding our data into that model. In the code below, replace `'relevance_model_saved.pkl'` with the path to the `.pkl` file where the trained Relevance model is saved.
```python
# Load the .pkl file where the model is saved
model = pickle.load(open('relevance_model_saved.pkl', 'rb'))
# Make predictions with the model
predictions = model.predict(X)
predictions = numeric_to_class(predictions)
```

### 12. Output the results
Finally, output the results to a dataframe:
```python
model_type = 'relevance'
model_predictions_df = pd.DataFrame({
    "complaint_id": complaint_ids,
    "model_prediction": predictions,
    "model_type": model_type,
    "prediction_timestamp": datetime.now().isoformat(),
})
```

### Putting everything together
To make the whole script run from the command line using the command shown at the top of this page, we just need to define our `main` function as below:
```python
def main():
    X, complaint_ids = transform_raw_data_to_features_pred(FEATURE_NAMES)
    predictions = run_predictions(X, complaint_ids, model_type="random_forest")

if __name__ == "__main__":
    main()
```
