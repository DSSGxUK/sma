# Relevance Model - Feature Engineering

This section presents a summary of each of the feature transformations which we tried in our model development phase. Note: not all tranformations are used in the models.
<hr>

# `num_words(df)`
::: src.helpers.feature_transformation.num_words

# `min_num_words(df, text_col='concat_text', n=20)`
::: src.helpers.feature_transformation.min_num_words

# `min_num_words(df, text_col='concat_text', n=1500)`
::: src.helpers.feature_transformation.max_num_words

# `natural_region(df)`
::: src.helpers.feature_transformation.natural_region

# `populated_districts(df)`
::: src.helpers.feature_transformation.populated_districts

# `facility_mentioned(df)`
::: src.helpers.feature_transformation.facility_mentioned

# `quarter(df)`
::: src.helpers.feature_transformation.quarter

# `month(df)`
::: src.helpers.feature_transformation.month

# `weekday(df)`
::: src.helpers.feature_transformation.weekday

# `proportion_urban(df)`
::: src.helpers.feature_transformation.proportion_urban

# `proportion_protected(df)`
::: src.helpers.feature_transformation.proportion_protected

# `proportion_poor_air(df)`
::: src.helpers.feature_transformation.proportion_poor_air

# `num_past_sanctions(df, complaints_facilities, facilities_sanctions, sanctions)`
::: src.helpers.feature_transformation.num_past_sanctions

# `total_past_fines(df, complaints_facilities, facilities_sanctions, sanctions)`
::: src.helpers.feature_transformation.total_past_fines

# `ComplaintType_archivo1(df)`
::: src.helpers.feature_transformation.ComplaintType_archivo1

# `top_tfidf_words(X_train, NUMBER_OF_TFIDF_FEATURES=100)`
::: src.models.three_class_relevance.top_tfidf_words

# `tfidf_word_counts(X_train, tfidf_words)`
::: src.models.three_class_relevance.tfidf_word_counts

# `perform_topic_modelling(X_train, X_test, NUMBER_OF_LDA_TOPICS=50)`
::: src.models.three_class_relevance.perform_topic_modelling
