# Feature Engineering
In this section, we will discuss the feature selection and transformations performed on the data
<hr>

## Introduction
Feature Engineering is a crucial part of any Machine Learning project. This section will dive into the features we selected when building the final models and will provide a high level overview of the feature engineering performed in our model development phase.


## Feature Selection - Exploratory Data Analysis
The first two weeks of this project focused on Exploratory Data Analysis (EDA) in order to identify various patterns in the data and establish relationships within the dataset. The datasets provided by SMA required some data merging, cleaning and basic transformations. The details of the data cleaning performed are discussed in the [Preparing the Data](../set-up/preparing-the-data) section.

Because the goal of this project is to classify complaints, the `ComplaintDetail` column, which contains the full text of the complaint itself, served as the primary source from which various feature transformations were performed to get synthetic features. The `EnvironmentalTopic`, which is an ordinal set of complaint types (noise, dust, vectors, etc) also provided some useful signal during the EDA process. Some complaints also had a facility and some geographical information associated with them: `FacilityRegion` and `FacilityEconomicSubSector` appeared to be particularly helpful variables. Other useful features provided in the dataset included the `FacilityDistrict`, `FacilityEconomicSector` and the complaint date (`DateComplaint`), as well as geographical information about the complaint district and several indicators relating to the population affected by the matter described in the complaint.


## Feature Engineering - List of feature transformations attempted
This section is intended to provide a quick technical summary of each of the feature transformations we tried in our modelling phase.
<hr>

#### Number of words
The number of words in the complaint (across all complaint details).
#### `num_words(df)`
::: src.helpers.feature_transformation.num_words
<hr>

#### Low word count
Indicates whether or not the complaint has an unusually low word count.
#### `min_num_words(df, text_col='concat_text', n=20)`
::: src.helpers.feature_transformation.min_num_words
<hr>

#### High word count
Indicates whether or not the complaint has an unusually high word count.
#### `max_num_words(df, text_col='concat_text', n=1500)`
::: src.helpers.feature_transformation.max_num_words
<hr>

#### Natural Region
The "Natural Region" of the complaint (see [Natural Regions of Chile](https://en.wikipedia.org/wiki/Natural_regions_of_Chile) on Wikipedia).
#### `natural_region(df)`
::: src.helpers.feature_transformation.natural_region
<hr>

#### Highly-populated districts
Indicates whether or not the population of the district exceeds 200,000 (according to 2014 data).
#### `populated_districts(df)`
::: src.helpers.feature_transformation.populated_districts
<hr>

#### Facility mentioned
Indicates whether or not the complaint was able to identify a specific facility.
#### `facility_mentioned(df)`
::: src.helpers.feature_transformation.facility_mentioned
<hr>

#### Quarter
The quarter of the year when the complaint was made.
#### `quarter(df)`
::: src.helpers.feature_transformation.quarter
<hr>

#### Month
The month of the year when the complaint was made.
#### `month(df)`
::: src.helpers.feature_transformation.month
<hr>

#### Weekday
The day of the week when the complaint was made.
#### `weekday(df)`
::: src.helpers.feature_transformation.weekday
<hr>

#### Proportion urban
The proportion of the district which is covered by urban zones.
#### `proportion_urban(df)`
::: src.helpers.feature_transformation.proportion_urban
<hr>

#### Proportion protected
The proportion of the district which is covered by protected areas.
#### `proportion_protected(df)`
::: src.helpers.feature_transformation.proportion_protected
<hr>

#### Proportion poor air quality
The proportion of the district which is covered by a declared area of poor air quality.
#### `proportion_poor_air(df)`
::: src.helpers.feature_transformation.proportion_poor_air
<hr>

#### Number of previous sanctions
The number of previous sanctions handed out to the facility in question.
#### `num_past_sanctions(df, complaints_facilities, facilities_sanctions, sanctions)`
::: src.helpers.feature_transformation.num_past_sanctions
<hr>

#### Total sum of previous fines
The total sum of fines handed out to the facility in question.
#### `total_past_fines(df, complaints_facilities, facilities_sanctions, sanctions)`
::: src.helpers.feature_transformation.total_past_fines
<hr>

#### Citizen complaints
Indicates whether or not the complaint was made by a citizen (`Archivo I` complaints almost always come from citizens).
#### `ComplaintType_archivo1(df)`
::: src.helpers.feature_transformation.ComplaintType_archivo1
<hr>

#### Most important words per target class (by TF-IDF score)
Uses Term Frequency-Inverse Document Frequency (TF-IDF) to find the most important words (and bi- and trigrams) in each target class. The idea behind this is that, for example, if we see a complaint where a word highly important to the `Relevant` class appears several times, this complaint is likely to be a `Relevant` complaint.

To be used in conjunction with `tfidf_word_counts()`
#### `top_tfidf_words(X_train, NUMBER_OF_TFIDF_FEATURES=100)`
::: src.models.three_class_relevance.top_tfidf_words
<hr>

#### Number of times each important TF-IDF word appears
Counts the number of times that each of the top TF-IDF words appears in the complaint text.

To be used in conjunction with `top_tfidf_words()`.
#### `tfidf_word_counts(X_train, tfidf_words)`
::: src.models.three_class_relevance.tfidf_word_counts
<hr>

#### Topic Modelling
Uses Latent Dirichlet Allocation (LDA) to perform topic modelling and learn the most common topics from the corpus (using both unigrams and bigrams to generate the topics).
#### `perform_topic_modelling(X_train, X_test, NUMBER_OF_LDA_TOPICS=50)`
::: src.models.three_class_relevance.perform_topic_modelling
<hr>

#### Presence of RAKE keywords
Indicates whether or not the complaint text contains any of the phrases extracted by the RAKE algorithm. (See [RAKE extraction](../api-docs/rake_extraction)).
