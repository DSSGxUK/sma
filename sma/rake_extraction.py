from rake_nltk import Rake
import pandas as pd
import es_core_news_lg

# Import stopwords from Spacy. See https://spacy.io/models/es#es_core_news_lg
STOP_WORDS = list(es_core_news_lg.load().Defaults.stop_words)

class DenseRakeFeatures():
    def __init__(self, stop_words=STOP_WORDS, language="es", num_phrases=1000, min_length=1, max_length=10, verbose=False):
        """Generate features from text using Rapid Automatic Keyword Extraction (Rake) algorithm.
           
           RAKE generates a list of keywords or key phrases with a corresponding score to each phrase. The word score is calculated as ratio of the word degree to its frequency. That is,
           .. math::
                Score(word) = \frac{deg(worf)}{freq(word))}
        Args:
            stop_words (list, optional): A list of words to be ignored for the keyword extraction. Defaults to the spanish stop words provided by Spacy. See (https://spacy.io/models/es#es_core_news_lg). 
            language (str, optional): The language used for stopwords. Defaults to "es".
            num_phrases (int, optional): The number of phrases to return. Defaults to 1000
            min_length (int, optional): Minimum length of words per phrase. Defaults to 1.
            max_length (int, optional): Maximum length of words per phrase. Defaults to 5.
            verbose (bool, optional): Print verbose information about keyword extraction. Defaults to False.
        
        Examples:
            >>> df = pd.DataFrame({"text": [
                "este es un ejemplo de prueba para el documento 1",
                "este es también otro documento"])
            >>> drf = DenseRakeFeatures(verbose=True)
            >>> rake_features = drf.fit_transform(df["text"])
            >>> print(rake_features)
        """
        self.verbose = verbose
        self.R = Rake(stopwords=stop_words,language=language, min_length=min_length, max_length=max_length)
        self._ranked_phrases = None
        self.num_phrases = num_phrases

    def _create_ranked_col(self, data_row):
        if(any(phrase in str(data_row) for phrase in self._ranked_phrases)):
            return 1
        return 0

    def fit(self, raw_documents):
        """Fit the raw documents to the rake pipeline

        Args:
            raw_documents (list): A list of documents to extract RAKE keywords from

        Returns:
            pd.DataFrame: A dataframe with the rake phrases and corresponding score
        """
        return self
    
    def transform(self, raw_documents):
        """Transform documents to RAKE phrases

        Args:
            raw_documents (list): A list of documents for feature extraction

        Returns:
            pd.DataFrame: A dataframe with the RAKE features and scores
        """
        if(isinstance(raw_documents, pd.Series)):
            raw_documents = raw_documents.fillna("nan").tolist()
        assert isinstance(raw_documents, list) == True

        if(self.verbose):
            print("Running RAKE keyword extraction...")
        self.R.extract_keywords_from_sentences(sentences=raw_documents)
        # rake_feature_scores = self.R.get_ranked_phrases_with_scores()
        self._ranked_phrases = self.R.get_ranked_phrases()[:self.num_phrases]

        # rake_features_df = pd.DataFrame(rake_feature_scores, columns=["rake_score", "rake_feature"])
        if(self._ranked_phrases is not None):
            has_ranked_word = map(self._create_ranked_col, raw_documents)
        rake_features_df = pd.DataFrame(has_ranked_word, columns=["rake_feature"])
        return rake_features_df

    def fit_with_classes(self, raw_documents: pd.Series, class_list: pd.Series):
        """Creates RAKE features with N-classes specified byt the class list parameter

        Args:
            raw_documents (pd.Series): The raw documents to extract the RAKE phrases from
            class_list (pd.Series): A classlist used as a grouping coulum for RAKE features

        Returns:
            pd.DataFrame: Rake keyword binaries per class
        """
        assert isinstance(raw_documents, pd.Series) and isinstance(class_list, pd.Series) == True
        if(class_list is None):
            return self.fit(raw_documents)
        text_data_with_classes = pd.concat([raw_documents, class_list], keys=["raw_documents", "rake_class"], axis=1)
        # get the unique classes from
        unique_classes = class_list.dropna().unique().tolist()
        
        for rake_class in unique_classes:
            raw_docs_per_class = text_data_with_classes.loc[text_data_with_classes["rake_class"] == rake_class]
            raw_docs_per_class = raw_docs_per_class.fillna("nan")
            self.R.extract_keywords_from_sentences(sentences=raw_docs_per_class["raw_documents"].tolist())
            self._ranked_phrases = self.R.get_ranked_phrases()[:self.num_phrases]
            if(self._ranked_phrases is not None):
                has_ranked_word = text_data_with_classes["raw_documents"].apply(self._create_ranked_col)
            _rake_class_col_name = f"rake_feature_{rake_class.replace(' ', '_')}"
            text_data_with_classes[_rake_class_col_name] = has_ranked_word
        
        # drop the raw documents and class columns
        text_data_with_classes = text_data_with_classes.drop(["raw_documents", "rake_class"], axis=1)
        return text_data_with_classes

    def fit_transform(self, raw_documents):
        """Fit the raw documents and transform the documents, the same as calling fit then transform

        Args:
            raw_documents (pd.Series): The raw documents to extract the RAKE phrases from

        Returns:
            pd.DataFrame: Rake keyword binaries
        """
        return self.fit(raw_documents=raw_documents).transform(raw_documents=raw_documents)
