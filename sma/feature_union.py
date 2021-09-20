from typing import List
import pandas as pd

class FeatureUnification():
    """Unify classification features

    Call this class to unify all classification features. Classification features should be passed in as individual dataframes. It is important that the number of sample is the samein all dataframes. That is, the shape should be (n_samples, ...)

    Examples:
    >>> f_union = FeatureUnification()
    >>> feature_1 = pd.DataFrame({"ComplaintDetail": ["complaint 1", "complaint 2", "complaint n"]})
    >>> feature_2 = pd.DataFrame({"tfidf_scores": [1.3, 0.002, 2.6]})
    >>> combined_features = f_union.unify([feature_1, feature_2])
    """
    def _check_shape(self, feature_dfs: List[pd.DataFrame]):
        """Check the shape of the individual feature dataframes to ensure every feature shape is the same. This function will raise an assertion error if the shapes are not the same

        Args:
            feature_dfs (List[pd.DataFrame]): A list of dataframes containing the features for classification
        """
        assert isinstance(feature_dfs, list) == True
        n_rows = feature_dfs[0].shape[0]
        for df in feature_dfs:
            assert df.shape[0] == n_rows

    def unify(self, feature_dfs: List[pd.DataFrame], shape_check=True):
        """Unify the features for classification

        Args:
            feature_dfs (List[pd.DataFrame]): A list of features to be unified
            shape_check (bool, optional): Specifies if the shape of the individual features should be check for equality. Defaults to True.

        Returns:
            pd.Dataframe: Unified features for classification
        """
        if(shape_check):
            self._check_shape(feature_dfs=feature_dfs)
        features = pd.concat(feature_dfs, axis=1, join="outer")
        return features
