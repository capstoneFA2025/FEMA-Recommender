import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.ensemble import MajorityVotingClassifier
from skmultilearn.cluster import FixedLabelSpaceClusterer
from skmultilearn.ensemble import RakelO
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from mlxtend.evaluate.time_series import GroupTimeSeriesSplit
from mlxtend.evaluate.time_series import plot_splits
from mlxtend.evaluate.time_series import print_cv_info
from mlxtend.evaluate.time_series import print_split_info
import optuna
from collected_functions import *


### Classification Functions

def create_set(
        df: pd.DataFrame,
        feature_columns: list,
        target_column: str,
        ) ->tuple[pd.DataFrame, pd.Series]:
    """
    Create X and y sets from a DataFrame and desired columns for each

    Returns X and y
    """
    df_predict = df.copy()
    # This groupby process will combine resulting target values into a list
    # This helps the model understand the connections between features and target values
    
    for col in feature_columns:
        # Use a lambda to check if the value is iterable (like a list/array) and join it.
        # This handles lists of strings, lists of numbers, or NumPy arrays.
        # .apply(str) handles cases where the column might contain a mix of lists and non-lists.
        df_predict[col] = df_predict[col].apply(
            lambda x: ','.join(map(str, x)) if isinstance(x, (list, np.ndarray)) else x
        )

    df_predict = df_predict.groupby(
            feature_columns,
        )[target_column].agg(
            lambda x: list(set(x))
        ).reset_index()
    df_predict = df_predict.sort_values(
            by='year',
            ascending=True
        )

    X = df_predict[feature_columns]
    y = df_predict[target_column]

    return X, y

class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    '''
    This was created with the help of an LLM as including MLB in a pipeline is not
    a simple task.
    '''
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
    
    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = X.ravel()
        else:
            X = X.squeeze()  # works for DataFrame/Series

        # X will be a 2D array (n_samples, 1), so flatten it
        self.mlb.fit(X)
        return self
    
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = X.ravel()
        else:
            X = X.squeeze()
        return self.mlb.transform(X)
    
    def get_feature_names_out(self, input_features=None):
        # Prefix with the input column name, if provided
        if input_features is not None:
            prefix = input_features[0]
            return np.array([f"{prefix}_{cls}" for cls in self.mlb.classes_])
        return np.array(self.mlb.classes_)
    
def transform_x_y_train(
    X: pd.DataFrame,
    y: pd.Series,
    feature_columns: list,
    multilabel_feature_columns: list,
    target_column: str,
    ) ->tuple[pd.DataFrame, pd.DataFrame, any, any]:
    """
    Transform X and y for use with future models.

    Returns X_transformer, y_transformer, transformer_X, and transformer_y
    """
    # Process X

    transformers = [
        ('OneHot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), feature_columns)
    ]

    for col in multilabel_feature_columns:
        transformers.append((f"MLB_{col}", MultiLabelBinarizerTransformer(), [col]))

    transformer_X = ColumnTransformer(transformers,
                                      remainder='passthrough',
                                      verbose_feature_names_out=False,
                                      )
   
    X_transformed = transformer_X.fit_transform(X)
    feature_names_x = transformer_X.get_feature_names_out()
    X_transformed = pd.DataFrame(X_transformed, columns=feature_names_x)
    
    # Process y
    transformer_y = MultiLabelBinarizer()

    y_transformed = transformer_y.fit_transform(y)
    feature_names_y = transformer_y.classes_
    feature_names_y = ['ESF_' + str(int(x)) for x in feature_names_y]
    y_transformed = pd.DataFrame(y_transformed, columns=feature_names_y)
    
    return X_transformed, y_transformed, transformer_X, transformer_y

def transform_x_y_test(
        X_test: pd.DataFrame,
        y_test: pd.Series,
        transformer_X: ColumnTransformer,
        transformer_y: MultiLabelBinarizer,
    ) ->tuple[pd.DataFrame, pd.Series]:
    X_test_transformed = transformer_X.transform(X_test)
    feature_names_x = transformer_X.get_feature_names_out()
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names_x)

    y_test_transformed = transformer_y.transform(y_test)
    feature_names_y = transformer_y.classes_
    feature_names_y = ['ESF_' + str(int(x)) for x in feature_names_y]
    y_test_transformed = pd.DataFrame(y_test_transformed, columns=feature_names_y)

    return X_test_transformed, y_test_transformed

def display_results(
        y_data_transformed:pd.DataFrame,
        predictions_dense: any
        ):
    print(f"True Multi-Label Targets:\n{y_data_transformed.head(5)}")
    print(f"\nPredicted Multi-Label Targets:\n{predictions_dense[:5]}")

    subset_accuracy = accuracy_score(y_data_transformed, predictions_dense)
    print(f"\nSubset Accuracy (Exact Match): {subset_accuracy:.2f}")

    hamming_loss_score = hamming_loss(y_data_transformed, predictions_dense)
    print(f"\nHamming Loss (Lower is Better): {hamming_loss_score:.2f}")

    map_score = average_precision_score(y_data_transformed,
                                        predictions_dense,
                                        average='macro',
                                        )
    print(f"\nMean Average Precision (mAP): {map_score:.2f}")

    macro_f1_score = f1_score(y_data_transformed,
                              predictions_dense,
                              average='macro',
                              )
    print(f"\nMacro F1 Score: {macro_f1_score:.2f}")

    report = classification_report(y_data_transformed,
                                   predictions_dense,
                                   target_names=y_data_transformed.columns,
                                   zero_division=0)

    print("\n--- Detailed Classification Report (Macro F1 Components) ---")
    print(report)

    correct_predictions = (predictions_dense == y_data_transformed)
    correct_predictions = np.all(correct_predictions, axis=1)
    print(f"\nRows with perfect predictions: {len(np.where(correct_predictions)[0])}")
    print(np.where(correct_predictions)[0])

    incorrect_predictions = (predictions_dense != y_data_transformed)
    incorrect_predictions = np.any(incorrect_predictions, axis=1)
    print(f"\nRows with incorrect predictions: {len(np.where(incorrect_predictions)[0])}")
    print(np.where(incorrect_predictions)[0])

    matching_cells = (y_data_transformed == predictions_dense)

    total_cells = predictions_dense.size
    total_matching_cells = np.sum(matching_cells)

    percent_matching_cells = total_matching_cells/y_data_transformed.shape[0]
    print(percent_matching_cells, np.mean(percent_matching_cells))

    combined_score = subset_accuracy + (1-hamming_loss_score) + map_score + macro_f1_score
    print(f"\nCombined Metrics Score: {combined_score:.2f}")
