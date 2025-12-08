import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import jaccard_score
import plotly.express as px
import plotly.graph_objects as go

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
    
    for col in feature_columns:
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

def create_set_AR(
        df: pd.DataFrame,
        grouping_columns: list,
        list_set_columns: list,
        feature_columns: list,
        target_column: str,
        ) ->tuple[pd.DataFrame, pd.Series]:
    """
    Create X and y sets from a DataFrame and desired columns for each

    Returns X and y
    """
    df_predict = df.copy()
    
    for col in feature_columns:
        df_predict[col] = df_predict[col].apply(
            lambda x: ','.join(map(str, x)) if isinstance(x, (list, np.ndarray)) else x
        )

    df_predict = df_predict.groupby(
            grouping_columns,
        )[list_set_columns].agg(
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
    target_string: str,
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
    feature_names_y = [target_string + str(int(x)) for x in feature_names_y]
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

def transform_x_y_testAR(
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
    feature_names_y = ['cluster_' + str(int(x)) for x in feature_names_y]
    y_test_transformed = pd.DataFrame(y_test_transformed, columns=feature_names_y)

    return X_test_transformed, y_test_transformed

def display_results(
        y_data_transformed:pd.DataFrame,
        predictions_dense: any
        ):
    
    if not isinstance(predictions_dense, np.ndarray):
        predictions_dense = predictions_dense.toarray()

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

    jaccard_sim_score = jaccard_score(y_data_transformed,
                                      predictions_dense,
                                      average='samples',
                                    )
    print(f"\nJaccard Similarity Score (Samples): {jaccard_sim_score:.2f}")

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

    combined_score = subset_accuracy + (1-hamming_loss_score) + map_score + macro_f1_score# + jaccard_sim_score
    print(f"\nCombined Metrics Score: {combined_score:.2f}")
    print("\nSum of Subset Accuracy, 1-Hamming Loss, Mean Average Precsion, Macro F1, and Jaccard Similarity (samples) scores")

def combined_score(y_true, y_pred):

    hamming_loss_score = hamming_loss(y_true, y_pred)

    map_score = average_precision_score(y_true,
                                        y_pred,
                                        average='macro',
                                        )

    macro_f1_score = f1_score(y_true,
                              y_pred,
                              average='macro',
                              zero_division=1,
                              )

    samples_f1_score = f1_score(y_true,
                              y_pred,
                              average='samples',
                              zero_division=1,
                              )

    combined_score = -hamming_loss_score + map_score + macro_f1_score + samples_f1_score
    return combined_score

def display_error_heatmap(
        y_true: pd.DataFrame,
        y_pred: np.array,
    ) ->None:
    '''
    Create an error heatmap from y_true and y_pred. Created with the help of Google Gemini
    '''

    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.toarray()
    
    is_TP = (y_true == 1) & (y_pred == 1)
    is_TN = (y_true == 0) & (y_pred == 0)
    is_FP = (y_true == 0) & (y_pred == 1)
    is_FN = (y_true == 1) & (y_pred == 0)

    prediction_states = np.empty(y_true.shape, dtype=object)

    prediction_states[is_TP] = 'True Positive (TP)'
    prediction_states[is_TN] = 'True Negative (TN)'
    prediction_states[is_FP] = 'False Positive (FP)'
    prediction_states[is_FN] = 'False Negative (FN)'

    df_vis = pd.DataFrame(prediction_states, columns=y_true.columns)
    df_vis['Sample'] = [f'Sample_{i}' for i in range(len(y_true))]
    df_vis_long = df_vis.melt(
        id_vars='Sample',
        var_name='Label',
        value_name='Prediction_State',
        )

    order = ['True Negative (TN)', 'True Positive (TP)', 'False Negative (FN)', 'False Positive (FP)']
    color_map = {
        'True Positive (TP)': 'green',
        'True Negative (TN)': 'lightgray',
        'False Negative (FN)': 'orange',
        'False Positive (FP)': 'red'
    }

    state_to_int = {state: i for i, state in enumerate(order)}
    int_to_state = {i: state for i, state in enumerate(order)}
    color_scale = [color_map[state] for state in order]

    prediction_ids = np.vectorize(state_to_int.get)(prediction_states)

    df_matrix = pd.DataFrame(
        prediction_ids,
        index=[f'Sample_{i}' for i in range(len(y_true))],
        columns=y_true.columns
    )

    fig = px.imshow(
        df_matrix.values,
        x=df_matrix.columns,
        y=df_matrix.index,
        color_continuous_scale=color_scale,
        title="Prediction Error Matrix (Sample vs. Label)",
    )

    tick_values = np.arange(len(order))
    tick_text = [int_to_state[i] for i in tick_values]

    fig.update_coloraxes(
        colorbar_tickvals=tick_values,
        colorbar_ticktext=tick_text,
        colorbar_title='Prediction State'
    )

    fig.update_layout(
        autosize=False,
        width=800,
        height=600,
        yaxis={'categoryorder':'category descending'}
    )

    fig.show()

def display_label_cardinality_error(
        y_true: pd.DataFrame,
        y_pred: np.ndarray,
    ) ->None:
    '''
    Create a scatterplot from y_true and y_pred to show accuracy of number of labels predicted.
    Created with the help of Google Gemini
    '''
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.toarray()

    true_cardinality = np.sum(y_true, axis=1)
    pred_cardinality = np.sum(y_pred, axis=1)

    mismatches = np.sum(y_true != y_pred, axis=1)
    hamming_loss_Score_ = mismatches / y_true.shape[1]

    df_error_vis = pd.DataFrame({
        'Sample_ID': [f'Sample_{i}' for i in range(y_true.shape[0])],
        'True_Cardinality': true_cardinality,
        'Predicted_Cardinality': pred_cardinality,
        'Hamming_Loss_Score_': hamming_loss_Score_,
        'Total_Errors': mismatches
    })

    fig = px.scatter(
        df_error_vis,
        x='True_Cardinality',
        y='Predicted_Cardinality',
        color='Hamming_Loss_Score_',
        hover_data=['Sample_ID', 'Total_Errors'],
        title='Multilabel Error Analysis: Predicted vs. True Label Cardinality',
        color_continuous_scale=px.colors.sequential.Plasma,
    )

    max_cardinality = y_true.shape[1]
    fig.update_xaxes(
        range=[-0.5, max_cardinality + 0.5],
        tickvals=np.arange(0, max_cardinality + 1),
        title="True Label Cardinality (Complexity)"
    )
    fig.update_yaxes(
        range=[-0.5, max_cardinality + 0.5],
        tickvals=np.arange(0, max_cardinality + 1),
        title="Predicted Label Cardinality (Model Output)"
    )

    fig.add_shape(
        type="line", line=dict(dash='dash', color='gray'),
        x0=0, y0=0, x1=max_cardinality, y1=max_cardinality
    )

    fig.show()

def display_hyperparameter_sensitivity(
        df_results: pd.DataFrame,
        hyperparameter_column: str,
        mean_column: str,
        std_column: str,
        hyperparameter_name: str,
        scoring_name: str,
    ) -> None:   
    
    # Calculate the upper and lower bounds for the error band
    df_results['upper_bound'] = df_results[mean_column] + df_results[std_column].abs()
    df_results['lower_bound'] = df_results[mean_column] - df_results[std_column].abs()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_results[hyperparameter_column],
        y=df_results['upper_bound'],
        line=dict(width=0),
        mode='lines',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=df_results[hyperparameter_column],
        y=df_results['lower_bound'],
        line=dict(width=0), 
        mode='lines',
        fill='tonexty',     
        fillcolor='rgba(0, 0, 0, 0.3)',
        name='Std Error Band'
    ))

    fig.add_trace(go.Scatter(
        x=df_results[hyperparameter_column],
        y=df_results[mean_column],
        line=dict(color='black', width=2),
        mode='lines',
        name='Mean ' + scoring_name
    ))

    fig.update_layout(
        title='Hyperparameter Sensitivity:' + hyperparameter_name,
        xaxis_title=hyperparameter_name,
        yaxis_title='Cross-validated ' + scoring_name + ' with standard error',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=1
        ), 
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=1
        ),
        xaxis_showline=True, 
        yaxis_showline=True,
    )

    fig.show()

def compare_true_to_pred(
        row,
        X_test,
        y_test,
        y_pred
    ):
    df_compare = y_test.iloc[row].to_frame(name='true')
    df_compare['predicted'] = y_pred[row].T
    non_matching_labels = df_compare[df_compare['true'] != df_compare['predicted']]
    matching_positive_labels = df_compare[(df_compare['true'] == 1) & (df_compare['predicted']== 1)]
    print(X_test.iloc[row], '\n', non_matching_labels, '\n', matching_positive_labels)