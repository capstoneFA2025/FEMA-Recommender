"""Create input dataframe for ESF prediction model."""

from typing import List
import pandas as pd


def _generate_feature_columns() -> List[str]:
    """Generate feature column names for the model.
    
    Returns:
        List of column names
    """
    states = [
        'AK', 'AL', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL',
        'GA', 'GU', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD',
        'ME', 'MI', 'MN', 'MO', 'MP', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ',
        'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD',
        'TN', 'TX', 'UT', 'VA', 'VI', 'VT', 'WA', 'WI', 'WV', 'WY'
    ]
    
    declaration_types = ['DR', 'EM']
    
    regions = list(range(1, 11))
    
    incident_types = [
        'Biological', 'Chemical', 'Coastal Storm', 'Dam/Levee Break', 'Earthquake',
        'Fire', 'Flood', 'Hurricane', 'Mud/Landslide', 'Other', 'Severe Ice Storm',
        'Severe Storm', 'Snowstorm', 'Straight-Line Winds', 'Terrorist', 'Tornado',
        'Tropical Depression', 'Tropical Storm', 'Typhoon', 'Volcanic Eruption',
        'Winter Storm'
    ]
    
    columns = []
    columns.extend([f'state_{s}' for s in states])
    columns.extend([f'declarationType_{dt}' for dt in declaration_types])
    columns.extend([f'region_{r}' for r in regions])
    columns.extend([f'designatedIncidentTypes_{it}' for it in incident_types])
    
    return columns


def stage_one_input(
    incident_types: List[str],
    state: str,
    declaration_type: str
) -> pd.DataFrame:
    """Create input dataframe for ESF prediction.
    
    Args:
        incident_types: List of incident types
        state: Two-letter state code
        declaration_type: Declaration type ('DR' or 'EM')
    
    Returns:
        DataFrame with one-hot encoded features
    """
    columns = _generate_feature_columns()
    df = pd.DataFrame(0, index=[0], columns=columns)
    
    for incident in incident_types:
        column_name = f'designatedIncidentTypes_{incident}'
        if column_name in df.columns:
            df[column_name] = 1
    
    state_column = f'state_{state}'
    if state_column in df.columns:
        df[state_column] = 1
    
    dec_column = f'declarationType_{declaration_type}'
    if dec_column in df.columns:
        df[dec_column] = 1
    
    return df  