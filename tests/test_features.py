
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
import pandas as pd
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from houseprice_model.config.core import config
from houseprice_model.processing.features import NullImputer, OutlierHandler, columnDropperTransformer


def test_outlierhandler(sample_input_data):
    # Given
    test_df = sample_input_data[0].copy()
    transformer = OutlierHandler(config.model_config.RM_var)
    Q1 = np.percentile(test_df.loc[:, config.model_config.RM_var], 25)
    Q3 = np.percentile(test_df.loc[:, config.model_config.RM_var], 75)
    deviation_allowed = 1.5*(Q3 - Q1)
    lower_bound = Q1 - deviation_allowed
    upper_bound = Q3 + deviation_allowed
    
    assert len(test_df[test_df[config.model_config.RM_var] > upper_bound]) >= 0

    # When
    subject = transformer.fit(test_df).transform(test_df)

    # Then
    assert len(subject[subject[config.model_config.RM_var] > upper_bound]) == 0

def test_columndropper_transformer(sample_input_data):
    # Given
    test_df = sample_input_data[0].copy()
    transformer = columnDropperTransformer(
        columns=config.model_config.TAX_var,  
    )
    
    assert len(test_df.columns) == 13
    
    
    # When
    subject = transformer.fit(test_df).transform(test_df)

    # Then
    assert len(subject.columns) == 12
