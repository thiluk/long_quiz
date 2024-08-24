import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from houseprice_model import __version__ as _version
from houseprice_model.config.core import config
from houseprice_model.processing.data_manager import load_pipeline
from houseprice_model.processing.data_manager import pre_pipeline_preparation
from houseprice_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
houseprice_pipe = load_pipeline(file_name = pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    
    validated_data, errors = validate_inputs(input_df = pd.DataFrame(input_data))
    
    #validated_data = validated_data.reindex(columns = ['dteday', 'season', 'hr', 'holiday', 'weekday', 'workingday', 
    #                                                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'yr', 'mnth'])
    validated_data = validated_data.reindex(columns = config.model_config.features)
    
    results = {"predictions": None, "version": _version, "errors": errors}
      
    if not errors:
        predictions = houseprice_pipe.predict(validated_data)
        results = {"predictions": np.floor(predictions), "version": _version, "errors": errors}
        print(results)

    return results



if __name__ == "__main__":

    data_in = {'CRIM':[0.023], 'ZN': [7.0],'INDUS':[6.23],'CHAS':[0.0],'NOX':[0.467],'RM':[5.781],
               'AGE':[45.1],'DIS':[5.69],'RAD':[3],'TAX':[234],'PTRATIO':[12.3],'B':[345.8], 'LSTAT':[3.48]}

    make_prediction(input_data = data_in)