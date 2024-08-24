import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from houseprice_model.config.core import config
from houseprice_model.processing.features import OutlierHandler, NullImputer,columnDropperTransformer

houseprice_pipe = Pipeline([

    ######### Null Imputation ###########
    ('CRIM_imputation', NullImputer(config.model_config.CRIM_var)),
    
    ('ZN_imputation', NullImputer(config.model_config.ZN_var)),
    
    ('INDUS_imp', NullImputer(config.model_config.INDUS_var)),
    
    ('CHAS_imp', NullImputer(config.model_config.CHAS_var)),
    
    ('AGE_imp', NullImputer(config.model_config.AGE_var)),
    
    ('LSTAT', NullImputer(config.model_config.LSTAT_var)),
    
    ######### Outlier Handle #############
    ('Handle_outlier_RM', OutlierHandler(config.model_config.RM_var)),
    
    ('Handle_outlier_DIS', OutlierHandler(config.model_config.DIS_var)),
    
    ('Handle_outlier_B', OutlierHandler(config.model_config.B_var)),
    
    ('Handle_outlier_PTRATIO', OutlierHandler(config.model_config.PTRATIO_var)),
    
    ######### Scale features ##############
    ('scaler', StandardScaler()),
    
    ########## Regressor ##################
    ('model_rf', RandomForestRegressor(n_estimators = config.model_config.n_estimators,
                                       max_depth = config.model_config.max_depth,
                                       random_state = config.model_config.random_state,
                                       max_leaf_nodes= config.model_config.max_leaf_nodes,
                                       max_features= config.model_config.max_features,
                                       min_samples_split= config.model_config.min_samples_split
                                      ))
    
    ])
