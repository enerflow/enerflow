#!/usr/bin/python

import sys, os, glob, shutil, json, re
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib 

import shap
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

#import sklearn as skl
#import statsmodels.api as sm

from sklearn.isotonic import IsotonicRegression

def load_dfs_trial(path, index_col=[0,1], header=[0,1]): 
    files = glob.glob(path)
    files = natural_sort(files)
    dfs_split = []
    for file in files: 
        df = pd.read_csv(file, index_col=index_col, header=header, parse_dates=True)
        dfs_split.append(df)
    
    return dfs_split


class Trial(object):
    def __init__(self, params_json):
        self.params_json = params_json

        # Mandatory input variables
        self.trial_name = params_json['trial_name']
        self.trial_comment = params_json['trial_comment']
        self.path_result = params_json['path_result']
        self.trial_path = self.path_result+self.trial_name
        self.path_preprocessed_data = params_json['path_preprocessed_data']
        self.filename_preprocessed_data = params_json['filename_preprocessed_data']
        self.sites = params_json['sites']
        self.features = params_json['features']
        self.target = params_json['target']
        self.model_params = params_json['model_params']
        self.regression_params = params_json['regression_params']
        self.save_options = params_json['save_options']
        self.early_stopping_by_cv = params_json.get('early_stopping_by_cv', False)
        
        if 'parallel_processing' in params_json:
            self.parallel_processing = params_json['parallel_processing']
        else:
            self.parallel_processing = {'backend': 'threading',
                                        'n_workers': 1}
            
        if 'quantile' in self.regression_params['type']:
            self.alpha_q = np.arange(self.regression_params['alpha_range'][0],
                                     self.regression_params['alpha_range'][1],
                                     self.regression_params['alpha_range'][2])
            if len(self.alpha_q) == 0: 
                raise ValueError('Number of quantiles needs to be larger than zero.')

        # Optional input variables
        if 'random_seed' in params_json:
            self.random_seed = params_json['random_seed']
        else:
            self.random_seed = None
        if 'categorical_features' in params_json:
            self.categorical_features = params_json['categorical_features']
        else: 
            self.categorical_features = 'auto'
        if 'feature_lags' in params_json:
            self.feature_lags = params_json['feature_lags']
        else: 
            self.feature_lags = None
        if 'diff_target_with_physical' in params_json:
            self.diff_target_with_physical = params_json['diff_target_with_physical']
        else: 
            self.diff_target_with_physical = False
        if 'target_smoothing_window' in params_json:
            self.target_smoothing_window = params_json['target_smoothing_window']
        else: 
            self.target_smoothing_window = 1
        if 'train_only_zenith_angle_below' in params_json:
            self.train_only_zenith_angle_below = params_json['train_only_zenith_angle_below']
        else: 
            self.train_only_zenith_angle_below = False
        if 'time_weight_params' in params_json: 
            self.time_weight_params = params_json['time_weight_params']
        else:
            self.time_weight_params = False
        if 'target_level_weight_params' in params_json: 
            self.target_level_weight_params = params_json['target_level_weight_params']
        else:
            self.target_level_weight_params = False     
        if 'custom_weight_column' in params_json: 
            self.custom_weight_column = params_json['custom_weight_column']
        else:
            self.custom_weight_column = False   

        if 'datetime_splits' in params_json: 
            self.datetime_splits = params_json['datetime_splits']
        elif 'valid_fraction' in params_json:
            self.valid_fraction = params_json['valid_fraction']
        elif 'cv_splits' in params_json:
            pass
        else:
            raise ValueError('One of `datetime_splits`, `train_test_splits` or `crossvalidation_splits` must be given in params_json.')
        
        if 'splits' in params_json:
            self.splits = params_json['splits']

        # Runtime
        self.parallel_backend = params_json.get("parallel_backend", "threading")


    def load_data(self, path_data=None):
        # Load preprocessed data

        if path_data is None:
            path_data = self.path_preprocessed_data+self.filename_preprocessed_data
        df = pd.read_csv(path_data, header=[0,1], index_col=[0,1], parse_dates=True)

        return df


    def initial_checks(self, df):
        if not all([feature in df.columns.levels[1] for feature in self.features]):
            raise ValueError('All specified features are not present in data.')


    def generate_splits(self, df):
        # Build up splits dependent on input 

        if hasattr(self, 'datetime_splits'):
            self.splits = self.params_json['datetime_splits']
        elif hasattr(self, 'valid_fraction'):
            index = df.groupby('valid_datetime').first().index
            n_index = len(index)
            self.splits = {'train': [[[index[0], index[int((1-self.valid_fraction)*n_index)]]]],
                           'valid': [[[index[int((1-self.valid_fraction)*n_index)+1], index[-1]]]]}
        
        self.params_json['splits'] = self.splits
        
        return self.splits


    def generate_dataset(self, df, split=None): 

        def add_lags(df, feature_lags): 
            # Lagged features
            vspec = pd.DataFrame([(k, lag) for k, v in feature_lags.items() for lag in v], columns=["Variable", "Lag"]) \
                                 .set_index("Variable") \
                                 .sort_values("Lag")

            dfs_lag = []
            for lag, variables in vspec.groupby("Lag").groups.items():
                df_lag = df.loc[:, sorted(variables)].groupby('ref_datetime').shift(lag)

                df_lag.columns = ['%s_lag%s' % (variable, lag) for variable in sorted(variables)]
                dfs_lag.append(df_lag)

            df_lags = pd.concat(dfs_lag, axis=1)
            df = pd.concat([df, df_lags], axis=1)
            lagged_features = list(df_lags.columns)

            return df, lagged_features

        # Split up dataset in features and target
        if split:
            df = pd.concat([df.loc[pd.IndexSlice[:, s[0]:s[1]], :] for s in split], axis=0).drop_duplicates(keep='first')
        df_X = df.loc[:, self.features]
        df_y = df.loc[:, [self.target]]

        # Add lagged variables
        if self.feature_lags is not None: 
            df_X, lagged_features = add_lags(df_X, self.feature_lags)            
            self.all_features = self.features+lagged_features
        else:
            self.all_features = self.features

        # Remove samples where either all features are nan or target is nan
        is_nan = df_X.isna().all(axis=1) | df_y.isna().all(axis=1)
        df_model = pd.concat([df_X, df_y], axis=1)[~is_nan]

        # Keep all timestamps for which zenith <= prescribed value (day timestamps)
        if self.train_only_zenith_angle_below:
            idx_day = df_model[df_model['zenith'] <= self.train_only_zenith_angle_below].index
            df_model = df_model.loc[idx_day, :]

        # Create target and feature DataFrames
        if self.diff_target_with_physical:
            df_model[self.target] = df_model[self.target]-df_model[self.diff_target_with_physical]

        # Use mean window to smooth target
        df_model[self.target] = df_model[self.target].rolling(self.target_smoothing_window, win_type='boxcar', center=True, min_periods=0).mean()

        # Apply time-based sample weighting
        if self.time_weight_params:
            weight_end = self.time_weight_params['weight_end']
            weight_shape = self.time_weight_params['weight_shape']
            valid_times = df_model.index.get_level_values('valid_datetime')
            days = np.array((valid_times[-1]-valid_times).total_seconds()/(60*60*24))
            time_weight = (1-weight_end)*np.exp(-days/weight_shape)+weight_end
        else:
            time_weight = np.ones(df.shape[0])

        # Apply target level-based sample weighting
        if self.target_level_weight_params:
            weight_end = self.target_level_weight_params['weight_end']
            weight_shape = self.target_level_weight_params['weight_shape']
            target = df_model[self.target]
            target_min = target.min()
            target_max = target.max()
            b = (1-weight_end)/(np.exp(-target_min/weight_shape)-np.exp(-target_max/weight_shape))
            a = weight_end+b*np.exp(-target_min/weight_shape)
            level_weight = a-b*np.exp(-target/weight_shape)
        else: 
            level_weight = np.ones(df.shape[0])

        # Apply custom sample weighting
        if self.custom_weight_column:
            df_custom_weight = df[self.custom_weight_column]
            custom_weight = df_custom_weight[df_model.index].values
        else:
            custom_weight = np.ones(df.shape[0])

        weight = time_weight*level_weight*custom_weight

        return df_X, df_y, df_model, weight


    def generate_dataset_split_site(self, df, split_set='train'):
        # Generate train and valid splits

        print('Generating dataset...')
        time.sleep(0.2)
        dfs_X_split_site, dfs_y_split_site, dfs_model_split_site, weight_split_site = [], [], [], []
        with tqdm(total=len(self.splits[split_set])*len(self.sites)) as pbar:
            for split in self.splits[split_set]:
                dfs_X_site, dfs_y_site, dfs_model_site, weight_site = [], [], [], []
                for site in self.sites:

                    df_X, df_y, df_model, weight = self.generate_dataset(df[site], split)

                    dfs_X_site.append(df_X)
                    dfs_y_site.append(df_y)
                    dfs_model_site.append(df_model)
                    weight_site.append(weight)

                    pbar.update(1)

                dfs_X_split_site.append(dfs_X_site)
                dfs_y_split_site.append(dfs_y_site)
                dfs_model_split_site.append(dfs_model_site)
                weight_split_site.append(weight_site)

        return dfs_X_split_site, dfs_y_split_site, dfs_model_split_site, weight_split_site


    def create_fit_model(self, model_name, df_model_train, objective='mean', alpha=None, df_model_valid=None, weight=None, num_rounds=None, early_stopping=None):
        # Create and fit model. This method could potentially be split up in create and fit seperately. 
        
        if df_model_valid is not None:
            eval_set =[(df_model_valid[self.all_features], df_model_valid[[self.target]])]
#            eval_set =[(df_model_train[self.all_features], df_model_train[[self.target]]), (df_model_valid[self.all_features], df_model_valid[[self.target]])]
        else:
            eval_set = []            
            #eval_set =[(df_model_train[self.all_features], df_model_train[[self.target]])]

        if model_name.split('_')[0] == 'lightgbm':
            if objective == 'mean': 
                objective_lgb = 'mean_squared_error'
                eval_key_name = 'l2'
            elif objective == 'quantile': 
                objective_lgb = 'quantile'
                eval_key_name = 'quantile'
            else: 
                raise ValueError("'objective' for lightgbm must be either 'mean' or 'quantile'")
                     
            model = lgb.LGBMRegressor(objective=objective_lgb,
                                      alpha=alpha,
                                      boosting_type=self.model_params[model_name].get('boosting_type', 'gbdt'),
                                      n_estimators=num_rounds,
                                      learning_rate=self.model_params[model_name].get('learning_rate', 0.1), 
                                      max_depth=self.model_params[model_name].get('max_depth', -1), 
                                      min_child_samples=self.model_params[model_name].get('min_data_in_leaf', 20), 
                                      num_leaves=self.model_params[model_name].get('max_leaves', 31),
                                      subsample=self.model_params[model_name].get('bagging_fraction', 1.0), 
                                      subsample_freq=self.model_params[model_name].get('bagging_freq', 0), 
                                      colsample_bytree=self.model_params[model_name].get('feature_fraction', 1.0), 
                                      reg_alpha=self.model_params[model_name].get('lambda_l1', 0.0), 
                                      reg_lambda=self.model_params[model_name].get('lambda_l2', 0.0), 
                                      random_state=self.random_seed,
                                      importance_type='gain',
                                      **self.model_params[model_name].get('kwargs', {}))         

            model.fit(df_model_train[self.all_features],
                      df_model_train[[self.target]],
                      sample_weight=weight,
                      eval_set=eval_set,
                      early_stopping_rounds=early_stopping,
                      verbose=False,
                      categorical_feature=self.categorical_features,
                      callbacks=None)

            # Remove eval_key_name level from dictionary
            evals_result = {key: value[eval_key_name] for key, value in model.evals_result_.items()}   
                
        elif model_name.split('_')[0] == 'xgboost':
            if objective == 'mean': 
                objective_xgb = 'reg:squarederror'
                eval_key_name = 'rmse'
            else: 
                raise ValueError("'objective' for xgboost must be 'mean'.")

            model = xgb.XGBRegressor(objective=objective_xgb,
                                         booster=self.model_params[model_name].get('booster', 'gbtree'),
                                         n_estimators=num_rounds,
                                         learning_rate=self.model_params[model_name].get('learning_rate', 0.1), 
                                         max_depth=self.model_params[model_name].get('max_depth', -1), 
                                         min_child_samples=self.model_params[model_name].get('min_data_in_leaf', 20), 
                                         num_leaves=self.model_params[model_name].get('max_leaves', 31),
                                         subsample=self.model_params[model_name].get('bagging_fraction', 1.0), 
                                         colsample_bytree=self.model_params[model_name].get('feature_fraction', 1.0), 
                                         reg_alpha=self.model_params[model_name].get('lambda_l1', 0.0), 
                                         reg_lambda=self.model_params[model_name].get('lambda_l2', 0.0), 
                                         random_state=self.random_seed,
                                         importance_type='gain', 
                                         **self.model_params[model_name]['kwargs'])

            model.fit(df_model_train[self.all_features],
                      df_model_train[[self.target]],
                      sample_weight=weight,
                      eval_set=eval_set,
                      early_stopping_rounds=early_stopping,
                      verbose=False,
                      callbacks=None)

            # Remove eval_key_name level from dictionary
            evals_result = {key: value[eval_key_name] for key, value in model.evals_result_.items()}   

        elif model_name.split('_')[0] == 'catboost':
            if objective == 'mean': 
                objective_cb = 'RMSE'
                eval_key_name = 'RMSE'
            elif objective == 'quantile': 
                objective_cb = 'Quantile:alpha={0:g}'.format(alpha)
                eval_key_name = 'Quantile:alpha={0:g}'.format(alpha)
            else: 
                raise ValueError("'objective' for catboost must be one of ['mean', 'quantile']")

            model = cb.CatBoostRegressor(objective=objective_cb,
                                          boosting_type=self.model_params[model_name].get('boosting_type', 'Plain'),
                                          grow_policy=self.model_params[model_name].get('grow_policy', 'SymmetricTree'),
                                          n_estimators=num_rounds,
                                          learning_rate=self.model_params[model_name].get('learning_rate', 0.1), 
                                          max_depth=self.model_params[model_name].get('max_depth', -1), 
                                          min_data_in_leaf=self.model_params[model_name].get('min_data_in_leaf', 20), 
                                          max_leaves=self.model_params[model_name].get('max_leaves', 31),
                                          subsample=self.model_params[model_name].get('bagging_fraction', 1.0),
                                          colsample_bylevel=self.model_params[model_name].get('feature_fraction', 1.0),
                                          reg_lambda=self.model_params[model_name].get('lambda_l2', 0.0), 
                                          random_state=self.random_seed,
                                          **self.model_params[model_name]['kwargs']) 

            model.fit(df_model_train[self.all_features],
                      df_model_train[[self.target]],
                      sample_weight=weight,
                      eval_set=eval_set, # Catboost already uses train set in eval_set. Therefore, should not be passed here. 
                      early_stopping_rounds=early_stopping,
                      verbose=False,
                      cat_features=self.categorical_features)

            evals_result = {key: value[objective_cb] for key, value in model.evals_result_.items()}

        else: 
            raise ValueError("No supported model detected. Supported models are ['lightgbm', 'xgboost', 'catboost'].")

        return model, evals_result


    def determine_num_rounds(self, df_model_train, model_name, objective='mean', weight=None):
        if self.early_stopping_by_cv.get("enabled", None) == True:
            if model_name.split('_')[0] == 'lightgbm':
                if objective == 'mean': 
                    objective_lgb = 'mean_squared_error'
                    eval_key_name = 'l2'
                elif objective == 'quantile': 
                    objective_lgb = 'quantile'
                    eval_key_name = 'quantile'
                else: 
                    raise ValueError("'objective' for lightgbm must be either 'mean' or 'quantile'")
                                    
                train_set = lgb.Dataset(df_model_train[self.all_features], 
                                        label=df_model_train[self.target], 
                                        weight=weight, 
                                        params={'verbose': -1}, 
                                        free_raw_data=False)
                model_p = self.model_params[model_name].copy()
                if 'kwargs' in model_p:
                    model_p_kwargs = model_p['kwargs']
                    model_p = {**model_p,
                               **model_p_kwargs}
                    del model_p['kwargs']
                cv_metrics = lgb.cv({**model_p,
                                     'objective': objective_lgb,
                                     'verbose': -1},
                                   train_set,
                                   num_boost_round=self.early_stopping_by_cv.get("max_num_rounds", 500),
                                   nfold=self.early_stopping_by_cv.get("nfold", 3),
                                   stratified=self.early_stopping_by_cv.get("stratified", False),
                                   metrics=[eval_key_name],
                                   verbose_eval=-1, 
                                   early_stopping_rounds=self.early_stopping_by_cv.get("early_stopping", 30)
                                   )                
                num_rounds = np.argmin(cv_metrics[f'{eval_key_name}-mean'])
                early_stopping = None
            else:
                raise NotImplementedError()
        else:
            num_rounds = self.model_params[model_name].get('num_trees', 100)
            early_stopping = self.model_params[model_name].get("early_stopping", None)

        return num_rounds, early_stopping



    def train(self, df_model_train, model_name, df_model_valid=None, weight=None): 
        
        model_q, evals_result_q = {}, {}
        if 'mean' in self.regression_params['type']:
            num_rounds, early_stopping = self.determine_num_rounds(df_model_train, model_name, objective='mean', weight=weight)
            # Train model for mean
            model, evals_result = self.create_fit_model(model_name, df_model_train, 
                                            objective='mean', df_model_valid=df_model_valid, 
                                            weight=weight, num_rounds=num_rounds, early_stopping=early_stopping)

            model_q['mean'] = model
            evals_result_q['mean'] = evals_result

        if 'quantile' in self.regression_params['type']:
            num_rounds, early_stopping = self.determine_num_rounds(df_model_train, model_name, objective='quantile', weight=weight)
            
            # Train models for different quantiles
            with joblib.parallel_backend(self.parallel_processing['backend']):
                results = joblib.Parallel(n_jobs=self.parallel_processing['n_workers'])(
                            joblib.delayed(self.create_fit_model)(model_name, 
                                                                  df_model_train,
                                                                  objective='quantile',
                                                                  alpha=alpha,
                                                                  df_model_valid=df_model_valid, 
                                                                  weight=weight,
                                                                  num_rounds=num_rounds, 
                                                                  early_stopping=early_stopping)
                            for alpha in self.alpha_q)

            for (model, evals_result), alpha in zip(results, self.alpha_q):
                model_q['quantile{0:.2f}'.format(alpha)] = model
                evals_result_q['quantile{0:.2f}'.format(alpha)] = evals_result

        if not (('mean' in self.regression_params['type']) or ('quantile' in self.regression_params['type'])):
            raise ValueError('Value of regression parameter "objective" not recognized.')

        # Convert evals_result_q to dataframe
        data = {(level1_key, level2_key): pd.Series(values)
                for level1_key in evals_result_q.keys()
                for level2_key, values in evals_result_q[level1_key].items()}
        df_evals_result_q = pd.DataFrame(data)
        df_evals_result_q.index.name = 'iterations'

        return model_q, df_evals_result_q


    def train_model_split_site(self, dfs_model_train_split_site, dfs_model_valid_split_site=None, weight_train_split_site=None):
        
        print('Training...')
        models_split_site, dfs_evals_result_site_split = [], []
        with tqdm(total=len(dfs_model_train_split_site)*len(dfs_model_train_split_site[0])*len(self.model_params.keys())) as pbar:
            for idx_split, dfs_model_train_site in enumerate(dfs_model_train_split_site):
                models_site, dfs_evals_result_site = [], []
                for idx_site, df_model_train in enumerate(dfs_model_train_site):
                    models, dfs_evals_result = {}, {}
                    for model_name in self.model_params.keys():

                        if dfs_model_valid_split_site is not None: 
                            df_model_valid = dfs_model_valid_split_site[idx_split][idx_site]
                        else:
                            df_model_valid = None

                        if weight_train_split_site is not None: 
                            weight = weight_train_split_site[idx_split][idx_site]
                        else:
                            weight = None
                        
                        model_q, df_evals_result_q = self.train(df_model_train, model_name, df_model_valid=df_model_valid, weight=weight)

                        models[model_name] = model_q
                        dfs_evals_result[model_name] = df_evals_result_q
                        pbar.update(1)

                    models_site.append(models)
                    dfs_evals_result_site.append(dfs_evals_result)
                        
                models_split_site.append(models_site)
                dfs_evals_result_site_split.append(dfs_evals_result_site)
                
        return models_split_site, dfs_evals_result_site_split
    

    def load_models(self, path=None):
        sites = range(len(self.sites))
        splits = range(len(self.datetime_splits['train']))
        model_path = path if path else self.trial_path+'/models/'
        model_files = glob.glob(model_path+'*.txt')
        model_names = list(set([file.split('models_')[1].split('_q_quantile')[0] for file in model_files]))

        models_split_site = []
        with tqdm(total=len(splits)*len(sites)*len(model_names)) as pbar:
            for split in splits:    
                models_site = []
                for site in sites:
                    model_q = {}
                    for model_name in model_names:
                        model_q[model_name] = {}
                        if 'mean' in self.regression_params['type']:
                            file_name = model_path+'models_'+model_name+'_mean_split_{0}_site_{1}.txt'.format(split, site)
                            if model_name == 'lightgbm': 
                                model = lgb.Booster(model_file=file_name)
                            model_q[model_name]['mean'] = model
                        if 'quantile' in self.regression_params['type']:
                            for alpha in self.alpha_q:
                                file_name = model_path+'models_'+model_name+'_q_quantile{0:.2f}_split_{1}_site_{2}.txt'.format(alpha, split, site)
                                if model_name == 'lightgbm': 
                                    model = lgb.Booster(model_file=file_name)
                                elif model_name == 'catboost':
                                    model = cb.CatBoostRegressor().load_model(file_name)
                                model_q[model_name]['quantile{0:.2f}'.format(alpha)] = model
                                
                        pbar.update(1)

                    models_site.append(model_q)
                models_split_site.append(models_site)
                
        return models_split_site


    def predict(self, df_X, model_q, model_name, return_shap=False): 
        # Use trained models to predict multiple quantiles and postprocess the predictions.

        def preprocess(df_X):
            # Preprocess input data. 

            # Keep all timestamps for which zenith <= prescribed value (day timestamps)
            if self.train_only_zenith_angle_below:
                idx_day = df_X['zenith'] <= self.train_only_zenith_angle_below
                # idx_night = df_X['zenith'] > self.train_only_zenith_angle_below
                df_X = df_X[idx_day]

            return df_X

        def postprocess(y_pred_q):
            # Postprocess predictions. 
            # 1) Add back physical forecast if physical is subtracted from target
            # 2) Clip target to min/max values or clearsky forecast
            # 3) Apply quantile postprocessing

            if self.diff_target_with_physical: 
                y_pred_q = y_pred_q+df_X[self.diff_target_with_physical].values.reshape(-1,1)
            
            if not self.regression_params['target_min_max'] == [None, None]: 
                target_min_max = self.regression_params['target_min_max']

                if target_min_max[1] == 'clearsky': 
                    idx_clearsky = y_pred_q > df_X['Clearsky_Forecast'].values
                    y_pred_q[idx_clearsky] = df_X['Clearsky_Forecast'].values[idx_clearsky]
                    
                    if not target_min_max[0] == None:
                        y_pred_q = y_pred_q.clip(min=target_min_max[0], max=None)

                else:
                    y_pred_q = y_pred_q.clip(min=target_min_max[0], max=target_min_max[1])

            if 'quantile_postprocess' in self.regression_params.keys():
                idx_q_start = 1 if 'mean' in self.regression_params['type'] else 0
                if self.regression_params['quantile_postprocess'] == 'none':
                    pass
                elif self.regression_params['quantile_postprocess'] == 'sorting': 
                    # Lazy post-sorting of quantiles
                    y_pred_q[idx_q_start:,:] = np.sort(y_pred_q[idx_q_start:,:], axis=-1)
                elif self.regression_params['quantile_postprocess'] == 'isotonic_regression': 
                    # Isotonic regression
                    regressor = IsotonicRegression()
                    y_pred_q = np.stack([regressor.fit_transform(self.alpha_q, y_pred_q[sample,:]) for sample in range(idx_q_start, y_pred_q.shape[0])])                    

            return y_pred_q

        def create_prediction_dataframe(y_pred_q, index):

            # Make DataFrame to store the predictions in
            columns = []
            if 'mean' in self.regression_params['type']:
                columns.append('mean')

            if 'quantile' in self.regression_params['type']:
                columns.extend(['quantile{0}'.format(int(round(100*alpha))) for alpha in self.alpha_q])
            
            df_y_pred_q = pd.DataFrame(index=index, columns=columns)

            df_y_pred_q.values[:] = y_pred_q

            df_y_pred_q = df_y_pred_q.astype('float64')

            return df_y_pred_q

        df_X = preprocess(df_X)

        # Run prediction loop over all quantiles
        y_pred_q, X_shap_q, y_pred_post_process_q = [], [], []
        for q in model_q.keys():
            
            y_pred = model_q[q].predict(df_X)

            if return_shap: 
                explainer = shap.TreeExplainer(model_q[q])
                X_shap = explainer.shap_values(df_X)
                X_shap_q.append(X_shap)

            y_pred_q.append(y_pred)

        # Convert list to numpy 2D-array
        if return_shap: X_shap_q = np.stack(X_shap_q, axis=-1)
        
        y_pred_q = np.stack(y_pred_q, axis=-1)

        y_pred_post_process_q = postprocess(y_pred_q)
        df_y_pred_q = create_prediction_dataframe(y_pred_post_process_q, df_X.index)

        if return_shap:
            return df_y_pred_q, y_pred_q, y_pred_post_process_q, X_shap_q
        else:
            return df_y_pred_q, y_pred_q, y_pred_post_process_q


    def predict_split_site(self, dfs_X_split_site, model_split_site):
        # Use trained models to predict for their corresponding split
        #TODO reformat so that model name is dict inside the lists. 

        print('Predicting...')
        time.sleep(0.2)
        dfs_y_pred_split_site = []
        with tqdm(total=len(dfs_X_split_site[0])*len(dfs_X_split_site)*len(self.model_params.keys())) as pbar:
            for dfs_X_site, model_site in zip(dfs_X_split_site, model_split_site):
                dfs_y_pred_site = []
                for dfs_X, model_q, in zip(dfs_X_site, model_site):
                    dfs_y_pred_models = {}
                    for model_name in self.model_params.keys(): 
                        df_y_pred_q, _, _ = self.predict(dfs_X, model_q[model_name], model_name)
                        dfs_y_pred_models[model_name] = df_y_pred_q
                        pbar.update(1)

                    dfs_y_pred_site.append(dfs_y_pred_models)

                dfs_y_pred_split_site.append(dfs_y_pred_site)

        return dfs_y_pred_split_site


    def calculate_loss(self, df_y_true, df_y_pred): 

        if 'mean' in self.regression_params['type']:
            y_true = df_y_true[[self.target]].values
            y_pred = df_y_pred[['mean']].values
            loss = (y_pred-y_true)**2
            df_loss_mean = pd.DataFrame(data=loss, index=df_y_pred.index, columns=['mean'])
        else:
            df_loss_mean = None

        if 'quantile' in self.regression_params['type']:
            a = self.alpha_q.reshape(1,-1)
            y_true = df_y_true[[self.target]].values
            y_pred = df_y_pred.filter(regex='quantile').values

            # Pinball loss with nan if true label is nan
            with np.errstate(invalid='ignore'):
                loss = np.where(np.isnan(y_true),
                                np.nan,
                                np.where(y_true < y_pred,
                                        (1-a)*(y_pred-y_true),
                                        a*(y_true-y_pred)))

                df_loss_quantile = pd.DataFrame(data=loss, index=df_y_pred.index, columns=df_y_pred.filter(regex='quantile').columns)
        else:
            df_loss_quantile = None
        
        df_loss = pd.concat([df_loss_mean, df_loss_quantile], axis=1)
    
        return df_loss


    def calculate_loss_split_site(self, dfs_y_pred_split_site, dfs_y_true_split_site):

        print('Calculating loss...')

        dfs_loss_split_site = []
        for dfs_y_pred_site, dfs_y_true_site in zip(dfs_y_pred_split_site, dfs_y_true_split_site):
            dfs_loss_site = []
            for dfs_y_pred, df_y_true in zip(dfs_y_pred_site, dfs_y_true_site):
                dfs_loss = {}
                for model_name in self.model_params.keys():
                    df_y_pred = dfs_y_pred[model_name]
                    df_loss = self.calculate_loss(df_y_true, df_y_pred)
                    dfs_loss[model_name] = df_loss

                dfs_loss_site.append(dfs_loss)

            dfs_loss_split_site.append(dfs_loss_site)
        
        return dfs_loss_split_site


    def calculate_score(self, dfs_loss):

        flatten = lambda l, key: [item[key] for sublist in l for item in sublist]
        score_model = {}
        for model_name in self.model_params.keys():
            score_model[model_name] = pd.concat(flatten(dfs_loss, model_name)).mean().mean()

        return score_model

   
    def create_folders(self): 
        if os.path.exists(self.trial_path):
            shutil.rmtree(self.trial_path)
        os.makedirs(self.trial_path)
        
        if self.save_options['data'] == True:
            os.makedirs(self.trial_path+'/'+'dfs_X_train')
            os.makedirs(self.trial_path+'/'+'dfs_X_valid')
            os.makedirs(self.trial_path+'/'+'dfs_y_train')
            os.makedirs(self.trial_path+'/'+'dfs_y_valid')
        if self.save_options['prediction'] == True:
            os.makedirs(self.trial_path+'/'+'dfs_y_pred_train')
            os.makedirs(self.trial_path+'/'+'dfs_y_pred_valid')
        if self.save_options['model'] == True:
            os.makedirs(self.trial_path+'/'+'models')
        if self.save_options['evals'] == True:
            os.makedirs(self.trial_path+'/'+'dfs_eval_result')        
        if self.save_options['loss'] == True:
            os.makedirs(self.trial_path+'/'+'dfs_loss_train')
            os.makedirs(self.trial_path+'/'+'dfs_loss_valid')


    def save_json(self, file_path=None):
        if file_path == None: 
            file_name_json = '/params_'+self.trial_name+'.json'
            file_path = self.trial_path+file_name_json 
        with open(file_path, 'w') as file:
            json.dump(self.params_json, file, indent=4)


    def save_data_prediction_evals_loss(self, df, key, model_name, split, site): 
        file_name = key+'_'+model_name+'_split_{0}_site_{1}.csv'.format(split, site)
        df.to_csv(self.trial_path+'/'+key+'/'+file_name)


    def save_model(self, model_q, key, model_name, split, site):
        for q in model_q.keys():
            model = model_q[q]
            if model_name.split('_')[0] in ['lightgbm']: 
                file_name = key+'_'+model_name+'_q_'+q+'_split_{0}_site_{1}.txt'.format(split, site)
                model.booster_.save_model(self.trial_path+'/'+key+'/'+file_name)
            if model_name.split('_')[0] in ['xgboost','catboost','skboost','skboosthist']: 
                file_name = key+'_'+model_name+'_q_'+q+'_split_{0}_site_{1}.pkl'.format(split, site)
                with open(self.trial_path+'/'+key+'/'+file_name, 'wb') as f:
                    pickle.dump(model, f)

    def consolidate_csv_sites(self, path=None): 
        # Consolidating all several files with seperate sites into one multiindex (on columns) dataframe. 
        # Not used currently. 
        files = glob.glob(path+'*.csv')
        files = natural_sort(files)

        file_split = [int(file.split('split_')[1].split('_')[0]) for file in files]
        idx_splits = list(set(file_split))

        dfs = []
        for idx_split in idx_splits: 
            # Get all idx in files that corresponds to same split (idx_split)
            idx_sites = [idx for idx, split in enumerate(file_split) if split==idx_split]
            for idx_site in idx_sites: 
                df = pd.read_csv(files[idx_site], index_col=[0,1], header=0)
                dfs.append(df)
                os.remove(files[idx_site])
                
            df_split = pd.concat(dfs, axis=1, keys=idx_sites)
            file_name = files[idx_site].split('/')[-1].split('_site')[0]+'.csv'

            df_split.to_csv(path+file_name)

    def save_result(self, params_json, result_data, result_prediction, result_model, result_evals, result_loss):

        print('Saving results...')
        self.create_folders()
        self.save_json()

        if self.save_options['data'] == True:
            for key in result_data.keys():
                for split in range(len(result_data[key])):
                    df = pd.concat(result_data[key][split], axis=1, keys=self.sites)
                    self.save_data_prediction_evals_loss(df, key, 'data', split, 'all') 
 
        if self.save_options['prediction'] == True:
            for key in result_prediction.keys():
                for split in range(len(result_prediction[key])):
                    for model_name in self.model_params.keys():
                        dfs = [df[model_name] for df in result_prediction[key][split]]
                        df = pd.concat(dfs, axis=1, keys=self.sites)
                        self.save_data_prediction_evals_loss(df, key, model_name, split, 'all')      

        if self.save_options['model'] == True:
            for key in result_model.keys():
                for split in range(len(result_model[key])):
                    for site in range(len(result_model[key][0])):
                        for model_name in self.model_params.keys():
                            model_q = result_model[key][split][site][model_name]
                            self.save_model(model_q, key, model_name, split, site)

        if self.save_options['evals'] == True:
            for key in result_evals.keys():
                for split in range(len(result_evals[key])):
                    for model_name in self.model_params.keys():
                        dfs = [df[model_name] for df in result_evals[key][split]]
                        df = pd.concat(dfs, axis=1, keys=self.sites)
                        self.save_data_prediction_evals_loss(df, key, model_name, split, 'all')      

        if self.save_options['loss'] == True:
            for key in result_loss.keys():
                for split in range(len(result_loss[key])):
                    for model_name in self.model_params.keys():
                        dfs = [df[model_name] for df in result_loss[key][split]]
                        df = pd.concat(dfs, axis=1, keys=self.sites)
                        self.save_data_prediction_evals_loss(df, key, model_name, split, 'all')      

        if self.save_options['overall_score'] == True:
            score_train_model = self.calculate_score(result_loss['dfs_loss_train'])
            score_valid_model = self.calculate_score(result_loss['dfs_loss_valid'])
            file_name = self.path_result+'/trial-scores.txt'

            for model_name in score_train_model.keys():
                if not os.path.exists(file_name):
                    with open(file_name, 'w') as file:
                        file.write('Name: {0}; Comment: {1}; Model: {2}; Train score {3}; valid score {4};\n'.format(self.trial_name, self.trial_comment, model_name, score_train_model[model_name], score_valid_model[model_name]))
                else:
                    with open(file_name, 'a') as file:
                        file.write('Name: {0}; Comment: {1}; Model: {2}; Train score {3}; valid score {4};\n'.format(self.trial_name, self.trial_comment, model_name, score_train_model[model_name], score_valid_model[model_name]))
        else:
            score_train_model = None
            score_valid_model = None
        print('Results saved to: '+self.trial_path)

        return score_train_model, score_valid_model


    def run_pipeline(self, df):
        # Run pipeline sequentially. 

        print('Running trial pipeline for trial: {0}...'.format(self.trial_name))
        print('Number of workers: {0}.'.format(self.parallel_processing['n_workers']))
        
        if df.index.nlevels == 1:
            df.index = df.index.rename('valid_datetime')
            df.loc[:,'ref_datetime'] = df.index[0]
            df = df.set_index('ref_datetime', append=True, drop=True)
            df.index = df.index.reorder_levels(['ref_datetime', 'valid_datetime'])
        elif df.index.nlevels == 2:
            df.index.rename(('ref_datetime', 'valid_datetime'))

        self.splits = self.generate_splits(df)
        dfs_X_train_split_site, dfs_y_train_split_site, dfs_model_train_split_site, weight_train_split_site = self.generate_dataset_split_site(df, split_set='train')
        dfs_X_valid_split_site, dfs_y_valid_split_site, dfs_model_valid_split_site, _ = self.generate_dataset_split_site(df, split_set='valid')

        models_split_site, dfs_eval_result_split_site = self.train_model_split_site(dfs_model_train_split_site, dfs_model_valid_split_site=dfs_model_valid_split_site, weight_train_split_site=weight_train_split_site)

        dfs_y_pred_train_split_site = self.predict_split_site(dfs_X_train_split_site, models_split_site)
        dfs_y_pred_valid_split_site = self.predict_split_site(dfs_X_valid_split_site, models_split_site)

        dfs_loss_train_split_site = self.calculate_loss_split_site(dfs_y_pred_train_split_site, dfs_y_train_split_site)
        dfs_loss_valid_split_site = self.calculate_loss_split_site(dfs_y_pred_valid_split_site, dfs_y_valid_split_site)

        result_data = {'dfs_X_train': dfs_X_train_split_site,
                       'dfs_X_valid': dfs_X_valid_split_site,
                       'dfs_y_train': dfs_y_train_split_site,
                       'dfs_y_valid': dfs_y_valid_split_site}
        result_model = {'models': models_split_site}
        result_evals = {'dfs_eval_result': dfs_eval_result_split_site}
        result_prediction = {'dfs_y_pred_train': dfs_y_pred_train_split_site,
                             'dfs_y_pred_valid': dfs_y_pred_valid_split_site}
        result_loss = {'dfs_loss_train': dfs_loss_train_split_site,
                       'dfs_loss_valid': dfs_loss_valid_split_site}

        score_train_model, score_valid_model = self.save_result(self.params_json, result_data, result_prediction, result_model, result_evals, result_loss)

        return score_train_model, score_valid_model



# Helper functions
def natural_sort(l):
    # https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    l_sorted = sorted(l, key=alphanum_key)

    return l_sorted


if __name__ == '__main__':
    params_path = sys.argv[1]
    with open(params_path, 'r', encoding='utf-8') as file:
        params_json = json.loads(file.read())

    trial = Trial(params_json)
    df = trial.load_data()
    trial.run_pipeline(df)
