# EnerFlow

## Introduction
`enerflow`, is an open source library for energy and weather forecasting which makes use of gradient boosting decision trees. It considers the forecasting problem as a tabular problem without the  spatio-temporal aspects included in the modeling prior. Instead spatio-temporal features can be included as (lagged) features in the tabular data. The code integrates the following popular gradient boosting implementations:

##### 1) `lightgbm` ([Documentation](https://lightgbm.readthedocs.io/en/latest/), [Paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf))
##### 2) `xgboost` ([Documentation](https://xgboost.readthedocs.io/en/stable/), [Paper](https://arxiv.org/pdf/1603.02754.pdf))
##### 3) `catboost` ([Documentation](https://catboost.ai/en/docs/), [Paper](https://arxiv.org/pdf/1706.09516.pdf))



## Installation

Clone and install the necessary libraries using:

```
git clone git@github.com:enerflow/enerflow.git
pip install -r requirements.txt
```



## How to use

Find use case examples in [examples](https://github.com/enerflow/enerflow/examples) folder



## Forecasting parameters

A description of the tool parameter inputs is provided below:

- `trial_name`: The name of the trial. It will be used as file name to store the results.
- `trial_comment`: User comment about the trial setup.
- `path_result`: Path to where the result will be stored.
- `path_preprocessed_data`: Folder path to the preprocessed input data.
- `filename_preprocessed_data`: File name of the preprocessed input data.
- `datetime_splits`: A dictionary whose keys can be `train`, `valid`, `test`. Each value is a list of lists of datetime splits (start and end time) of the form `[[start_time_1, end_time_1], [start_time_2, end_time_2], ...]`. Strings `start_time` and `end_time` should have the format `YYYY-mm-dd HH:MM:SS` and is assumed to be UTC. 
- `sites`: List of name of the sites to train on. Sites names corresponding to names of columns in preprocessed data.
- `features`: List of features to use for model prediction. Feature names corresponding to names of columns in preprocessed data.
- `feature_lags`: Dictionary of feature-lags pairs `[{feature_1: lags_1}, {feature_2: lags_2}, ...]` where `feature` is a feature from the `features` list and `lags` is a list of lags (non-zero, positive or negative integers) to include as additional `features` in the model.  
- `categorical_features`: List of features to be handled as categorical.
- `target`: String with name of the target variable to forecast. Target name corresponding to name of column in preprocessed data.
- `diff_target_with_physical`: Boolean (`false` or `true`) if to use physical model as base model and learn the residuals with gradient boosting decision tree model.
- `target_smoothing_window`: Default 1. Window to smooth the target variable before training. Smoothing is done with a centered boxcar window. Should be an odd number for window to be centered.
- `regression_params`:
  - `type`: Type of regression. Either `mean` or `quantile`.
  - `alpha_range`: Range of quantiles on the form `[start, stop, step]` creates a list of quantiles through `numpy.arange(start, stop, step)`.
  - `y_min_max`: List with min and max values [`y_min`, `y_max`] to clip model predictions. If `Clearsky_Forecast` is in `features` then it can be used as upper limit by setting `y_max="clearsky"`. Set to `[null, null]` to disable clipping of predictions.
- `model_params`: Gradient boosting algorithm parameters. See algorithms' documentations.
- `weight_params`: Allows to weight recent samples in training data more compared to outdated samples. Applies an exponential decay on the form `weight = (1-weight_end)*numpy.exp(-days/weight_shape)+weight_end`, where `days` are number of days from the most recent sample.
  - `weight_end`: Weight of the most outdated sample. Should be a number in the range [0,1]. Set to 1 to disable sample weighting.
  - `weight_shape`: Shape of the exponential weighting function.
- `save_options`: Dictionary with keys according to below.
  - `data`: Boolean if to save data to result.
  - `prediction`: Boolean if to save predictions to result.
  - `model`: Boolean if to save models to result.
  - `evals`: Boolean if to save evaluations to result.
  - `loss`: Boolean if to save loss to result.
  - `overall_score`: Boolean if to save overall score to result.



## Acknowledgement

