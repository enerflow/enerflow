import json
import pandas as pd
from enerflow import forecast


# Import data
data = pd.read_csv('data.csv', header=0, index_col=0, parse_dates=True)


# Import parameters
params_path = 'parameters.json'
with open(params_path, 'r', encoding='utf-8') as file:
    params = json.loads(file.read())
    

# Add site name in the dataframe to comply with the data input structure
site_name = params['sites'][0]
data.columns = pd.MultiIndex.from_product([[site_name], data.columns])


# Create and train model    
trial = forecast.Trial(params)
score_train_model, score_valid_model = trial.run_pipeline(data)
