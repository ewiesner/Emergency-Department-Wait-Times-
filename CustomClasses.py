import numpy as np
import pandas as pd

class DataCleaner:
    def __init__(self, dataframe):
        self.df = dataframe
    
    def clean_data(self):
        # Apply the condition to set temperature values to NaN
        condition_nan_temp = (self.df['temperature'] > 110) | (self.df['temperature'] < 82.4)
        self.df.loc[condition_nan_temp, 'temperature'] = np.nan
        
        # Apply the condition to convert temperature from Celsius to Fahrenheit
        condition_fahrenheit = (self.df['temperature'] >= 28) & (self.df['temperature'] <= 43.3)
        self.df.loc[condition_fahrenheit, 'temperature'] = self.df['temperature'] * 9/5 + 32
        
        # Apply the condition to scale temperature values by 10 (if between 8.24 and 10.10)
        condition_temp_scale_up = (self.df['temperature'] >= 8.24) & (self.df['temperature'] <= 10.10)
        self.df.loc[condition_temp_scale_up, 'temperature'] *= 10
        
        # Apply the condition to scale temperature values down by 0.10 (if between 824 and 1010)
        condition_temp_scale_down = (self.df['temperature'] >= 824) & (self.df['temperature'] <= 1010)
        self.df.loc[condition_temp_scale_down, 'temperature'] *= 0.10
        
        # Apply the condition to set 'resprate' to NaN if greater than 300
        condition_resprate = self.df['resprate'] > 300
        self.df.loc[condition_resprate, 'resprate'] = np.nan
        
        # Apply the condition to set 'sbp' to NaN if greater than 1000
        condition_sbp = self.df['sbp'] > 1000
        self.df.loc[condition_sbp, 'sbp'] = np.nan
        
        # Apply the condition to set 'dbp' to NaN if greater than 1000
        condition_dbp = self.df['dbp'] > 1000
        self.df.loc[condition_dbp, 'dbp'] = np.nan
    
    def get_dataframe(self):
        return self.df
