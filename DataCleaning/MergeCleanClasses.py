import numpy as np
import pandas as pd
import re

class DataMerge:
    def __init__(self, df_edstays, df_triage, df_patients):
        """
        Merge the four dataframes together and calculate the patient age at the adimission time to emergency department.
        :param df_edstays: table from ED module
        :param df_triage: table from ED module
        :param df_patients: table from HOSP module containing the patient's anchor age and anchor year
        :return: a new combined dataframe with the patient's age at the admission time to emergency department
        """ 
        df_patients_cleaned = df_patients[['subject_id', 'anchor_age', 'anchor_year']].drop_duplicates() 
        self.df = pd.merge(df_edstays, df_patients_cleaned, on='subject_id', how='left') 
        self.df['admission_age'] = pd.to_datetime(self.df['intime']).dt.year - self.df['anchor_year'] + self.df['anchor_age'] # calculate age at admission time
        self.df = self.df.drop(columns=['anchor_age', 'anchor_year']) # drop anchor_age and anchor_year and only keep admission_age 
        # Merge these two dataframes along the stay_id column.
        self.df = pd.merge(self.df, df_triage, how='outer', on='stay_id')

        # Eliminate the duplicate and simplify variable name:
        self.df.rename(columns={'subject_id_x': 'subject_id'}, inplace=True)
        self.df.drop('subject_id_y', axis=1, inplace=True)

    
    def get_dataframe(self):
        return self.df
  
class DataCleaner:
    def __init__(self, dataframe):
        self.df = dataframe

    def clean_data(self):
        # If admission age > 91, set to 91
        self.df['admission_age'] = self.df['admission_age'].apply(lambda x:min(x, 91))
        # Here I compute the target variable, stay length, in terms of hours.
        self.df['stay_length']=pd.to_datetime(self.df['outtime'])-pd.to_datetime(self.df['intime'])
        # self.df['stay_length_hours']= self.df['stay_length'].dt.total_seconds() / 3600
        self.df['stay_length_minutes'] = self.df['stay_length'].dt.total_seconds() / 60

        # Now I can drop the 'stay_length' variable.
        self.df.drop('stay_length', axis=1, inplace=True)

        # There are some values for stay_length_hours, which are negative. This is nonsensical. We remove them.
        # We should also remove extreme high values for length of stay, as these are probably inaccurate. Looking at the graph, it seems we have a clear break between the two most extreme outliers and the rest of the data, so I am just cutting out the two high values. 
        # May want to revise this cut-off. Currently it is 300 hours.
        # stay_length_hours_upper_bound = 300
        # self.df = self.df[(self.df['stay_length_hours'] >= 0) & (self.df['stay_length_hours'] <= stay_length_hours_upper_bound)]

        stay_length_minutes_upper_bound = 18000
        self.df = self.df[(self.df['stay_length_minutes'] >= 1) & (self.df['stay_length_minutes'] <= stay_length_minutes_upper_bound)]

        # Apply the condition to set temperature values to NaN
        condition_nan_temp = (self.df['temperature'] > 110) | (self.df['temperature'] < 82.4)
        self.df.loc[condition_nan_temp, 'temperature'] = np.nan
        
        # Apply the condition to convert temperature from Celsius to Fahrenheit
        condition_fahrenheit = (self.df['temperature'] >= 28) & (self.df['temperature'] <= 43.3)
        self.df.loc[condition_fahrenheit, 'temperature'] = self.df['temperature'] * 9/5 + 32
        
        # Apply the condition to scale temperature values up by 10 (if between 8.24 and 10.10)
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
        # Eliminate O2sat above 100. 
        self.df.loc[(self.df['o2sat'] > 100), 'o2sat'] = np.nan

        #This takes all non-numeric entries to NaN.
        # self.df['pain_cleaned']=pd.to_numeric(self.df['pain'], errors='coerce')

        # #This sends numerical entries outside of range to NaN.
        # self.df.loc[~((self.df['pain_cleaned'] <=10) & (self.df['pain_cleaned'] >=0)), 'pain_cleaned'] = np.nan

        # Apply the transformation function to the column
        self.df = self.df.assign(pain_cleaned_advanced=self.df['pain'].apply(self.pain_cleaner))



        # Apply the transformation function to the column
        self.df = self.df.assign(race_condensed=self.df['race'].apply(self.race_cleaner))


        #Also, we are not going to make predictions for when the disposition (that is, means of leaving the hospital) is 'eloped', 'left without being seen', 
        #'left against medical advice'. We also shouldn't use the variable 'disposition' as part of the predictor variable set, so I am going to drop it.
        self.df = self.df[~self.df['disposition'].isin(['ELOPED','LEFT WITHOUT BEING SEEN', 'LEFT AGAINST MEDICAL ADVICE'])]
        self.df = self.df.drop('disposition', axis=1)
        
    def pain_cleaner(self, entry):
        # Check if it's a range (number-number). For example, 6-9. I will replace this range by the average of the endpoints of the range.
        if isinstance(entry, str) and '-' in entry:
            try:
                # Split the string at the hyphen and calculate the average of the two numbers. Round down to 10 if the result is larger than 10.
                num1, num2 = map(int, entry.split('-'))
                num=(num1 + num2) / 2
                if num > 10: #Round down to 10 if the result is larger than 10.
                    num=10
                return num
            except ValueError:
                return np.nan  # If the split values are not integers, return NaN
        
        # Check if it's a number followed by a symbol (e.g., '9+'). In this case, I will strip off the extra symbols and just keep the number.
        match = re.match(r'(\d+(\.\d+)?)\D*$', str(entry))  # Matches a number followed by any non-digit symbol(s)
        if match:
            num=float(match.group(1))  # Return the numeric part as a float. Round down to 10 if the result is larger than 10.
            if num > 10:
                num=10
            return num
        # Check if it's a single number
        try:
            num=float(entry)  # Try converting to a float (covers integer and decimal). Round down to 10 if the result is larger than 10.
            if num > 10:
                num=10
            return num
        except ValueError:
            return np.nan  # If it's not a number, return NaN

    def race_cleaner(self, entry):
        if 'ASIAN' in entry:
            return 'ASIAN'
        elif 'WHITE' in entry:
            return 'White'
        elif 'HISPANIC' in entry:
            return 'HISPANIC/LATINO'
        elif 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER' in entry:
            return 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER'
        elif 'AMERICAN INDIAN/ALASKA NATIVE' in entry:
            return 'AMERICAN INDIAN/ALASKA NATIVE'
        elif 'BLACK' in entry:
            return 'BLACK'
        elif entry in ['OTHER','MULTIPLE RACE/ETHNICITY','SOUTH AMERICAN', 'PORTUGUESE']:
            return 'OTHER'
        else:
            return np.nan

    def get_dataframe(self):
        return self.df
