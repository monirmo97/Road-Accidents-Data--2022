import numpy as np
import pandas as pd

def clean_dataset(data):
    # Make a copy of the original data to avoid modifying it in-place
    cleaned_data = data.copy()

    # Drop rows with missing values
    cleaned_data.dropna(inplace=True)

    # Drop duplicate rows
    cleaned_data.drop_duplicates(inplace=True)

    # Drop specified columns
    columns_to_drop = ['status', 'accident_year']
    cleaned_data.drop(columns=columns_to_drop, inplace=True)

    # Dictionary to store column names and their outlier values
    outlier_columns = {
        'vehicle_reference': [61, 8, 9, 227],
        'casualty_reference': [148, 22, 16, 15, 13, 14, 12, 11],
        'sex_of_casualty': [-1, 9],
        'age_band_of_casualty': [-1],
        'pedestrian_location': [3, 2, 7],
        'pedestrian_movement': [6],
        'car_passenger': [-1, 9],
        'bus_or_coach_passenger': [-1, 9],
        'pedestrian_road_maintenance_worker': [-1],
        'casualty_type': [-1, 18]
    }

    # Function to handle outliers and missing values for a specific column
    def handle_outliers_and_missing(column_name, outlier_values, cleaned_data):
        outlier_rows = cleaned_data[column_name].isin(outlier_values)
        cleaned_data = cleaned_data[~outlier_rows]
        return cleaned_data

    # Loop through the dictionary and apply the function for each column
    for column_name, outlier_values in outlier_columns.items():
        cleaned_data = handle_outliers_and_missing(column_name, outlier_values, cleaned_data)

    return cleaned_data


def handle_special_columns(data, special_columns):
    for col in special_columns:
        if col == 'lsoa_of_casualty':
            most_frequent_value = data[col].replace('-1', np.nan).mode().iloc[0]
            replace_probabilities = data[col].replace('-1', np.nan).value_counts(normalize=True).to_dict()
            replace_probabilities[most_frequent_value] = replace_probabilities.get(most_frequent_value, 0) + replace_probabilities.pop('-1', 0)
            replacement = np.random.choice(list(replace_probabilities.keys()), size=data[col].eq('-1').sum(), p=list(replace_probabilities.values()))
            data.loc[data[col] == '-1', col] = replacement
        else:
            most_frequent_value = data[col].replace(-1, np.nan).mode().iloc[0]
            replace_probabilities = data[col].replace(-1, np.nan).value_counts(normalize=True).to_dict()
            replace_probabilities[most_frequent_value] = replace_probabilities.get(most_frequent_value, 0) + replace_probabilities.pop(-1, 0)
            replacement = np.random.choice(list(replace_probabilities.keys()), size=data[col].eq(-1).sum(), p=list(replace_probabilities.values()))
            data.loc[data[col] == -1, col] = replacement
    return data


