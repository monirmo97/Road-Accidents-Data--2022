import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from clean_data import *
from model import *

def feature_engineering(data):
    try:
        # Selecting specified features
        selected_features = ['sex_of_casualty', 'age_band_of_casualty', 'casualty_imd_decile', 'casualty_home_area_type', 'casualty_severity']
        data = data[selected_features]
        return data
    except Exception as error:
        print(f"Error during feature engineering: {error}")
        return None

def main():
    
        path_file = 'data.csv'
        data = pd.read_csv(path_file)
        
        # Call the clean data function
        cleaned_data = clean_dataset(data)
        special_columns = ['casualty_home_area_type', 'casualty_imd_decile', 'lsoa_of_casualty']
        cleaned_data = handle_special_columns(cleaned_data, special_columns)
        
        # Call the feature engineering function
        cleaned_data = feature_engineering(cleaned_data)
        if cleaned_data is None:
            print("Error in feature engineering. Exiting.")
            return
        
        # Select target variable and features
        X = cleaned_data.drop('casualty_severity', axis=1)
        y = cleaned_data['casualty_severity']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Parse command-line arguments for model selection
        parser = argparse.ArgumentParser(description='Train and evaluate different models on road accident data.')
        parser.add_argument('-model_name', type=str, default='gradient_boosting', help='Name of the model (random_forest, gradient_boosting, decision_tree, deep_neural_network)')
        args = parser.parse_args()

        # Train and evaluate the specified model
        if args.model_name == 'random_forest':
            train_evaluate_random_forest(X_train, X_test, y_train, y_test)
        elif args.model_name == 'gradient_boosting':
            train_evaluate_gradient_boosting(X_train, X_test, y_train, y_test)
        elif args.model_name == 'decision_tree':
            train_evaluate_decision_tree(X_train, X_test, y_train, y_test)
        elif args.model_name == 'deep_neural_network':
            train_evaluate_deep_neural_network(X_train, X_test, y_train, y_test)
        else:
            print("Invalid model name. Please choose from random_forest, gradient_boosting, decision_tree, deep_neural_network.")   

if __name__ == "__main__":
    main()

