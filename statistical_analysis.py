import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from clean_data import *
import matplotlib
matplotlib.use('Agg')

def generate_correlation_matrix_heatmap(data, save_folder):
    # Exclude non-numeric columns from the correlation matrix
    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    # Check if there are any numeric columns left
    if not numeric_data.empty:
        correlation_matrix = numeric_data.corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix Heatmap')
        
        # Create 'plots' folder if it doesn't exist
        os.makedirs(save_folder, exist_ok=True)
        
        # Save the plot inside the 'plots' folder
        plt.savefig(os.path.join(save_folder, 'correlation_matrix_heatmap.png'), bbox_inches='tight')
        plt.show()
    else:
        print("No numeric columns found for correlation matrix.")


def plot_casualty_class_distribution(data, save_folder):
    casualty_class = data['casualty_class'].value_counts()
    casualty_labels = {
        1: 'Driver or Rider',
        2: 'Passenger',
        3: 'Pedestrian',
    }

    # Map the values to their corresponding labels for plotting
    casualty_class.index = casualty_class.index.map(casualty_labels)

    # Calculate the percentage of each casualty class
    casualty_percentage = casualty_class / casualty_class.sum() * 100

    # Plot the bar chart with percentages
    plt.figure(figsize=(8, 6))
    casualty_percentage.sort_index().plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('Distribution of Casualty Class')
    plt.xlabel('Casualty Class')
    plt.ylabel('Percentage')
    plt.savefig(os.path.join(save_folder, 'Distribution_of_Casualty_Class.png'))
    plt.close()

def plot_sex_of_casualty_distribution(data, save_folder):
    # Map values to their corresponding labels
    sex_labels = {
        1: 'Male',
        2: 'Female',
    }

    # Map sex_of_casualty values to labels
    data['sex_of_casualty_label'] = data['sex_of_casualty'].map(sex_labels)

    # Calculate the percentage of each sex_of_casualty type
    sex_percentage = data['sex_of_casualty_label'].value_counts(normalize=True) * 100

    # Plotting
    plt.figure(figsize=(10, 6))
    sex_percentage.sort_index().plot(kind='bar', color=['lightcoral', 'lightgreen'])
    plt.title('Distribution of Sex of Casualty')
    plt.xlabel('Sex of Casualty')
    plt.ylabel('Percentage')
    plt.savefig(os.path.join(save_folder, 'Distribution_of_Sex_of_Casualty.png'), bbox_inches='tight')
    plt.close()
    
def plot_age_band_distribution(data, save_folder):
    # Map values to their corresponding labels
    age_band_labels = {
        1: '0-5',
        2: '6-10',
        3: '11-15',
        4: '16-20',
        5: '21-25',
        6: '26-35',
        7: '36-45',
        8: '46-55',
        9: '56-65',
        10: '66-75',
        11: 'Over 75',
    }
    # Map age_band_of_casualty values to labels
    data['age_band_label'] = data['age_band_of_casualty'].map(age_band_labels)
    
    # Calculate the percentage of each age_band_of_casualty type
    age_band_percentage = data['age_band_label'].value_counts(normalize=True) * 100
    
    # Define colors for each age band
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightsalmon', 
            'lightblue', 'lightpink', 'lightgreen', 'lightcyan', 'lightgray', 'lightseagreen']
    # Plotting
    plt.figure(figsize=(12, 6))
    age_band_percentage.sort_index().plot(kind='bar', color=colors)
    plt.title('Distribution of Age Band of Casualty')
    plt.xlabel('Age Band of Casualty')
    plt.ylabel('Percentage')
    plt.savefig(os.path.join(save_folder, 'Distribution_of_age_band.png'), bbox_inches='tight')
    plt.close()

def main():

    file_path = 'data.csv'
    data = pd.read_csv(file_path)
    # Call the clean data function
    cleaned_data = clean_dataset(data)

    save_folder = 'plots'
    # Generate and save correlation matrix heatmap
    generate_correlation_matrix_heatmap(cleaned_data, save_folder)

    # Plot and save casualty class distribution
    plot_casualty_class_distribution(cleaned_data, save_folder)

    # Plot and save sex of casualty distribution
    plot_sex_of_casualty_distribution(cleaned_data, save_folder)
    
    # Plot and save sex of casualty distribution
    plot_age_band_distribution(cleaned_data, save_folder)

if __name__ == "__main__":
    main()
