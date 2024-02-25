import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score


# Function to train and evaluate the Random Forest Classifier
def train_evaluate_random_forest(X_train, X_test, y_train, y_test):
    # Initializing the model
    model = RandomForestClassifier(random_state=42)
    # Training the model
    model.fit(X_train, y_train)
    # Making predictions on the test set
    y_pred = model.predict(X_test)
    # Displaying results
    print("Random Forest Classifier:")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    # Retrieving and displaying feature importances
    feature_importances = model.feature_importances_
    display_top_features(X_train.columns, feature_importances)

# Function to train and evaluate the Gradient Boosting Classifier
def train_evaluate_gradient_boosting(X_train, X_test, y_train, y_test):
    # Initializing the model
    model = GradientBoostingClassifier(random_state=42)
    # Training the model
    model.fit(X_train, y_train)
    # Making predictions on the test set
    predictions = model.predict(X_test)
    # Displaying results
    print("Gradient Boosting Classifier:")
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    # Retrieving and displaying feature importances
    feature_importances = model.feature_importances_
    display_top_features(X_train.columns, feature_importances)

# Function to train and evaluate the Decision Tree Classifier
def train_evaluate_decision_tree(X_train, X_test, y_train, y_test):
    # Initializing the model
    model = DecisionTreeClassifier(random_state=42)
    # Training the model
    model.fit(X_train, y_train)
    # Making predictions on the test set
    predictions = model.predict(X_test)
    # Displaying results
    print("Decision Tree Classifier:")
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    # Retrieving and displaying feature importances
    feature_importances = model.feature_importances_
    display_top_features(X_train.columns, feature_importances)

# Function to train and evaluate the Deep Neural Network
def train_evaluate_deep_neural_network(X_train, X_test, y_train, y_test):
    # Scaling the features (required for DNNs)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Initializing the model
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=10, random_state=42)
    # Training the model
    model.fit(X_train_scaled, y_train)
    # Making predictions on the scaled test set
    predictions = model.predict(X_test_scaled)
    # Displaying results
    print("Deep Neural Network:")
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")

# Function to display the top features based on their importances
def display_top_features(feature_names, feature_importances, top_n=5):
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("Top Features:")
    print(importance_df.head(top_n))
