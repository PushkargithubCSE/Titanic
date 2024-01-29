import seaborn as sns
import pandas as pd

# Load the Titanic dataset from Seaborn
titanic_data = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print(titanic_data.head())

# Display summary information about the dataset
print(titanic_data.info())

# Check for missing values
print(titanic_data.isnull().sum())
# Handle missing values (e.g., fill missing age values with median)
titanic_data['age'].fillna(titanic_data['age'].median(), inplace=True)

# Convert categorical variables into dummy/indicator variables
titanic_data = pd.get_dummies(titanic_data, columns=['sex', 'embarked'], drop_first=True)

# Select features (independent variables)
X = titanic_data[['age', 'fare', 'pclass', 'sex_male', 'embarked_Q', 'embarked_S']]

# Select target variable (dependent variable)
y = titanic_data['survived']

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

# Create and train the Random Forest classifier model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display classification report
print(classification_report(y_test, y_pred))

# Example: Predict survival for a new passenger
new_passenger_features = [[25, 50, 3, 1, 0, 1]]  # Example features: [age, fare, pclass, sex_male, embarked_Q, embarked_S]
predicted_survival = model.predict(new_passenger_features)
print("Predicted Survival:", predicted_survival)
