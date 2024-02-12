# function from scikit-learn, which is used to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
# RandomForestClassifier class, which is an ensemble learning method based on decision tree classifiers.
from sklearn.ensemble import RandomForestClassifier
#These imports evaluation metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
# should data.csv replace with actule data 
dff= pd.read_csv('data.csv')
x=dff.iloc[:,:-1] #Extracts the feature columns from the DataFrame
y=dff.iloc[:,-1] #Extracts the target column from the DataFrame 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize and train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")

