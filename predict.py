import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Load dataset to get feature names
df = pd.read_csv('../dataset/employees.csv')
features = ['HoursWorked','TasksCompleted','OvertimeHours','LeavesTaken','JobSatisfaction']

# Train simple Decision Tree for demonstration
X = df[features]
y = df['StressLevel']
le = LabelEncoder()
y_enc = le.fit_transform(y)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X, y_enc)

# Example input
new_employee = [45, 10, 3, 1, 4]
X_new = pd.DataFrame([new_employee], columns=features)
pred_num = dt.predict(X_new)
pred_label = le.inverse_transform(pred_num)

print("Predicted Stress Level:", pred_label[0])
