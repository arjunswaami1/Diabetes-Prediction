import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, header=None, names=columns)

print("First 5 rows of the dataset:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

print("\nMissing Values in Each Column:")
print(data.isnull().sum())

print("\nHistograms of each feature to understand their distributions:")
data.hist(bins=15, color='steelblue', edgecolor='black', figsize=(15, 10))
plt.suptitle('Histograms of Features')
plt.show()

print("\nBoxplots to detect outliers:")
fig, axs = plt.subplots(2, 4, figsize=(15, 10))
axs = axs.ravel()
for i, col in enumerate(columns[:-1]):
    sns.boxplot(data[col], ax=axs[i], color='skyblue')
    axs[i].set_title(col)
plt.tight_layout()
plt.show()

print("\nCorrelation Matrix:")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
plt.title('Correlation Matrix')
plt.show()

print("\nPairplot to visualize relationships between features and the target:")
sns.pairplot(data, hue='Outcome', palette='coolwarm')
plt.show()

X = data.drop('Outcome', axis=1)
y = data['Outcome']

print("\nSplitting the data into training and testing sets (70% training, 30% testing)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nStandardizing the features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTraining the Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nMaking predictions on the test data...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'\nModel Accuracy: {accuracy:.2f}')

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'], ax=ax)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print("\nROC-AUC Curve:")
y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc='lower right')
plt.show()

print("\nCoefficients of the Logistic Regression Model:")
coefficients = pd.DataFrame(model.coef_[0], index=X.columns, columns=['Coefficient'])
print(coefficients.sort_values(by='Coefficient', ascending=False))

print("\nConclusion:")
print("The Logistic Regression model has been trained and evaluated. The detailed analysis, including feature distributions, outlier detection, and relationships between features, helps in understanding the data better. The classification report, confusion matrix, and ROC-AUC curve provide a comprehensive view of the model's performance.")