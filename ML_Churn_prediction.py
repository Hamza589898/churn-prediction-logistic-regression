# import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,roc_curve,classification_report,roc_auc_score

# import csv file
df = pd.read_csv("Churn.csv")

# Data cleaning
df["TotalCharges"] = pd.to_numeric(df['TotalCharges'],errors='coerce')
df = df.fillna(0)
df = df.drop_duplicates()
df = df.drop('customerID',axis=1)

# Apply Min Max scalar
scalar = MinMaxScaler()
df[['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']] = scalar.fit_transform(df[['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']])

# Encode data Frame
enncode_df = pd.get_dummies(df,columns=['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','Churn'])

# Take percentage 
gender_churn = df.groupby(['gender','Churn']).size().reset_index(name="Count")
total_per_gender  = gender_churn.groupby("gender")['Count'].transform('sum')
gender_churn["Percentage"] = (gender_churn['Count']/total_per_gender)*100

# Plot barplot
sns.barplot(data=gender_churn,x='gender',y='Percentage',hue='Churn')
plt.title("Churn & by gender")
plt.show()

# Target values for ML
X = enncode_df.drop(['Churn_Yes','Churn_No'],axis=1)
y = enncode_df['Churn_Yes']

# Train data
X_train,X_test,y_Train,y_Test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

# Logistic Regression
model = LogisticRegression(max_iter=1000,class_weight='balanced')
model.fit(X_train,y_Train)

# Probability for ROC AUC 
y_prob = model.predict_proba(X_test)[:,1]

# Predict value
y_pre = model.predict(X_test)
precision = precision_score(y_Test,y_pre)
f1_scor = f1_score(y_Test,y_pre)
recall = recall_score(y_Test,y_pre)
classification = classification_report(y_Test,y_pre)

# Print accuracy
print("Acuracy : ",accuracy_score(y_Test,y_pre))
print("Confussion Matrics : ",confusion_matrix(y_Test,y_pre))
print("Precision : ",precision)
print("F1,score : ",f1_scor)
print("recall : ",recall)
print("classification_report : ",classification)
