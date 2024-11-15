#Importing all packages
#for installing sklearn modules: python -m venv sklearn-env,sklearn-env\Scripts\activate,pip install -U scikit-learn
#For basic packages: pip install pandas,pip install numpy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import pickle


#Loading the cleaned Dataset
data_f=pd.read_csv('C:\\Users\\dasar\\OneDrive\\Desktop\\SQL\\Bangalorehousedata_cleaned.csv')

#Defining Features and Target Variables
X=data_f.drop(columns=['price'])
y=data_f['price'] #this is the target variable
#encoding
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
print("Non-numeric columns:", non_numeric_cols)
X = pd.get_dummies(X, drop_first=True)
#fill missing values
X = X.fillna(X.mean())

#split the data into training and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Defining a list of models to select the best one
models={
    'Linear_Regressiom':LinearRegression(),
    'Decision_Tree':DecisionTreeRegressor(),
    'Random_Forest':RandomForestRegressor(),
    'Gradient_Boosting':GradientBoostingRegressor(),
    'Support_Vector_Regressor':SVR()
}

#Store the results 
model_results={}
for model_name,model in models.items():
    scores=cross_val_score(model,X_train,y_train,cv=5,scoring='neg_mean_squared_error')
    mean_mse=-np.mean(scores)
    model_results[model_name]=mean_mse
    print(f"{model_name}:Mean Squared Error={mean_mse}")

#finding the best model for this case
best_model_name=min(model_results,key=model_results.get)
best_model=models[best_model_name]
print(f"\nBest Model:{best_model_name}")

#Train the best model for full training on the set
best_model.fit(X_train,y_train)
with open('best_model.pk1','wb') as f:
    pickle.dump(best_model,f)