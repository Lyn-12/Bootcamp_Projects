import pandas as pd
import pyforest
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, log_loss, classification_report)
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")
## python3 -m pip install


## import data from csv
df = pd.read_csv('/Users/lynxy/Desktop/Lux_Datascience/Bootcamp_Projects/Project_data.csv', sep = ',')
print(df.columns)
# ['customer_id', 'first_name', 'last_name', 'start_date', 'gender','product', 'attrition', 'end_date', 
#  'location', 'tenure', 'age','payment_history']

### convert some columns to Binary
#0 - for customer left and 1 - for customer is still present,
df['attrition'] = df['attrition'].map({'Female': 1, 'Male': 0})
df['attrition'] = df['attrition'].astype(int) ## convert to integer
df['payment_history'] = df['payment_history'].map({'Female': 1, 'Male': 0})
#print(df.head())

## Explore the rate of churning by gender
gender_churn = df.groupby(['attrition' , 'gender'], as_index=False)['customer_id'].count()
print(gender_churn)

general_churn = df.groupby(['attrition'], as_index=False)['customer_id'].count()
print(general_churn)

## Prepare the data for training
# Empty list to store columns with categorical data
churning = df.copy()
categorical = []
for col, value in churning.iteritems():
    if value.dtype == 'object':
        categorical.append(col)

# Store the numerical columns in a list numerical
numerical = churning.columns.difference(categorical)

attrition_cat = churning[categorical]
attrition_cat = pd.get_dummies(attrition_cat) ## change categorical values to numeric

attrition_num = churning[numerical]
attrition_final = pd.concat([attrition_num, attrition_cat], axis=1).fillna(0)

## Separate your data to explanatory and response variables 
X = attrition_final.drop(['attrition'], axis = 1) 
y = attrition_final[['attrition']]

# Splitting the dataset into the Training set and Test set - test set is 20% of the whole data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

seed = 0   # We set our random seed to zero for reproducibility
# Set Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'max_features': 3,
    'max_depth': 80,
    'min_samples_leaf': 5,
    'random_state' : seed,
    'verbose': 0
}

rf = RandomForestClassifier(**rf_params)
rf.fit(X_train, y_train)

## make predictions for 
rf_predictions = rf.predict(X_test)
print("Accuracy score: {}".format(accuracy_score(y_test, rf_predictions)))
print("="*80)
print(classification_report(y_test, rf_predictions))

## calculate confusion matrix
CM = confusion_matrix(y_test, rf_predictions,labels = [0,1])
#print(CM)
## Plot the confusion Matrix with seaborn
sns.heatmap(CM , annot=True, fmt='d').set_title('Terminated vs continuing confusion matrix (0 = terminated, 1 = continuing)')


def plot_feature_importance(importance,names,model_type):
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(40,80))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Set the font size of xticks
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.title(model_type + ' FEATURE OF IMPORTANCE', fontsize=35)
    plt.xlabel('FEATURE IMPORTANCE', fontsize=30)
    plt.ylabel('FEATURE NAMES', fontsize=30)
    
plot_feature_importance(rf.feature_importances_,X_train.columns,'RANDOM FOREST')   

## save the model
with open('RFclassificationmodel.pkl', 'wb') as model_file:
    pickle.dump(rf, model_file)