# Week 1 Project:

## Problem Definition
Question: Imagine you're working with Sprint, one of the biggest telecom companies in the USA. They're really keen on figuring out how many customers might decide to leave them in the coming months. Luckily, they've got a bunch of past data about when customers have left before, as well as info about who these customers are, what they've bought, and other things like that.

So, if you were in charge of predicting customer churn how would you go about using machine learning to make a good guess about which customers might leave? Like, what steps would you take to create a machine learning model that can predict if someone's going to leave or not?

## Proposed Solution

**Understanding the problem statement**

Customer churning is the action of when a customer chooses to abandon their service provider. Firms such as Sprint recognize customer churning as a great loss since they have already invested in attracting these customers. This is one of the major reasons that customer retention is beneficial for a firm. Customers can churn for many reasons and it is hard to pinpoint a general reason for churning. The availability of information has given consumers a bargaining power, and nowadays customers can easily find the service provider, which provides the same product with a more satisfying deal.
To manage this, companies invest in churn prediction methods by trying to predict which of their customers
will churn, so that they can apply preventative measures. Analyzing customer behavior serves as the basis for predicting customers who might churn.

I propose using a machine learning model to predict which customers are most likely to leave in the upcoming months. We need to understand the meaning of customer churning in Sprint company context. For the purpose of these project, I will assume that customer churning means that the customer switches to the competitor, so they no longer use services provided by Sprint. I will assume we are predicting customer churn for the next 6 months. Additionally, we will need to understand what kind of data is available and how much historical data Sprint company has because the predictive model will use the trends from historical data to ake predictions using the current data.

The following steps have been used to provide a solution to the problem of customer churn in order for the firm to take preventative actions in advance.

### Steps to solution

Follow code in __Week1_Project.py__

**1. Data Collection**

The first step for this analysis is data collection and since there is no pre defined data, we shall create our own data set from https://www.mockaroo.com/.
The data has attributes / columns such as customer name, date the customer joined the company , the products they have purchased, date customer left the company (could be null indicating that the customer is still active), attrition which is the response variable and will take the values 0 - for customer left and 1 - for customer is still present, location of customer, Length of time a customer has been active since they first joined the company, payment history which shows if the customer was late for payments of the services or not, gender, age of the customer.

**2. Data Cleaning Analysis and EDA**

The second step after data collection is cleaning and manipulation of the data as well as exploring any patterns present in the data. Explore the relationships between churning rate and other explanatory variables. Convert continuous explanatory variables into categories, check for seasonality.

**3. Feature Engineering and Selection**

Feature engineering is the process of transforming raw data into features that are suitable for machine learning models. Feature Engineering plays an extremely pivotal role in determining the performance of any machine learning model. The steps I would take in the feature engineering phase is imputing missing values using mean / median or mode for continuous variables and encoding the categorical variables to numeric using  dummy variables. Other steps could involve using Principal Component Analysis (PCA) for feature selection, which is the process of convert high dimensional data to low dimensional data by selecting the most important features that capture maximum information about the dataset. You can implement PCA using 
Scikit-Learn library and select the variables that have the highest explained variance.

**4. Model Selection**

Choose the best machine learning model to predict which customers will churn. In this case I will choose Random forests which is a popular supervised machine learning algorithm. Random forests are used for supervised machine learning, where there is a labeled target variable.
Random forests can be used for solving regression (numeric target variable) and classification (categorical target variable) problems.
For a random forest classification problem (like the one we are dealing with), multiple decision trees are created using different random subsets of the data and features. Each decision tree is like an expert, providing its opinion on how to classify the data. Predictions are made by calculating the prediction for each decision tree, then taking the most popular result. 


**4. Model Training and Testing**

Before training the model, there's need for plitting data into training and testing sets. We are expecting there will be some relationship between all the features and the target value, and the model’s job is to learn this relationship during training. When it comes to evaluate the model, we ask it to make predictions on a testing set where it only has access to the features (not the target values) and  wecan compare these predictions to the true value to judge how accurate the model is. Training the data to train and test is done randomly and we retain a seed value to be able to get the same responses for each model training, meaning the results are reproducible.

The next step is figuring out how good the model is. To do this we make predictions on the test features in our case (X_test) and then compare the predictions to the known response values (y_test).
We can calculate an accuracy using the mean average percentage error subtracted from 100 %, however, for unbalanced datasets, accuracy may not be the best evaluation metric to use  since it does not distinguish between the numbers of correctly classified examples of different classes. Hence, it may lead to erroneous conclusions. Considering that a lower percentage (assuming 10% of all customers churn) of customers may leave the company Sprint, then the distribution of the samples in the training dataset across the classes is not equal. We calculate the confusion matrix which enables calculation of other  more reliable evaluation metrices such as f1score to evaluate how good the model is.

In order to quantify the usefulness of all the variables in the entire random forest, we can look at the relative importances of the variables. The variables that have a higher value are better predictors for example if 'payment_history' is at the top of the list, then it could mean that the variable is a good predictore for customer churning, meaning when a customer is late on the bill could indicat possible future churn. We can evaluate the results using real data, because the current data used above is just to showcase the steps to take in creating a model that predicts customer churn.

# Week 2 Project:

## Problem Definition
Let’s say we want to build a model to predict booking prices on Airbnb. Between linear regression and random forest regression, which model would perform better and why?

## Solution 

**Refer to Week2_Project folder for the code**

- Linear regression (LR) model that explains the relationship between a response variable (logprice) and explanatory variables. It assumes there is a linear relationship between the response and explanatory variables. In real life, the relationship between the response variable and explanatory variables may not be linear.

- Random forest (RF) for regression is a supervised learning algorithm and bagging technique that uses an ensemble learning method for regression. RF is a treebased model that is more robust and its able to explain a more complex relationship between the response and explanatory variables.

- Comparing the results above, we see that RF performs better than LR, this is in relation to the metrics as displayed above. The R2 value for RF is greater than the R2 value for LR, meaning that 74% of the variance in the response variable can be explained by the explanatory variables in RF model compared to only 57% explained variance in LR.
This shows that the RF model fits the data much better than the LR model.

- Comparing the other metrics for performance evaluation, we observe that RF model has less errors accross compared to LR. These errors show the difference between the actual proce values and predicted prices. The lower value of MAE, MSE, and RMSE then higher accuracy of a regression model, comparing teh values in the table above, RF has lower error values compared to LR model. For example, RF model has an RMSE value of 0.34 while LR model has an RMSE value of 0.44, meaning the RF model can predict the value of a response variable in absolute terms, with lesser error compared to LR model.

In Conclusion, RF model perfroms better than LR model for prediction of Airbnb prices after comparing the performance metrics.
