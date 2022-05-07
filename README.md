# DUNNHUMBY – THE COMPLETE JOURNEY

Dunnhumby works with a range of sectors, including grocery retail, retail pharmacy, and retailer financial services. The expert teams work alongside their global clients in these sectors, helping them to make the most of their customer data and to put the Customer at the heart of everything they do. This dataset contains household level transactions over two years from a group of 2,500 households who are frequent shoppers at a retailer. It contains all of each household’s purchases, not just those from a limited number of categories. For certain households, demographic information as well as direct marketing contact history are included.

There were different campaigns ran by Dunnhumby by introducing coupons for discounts on various households. Some of them reacted and showed interest, made purchasing and others did not redeem the coupons. In this current situation we have to study the kind of households we are dealing with us and analyze their buying behaviours. A marketing and machine learning based solution should be designed to approach this situation for the growth of our retail stores and help them to maximize profit.


## Goals

### Household Segmentation using RFM modeling

Segregated 2500 households into different groups based on similar characteristics so companies can market to each group effectively and appropriately. Customer segmentation is the process of dividing customers into groups based on common characteristics so companies can market to each group effectively and appropriately. Segmentation allows marketers to better tailor their marketing efforts to various audience subsets. Those efforts can relate to both communications and product development. Customer segmentation requires a company to gather specific information – data – about customers and analyze it to identify patterns that can be used to create segments.

Segregated the customers on :

Can't Loose Them
The customers who have been recent, frequent and have spend a decent amount in the retail store

Champions Spenders
The customers who have spend a huge amount of money and have a decent Frequency and Recency score

Loyal Customers
The customers who has an averge score in all the three parameters such as Frequency, Recency and Monetary Value

Potential Loyalists
The customers who might not have spend much on the retailer but but have an averge recency and frequency score

Needs Attention
The customers who have an average frequency score but their recency score has been pretty low

Almost Lost
Almost Lost are the customers who might have either a decent frequency score but their recency score is poor

Lost Customers
The customers who have a poor recency, frequency and Monetary Value score are the lost customers and there is no point in investing to trying to get them back

<img src="/readme_images/custseg.png"/>

<img src="/readme_images/importance.png"/>

### Churn prediction using transactional data

Using Transactional data we have extracted features to predict if the customer is likely to churn or not. This is a critical need for subscription-based businesses as we can use this knowledge to retain the customers who are likely to churn .

extracted the below features

List of campaigns received by each household
Total number of received campaigns per household
List of campaigns resulted in coupon redemption
The number of redemptions made by each household
Most Frequent Campaign Type (A,B,C) received by each household
Top 20 stores with high number of households which have more high out weeks
Amount of purchase of a household within two years

we trained a Machine Learning model with the training data. The machine learning model is chosen as XGBoost(Extreme Gradient Boosting).We then transformed our categorical variables using one-hot encoding algorithm (get_dummies) to be able to use in our classifier. Then separated the data into train and test 75%/25%.Generated an XGBoost classifier with it's default parameters and trained it with the training set.

Even though accuracy for the test set is 85% it is misleading as our target variable is skewed towards not churned (86% are not churned). Even a very basic model which selects majority class all times would score 86% accuracy.

Therefore we will have to focus on how well our model performs on the minority class (churned households). On the test set we have 201 samples only 28 of them being churned. Our model could not manage to detect any of them, therefore test set recall has come out as 7%. This is the part we need to aim to increase.


### Customer lifetime value prediction

Customer Lifetime Value is a monetary value that represents the amount of revenue or profit a customer will give the company over the period of the relationship.
CLTV demonstrates the implications of acquiring long-term customers compared to short-term customers.
Customer lifetime value (CLTV) can help you to answers the most important questions about sales to every company

Identify the most profitable customers
Formulate how can a company offer the best products and make the most money
Calculate how much budget is needed to spend to acquire customers


Churn rate Calculation

Repeat Rate = Number of Customer who have ordered in last 3 months / Total Number of Customers
Churn Rate = 1 - Repeat Rate

Customer Lifetime Value Calculation

Average Order Value = Total Revenue / Total Number of Orders
Purchase Frequency = Total Number of Orders / Total Number of Customers
CLTV = ((Average Order Value x Purchase Frequency)/Churn Rate) x Profit margin.
Customer Value = Average Order Value * Purchase Frequency

To understand model performance, dividing the dataset into a training set and a test set is a good strategy.
You need to pass 3 parameters features, target, and test_set size. Additionally, you can use random_state as a seed value to maintain reproducibility, which means whenever you split the data will not affect the results. Also, if random_state is None, then random number generator uses np.random for selecting records randomly. It means If you don't set a seed, it is different each time.
First, import the Linear Regression module and create a Linear Regression object. Then, fit your model on the train set using fit() function and perform prediction on the test set using predict() function.

# Streamlit Heroku Deployment

https://adm-final-project-dunnhumby.herokuapp.com


Claat Document 

https://codelabs-preview.appspot.com/?file_id=1gDE84KCJhJERDxyNV0TG5vMm8EGpzkdBF_LmTaSo9d0#1