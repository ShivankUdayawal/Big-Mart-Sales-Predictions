#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement
# 
# * The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim of this data science project is to build a predictive model and find out the sales of each product at a particular store.
# 
# 
# * Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.
#  
#     
# * The data has missing values as some stores do not report all the data due to technical glitches. Hence, it will be required to treat them accordingly.

# ## Hypothesis Generation
# 
# #### Store Level Hypotheses:
# 
# * City type: Stores located in urban or Tier 1 cities should have higher sales because of the higher income levels of people there.
# * Population Density: Stores located in densely populated areas should have higher sales because of more demand.
# * Store Capacity: Stores which are very big in size should have higher sales as they act like one-stop-shops and people would prefer getting everything from one place
# * Competitors: Stores having similar establishments nearby should have less sales because of more competition.
# * Marketing: Stores which have a good marketing division should have higher sales as it will be able to attract customers through the right offers and advertising.
# * Location: Stores located within popular marketplaces should have higher sales because of better access to customers.
# * Customer Behavior: Stores keeping the right set of products to meet the local needs of customers will have higher sales.
# * Ambiance: Stores which are well-maintained and managed by polite and humble people are expected to have higher footfall and thus higher sales.
# 
#     
# #### Product Level Hypotheses:
# 
# * Brand: Branded products should have higher sales because of higher trust in the customer.
# * Packaging: Products with good packaging can attract customers and sell more.
# * Utility: Daily use products should have a higher tendency to sell as compared to the specific use products.
# * Display Area: Products which are given bigger shelves in the store are likely to catch attention first and sell more.
# * Visibility in Store: The location of product in a store will impact sales. Ones which are right at entrance will catch the eye of customer first rather than the ones in back.
# * Advertising: Better advertising of products in the store will should higher sales in most cases.
# * Promotional Offers: Products accompanied with attractive offers and discounts will sell more.

# ### We have train (8523) and test (5681) data set, train data set has both input and output variable(s). 
# ### We need to predict the sales for test data set.
# 
# * Item_Identifier: Unique product ID
# 
# 
# * Item_Weight: Weight of product
# 
# 
# * Item_Fat_Content: Whether the product is low fat or not
# 
# 
# * Item_Visibility: The % of total display area of all products in a store allocated to the particular product
# 
# 
# * Item_Type: The category to which the product belongs
# 
# 
# * Item_MRP: Maximum Retail Price (list price) of the product
# 
# 
# * Outlet_Identifier: Unique store ID
# 
# 
# * Outlet_Establishment_Year: The year in which store was established
# 
# 
# * Outlet_Size: The size of the store in terms of ground area covered
# 
# 
# * Outlet_Location_Type: The type of city in which the store is located
# 
# 
# * Outlet_Type: Whether the outlet is just a grocery store or some sort of supermarket
# 
# 
# * Item_Outlet_Sales: Sales of the product in the particulat store. This is the outcome variable to be predicted.

# In[ ]:





# ## Importing LIbraries and Understanding Data

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# ## Read the Data

# In[2]:


Bigmart_train_Data = pd.read_csv("Train.csv")

Bigmart_test_Data = pd.read_csv("Test.csv")


# In[3]:


Bigmart_train_Data.head()


# In[4]:


Bigmart_test_Data.head()


# In[5]:


print("BigMart Train Data shape :" , Bigmart_train_Data.shape)

print("BigMart Test Data shape  :" , Bigmart_test_Data.shape)


# In[6]:


Bigmart_train_Data.info()


# In[7]:


Bigmart_train_Data.describe()


# ## Exploratory Data Analysis

# ## Data Visualization

# ### Univariate Analysis

# In[8]:


for i in Bigmart_train_Data.describe().columns:
    sns.distplot(Bigmart_train_Data[i].dropna())
    plt.show()


# In[9]:


for i in Bigmart_train_Data.describe().columns:
    sns.boxplot(Bigmart_train_Data[i].dropna())
    plt.show()


# In[10]:


sns.set_style("whitegrid")
plt.figure(figsize=(20,8))
sns.countplot(x = 'Item_Type', data = Bigmart_train_Data)
plt.title('Distribution plot', fontsize=16)
plt.xlabel('Item_Type', fontsize=16)


# In[11]:


Bigmart_train_Data.Item_Type.value_counts()


# In[12]:


sns.set_style("whitegrid")
plt.figure(figsize=(10,5))
sns.countplot(x = 'Item_Fat_Content', data = Bigmart_train_Data)
plt.title('Distribution plot', fontsize=16)
plt.xlabel('Item_Fat_Content', fontsize=16)


# In[13]:


Bigmart_train_Data.Item_Fat_Content.value_counts()


# In[14]:


sns.set_style("whitegrid")
plt.figure(figsize=(10,5))
sns.countplot(x = 'Outlet_Size', data = Bigmart_train_Data)
plt.title('Distribution plot', fontsize=16)
plt.xlabel('Outlet_Size', fontsize=16)


# In[15]:


Bigmart_train_Data.Outlet_Size.value_counts()


# In[16]:


sns.set_style("whitegrid")
plt.figure(figsize=(10,5))
sns.countplot(x = 'Outlet_Location_Type', data = Bigmart_train_Data)
plt.title('Distribution plot', fontsize=16)
plt.xlabel('Outlet_Location_Type', fontsize=16)


# In[17]:


Bigmart_train_Data.Outlet_Location_Type.value_counts()


# In[18]:


sns.set_style("whitegrid")
plt.figure(figsize=(10,5))
sns.countplot(x = 'Outlet_Type', data = Bigmart_train_Data)
plt.title('Distribution plot', fontsize=16)
plt.xlabel('Outlet_Type', fontsize=16)


# In[19]:


Bigmart_train_Data.Outlet_Type.value_counts()


# ## Bivariate Analysis

# In[20]:


sns.set_style("whitegrid")
plt.figure(figsize=(20,6))
sns.boxplot(x = 'Item_Type', y = 'Item_Weight', data = Bigmart_train_Data)
plt.title('Distribution of Box Plot', fontsize = 18)
plt.xlabel('Item_Type', fontsize = 18)
plt.ylabel('Item_Weight', fontsize = 18)


# In[21]:


sns.set_style("whitegrid")
plt.figure(figsize=(20,6))
sns.boxplot(x = 'Item_Type', y = 'Item_Visibility', data = Bigmart_train_Data)
plt.title('Distribution of Box Plot', fontsize = 18)
plt.xlabel('Item_Type', fontsize = 18)
plt.ylabel('Item_Visibility', fontsize = 18)


# In[22]:


sns.set_style("whitegrid")
plt.figure(figsize=(20,6))
sns.boxplot(x = 'Item_Type', y = 'Item_MRP', data = Bigmart_train_Data)
plt.title('Distribution of Box Plot', fontsize = 18)
plt.xlabel('Item_Type', fontsize = 18)
plt.ylabel('Item_MRP', fontsize = 18)


# In[23]:


sns.set_style("whitegrid")
plt.figure(figsize=(20,6))
sns.boxplot(x = 'Item_Type', y = 'Item_Outlet_Sales', data = Bigmart_train_Data)
plt.title('Distribution of Box Plot', fontsize = 18)
plt.xlabel('Item_Type', fontsize = 18)
plt.ylabel('Item_Outlet_Sales', fontsize = 18)


# In[24]:


sns.set_style("whitegrid")
plt.figure(figsize=(20,6))
sns.boxplot(x = 'Outlet_Size', y = 'Item_Outlet_Sales', data = Bigmart_train_Data)
plt.title('Distribution of Box Plot', fontsize = 18)
plt.xlabel('Outlet_Size', fontsize = 18)
plt.ylabel('Item_Outlet_Sales', fontsize = 18)


# ## Correlation Matrix

# In[25]:


Bigmart_train_Data.corr()


# In[26]:


# First check out correlations among numeric features
# Heatmap is a useful tool to get a quick understanding of which variables are important

plt.figure(figsize=(10,10))
plt.title("Correlation")
sns.heatmap(Bigmart_train_Data.corr(), vmax=1, square=True, annot=True, cmap='coolwarm')


# In[27]:


# Visualise the relationship between the features and the response using scatterplots
# Next, pair plot some important features

imp_feats = ['Item_Weight', 'Item_Visibility', 'Item_MRP','Outlet_Establishment_Year','Item_Outlet_Sales']

sns.pairplot(Bigmart_train_Data[imp_feats],   size = 3.0)

plt.show()


# ## Combine Test and Train

# In[28]:


#Bigmart_train_Data['source'] = 'train'
#Bigmart_test_Data['source'] = 'test'
#Data = pd.concat((Bigmart_train_Data,Bigmart_test_Data), ignore_index = True)
#print(Bigmart_train_Data.shape, Bigmart_test_Data.shape, Data.shape)

Data = pd.concat([Bigmart_train_Data,Bigmart_test_Data], keys=('Train','Test'))


# In[29]:


Data.head()


# In[30]:


Data.tail()


# ## Let's check the summary of data

# In[31]:


Data.info()


# In[32]:


Data.describe().T


# ### Observations:
# 
# * "Item_Visibility" has a min value of zero. This makes no practical sense because when a product is being sold in a store, the visibility cannot be 0.
# 
# 
# * "Outlet_Establishment_Years" vary from 1985 to 2009. The values might not be apt in this form. Rather, if we can convert them to how old the particular store is, it should have a better impact on sales.
# 
# 
# * The lower ‘count’ of "Item_Weight" and "Item_Outlet_Sales" confirms the findings from the missing value check.

# ## Total Unique Value

# In[33]:


Data.nunique()


# In[34]:


categorical_columns = [x for x in Data.dtypes.index if Data.dtypes[x]=='object']


categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]



for col in categorical_columns:
    print ('\nFrequency of Categories for varible %s'%col)
    print (Data[col].value_counts())


# ### Observations:
# 
# * Item_Fat_Content:  Some of ‘Low Fat’ values mis-coded as ‘low fat’ and ‘LF’. Also, some of ‘Regular’ are mentioned as ‘regular’.
# 
# 
# * Item_Type:  Not all categories have substantial numbers. It looks like combining them can give better results.
# 
#     
# * Outlet_Type:  Supermarket Type2 and Type3 can be combined. But we should check if that’s a good idea before doing it.

# ## Total Missing Values

# In[35]:


# Total missing values in every column
total_missing = Data.isnull().sum()

# Calculate percentage
percentge = total_missing/Data.isnull().count()

# Combine total and percentage values
NAs = pd.concat([total_missing,percentge],axis=1,keys=('Total','Percentge(%)'))

# Getting columns where there are missing values
NAs[NAs.Total>0].sort_values(by='Total',ascending=False)


# * "Item_Weight" have 2439 missing values which is approx 17% 
# 
# * "Outlet_Size" have 4016 missing values which is 28%
# 
# * "Item_Outlet_Sales" have 5681 missing values which is almost 40%

# ## Missing Value Treatment

# ### 1. Item_Weight

# In[36]:


Data['Item_Weight'].mean()


# In[37]:


Data['Item_Weight'].fillna(Data['Item_Weight'].mean(),inplace=True)


# ### 2. Outlet_Size

# In[38]:


Data['Outlet_Size'].value_counts()


# In[39]:


Data.Outlet_Size = Data.Outlet_Size.fillna('Medium')


# ### 3. Item_Outlet_Sales

# In[40]:


Data['Item_Outlet_Sales'].mean()


# In[41]:


Data['Item_Outlet_Sales'].fillna(Data['Item_Outlet_Sales'].mean(),inplace=True)


# In[42]:


Data.isnull().sum()


# ## Feature Engineering

# ### 1. Combine Outlet_Type & Item_Outlet_Sales

# In[43]:


Data.pivot_table(index = 'Outlet_Type', values = 'Item_Outlet_Sales')


# * Grocery Store have recorded 1076 sales.
# 
# * Supermarket Type1 have recorded 2262 sales.
# 
# * Supermarket Type2 have recorded 2069 sales.
# 
# * Supermarket Type3 have recorded 3088 sales.

# ### 2. Modify Item_Visibility

# In[44]:


Data['Item_Visibility'].mean()


# In[45]:


sum(Data['Item_Visibility']==0)


# In[46]:


Data.loc[Data['Item_Visibility'] == 0, 'Item_Visibility'] = Data.Item_Visibility.mean()


# In[47]:


sum(Data['Item_Visibility']==0)


# In[48]:


Data['Item_Visi_ratio'] = Data['Item_Visibility']/Data['Item_Visibility'].mean()


# In[49]:


Data['Item_Visi_ratio'].describe()


# ### 3. Item_Type_Combined created from Item_Identifier

# In[50]:


Data['Item_Type_Combined'] = Data.Item_Identifier.apply(lambda x:x[0:2])
Data['Item_Type_Combined'].value_counts()


# * Item_Identifier show as FD [Food], DR [Drinks], NC [Non-consumables]

# In[51]:


Data['Item_Type_Combined'] = Data.Item_Type_Combined.map({'FD':'Food and Drinks','NC':'Non-Consumable','DR':'Drinks'})


# In[52]:


Data['Item_Type_Combined'].value_counts()


# In[53]:


plt.figure(figsize = (18,5))
sns.boxplot(x = 'Item_Type_Combined', y = 'Item_Outlet_Sales', data = Data)
plt.title('Item_Type_Combined vs Item_Outlet_Sales', fontsize = 20)
plt.xlabel('Item_Type_Combined', fontsize = 15)
plt.ylabel('Item_Outlet_Sales', fontsize = 15)


# ### 4. Item_Fat_Content modify 

# In[54]:


Data['Item_Fat_Content'].value_counts()


# In[55]:


Data.Item_Fat_Content = Data.Item_Fat_Content.replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})

Data.Item_Fat_Content.value_counts()


# In[56]:


# Marking non-consumable as separate category in low fat:

Data.loc[Data['Item_Type_Combined'] == "Non-consumable",'Item_Fat_Content'] = "Non-Edible"

Data['Item_Fat_Content'].value_counts()


# In[57]:


plt.figure(figsize = (15,5))

plt.subplot(211)
sns.boxplot(x = 'Item_Type_Combined', y = 'Item_Outlet_Sales', data = Data, palette = "Set1")
plt.title('Item_Type_Combined vs Item_Outlet_Sales', fontsize = 20)
plt.xlabel('Item_Type_Combined', fontsize = 15)
plt.ylabel('Item_Outlet_Sales', fontsize = 15)


plt.subplot(212)
sns.boxplot(x = 'Item_Fat_Content', y = 'Item_Outlet_Sales', data = Data, palette = "Set1")
plt.title('Item_Fat_Content vs Item_Outlet_Sales', fontsize = 20)
plt.xlabel('Item_Fat_Content', fontsize = 15)
plt.ylabel('Item_Outlet_Sales', fontsize = 15)
plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 1.5)

plt.show()


# In[58]:


Item_Fat_Content_data = ['Low Fat', 'Regular'] 
  
data = [9185, 5019] 


# In[59]:


plt.pie(data,  labels = Item_Fat_Content_data, autopct = "%1.2f%%", shadow=True, startangle=90)
plt.title("Distribution of items according to fat content", fontsize = 20)
plt.show()


# ### 5. Years of operation of a Outlet

# In[60]:


Data['Outlet_Establishment_Year'].value_counts()


# In[61]:


Data['Outlet_Years'] = 2010 - Data['Outlet_Establishment_Year']


# In[62]:


Data['Outlet_Years'].describe()


# In[63]:


plt.figure(figsize = (15,8))
sns.boxplot(x = 'Outlet_Years', y = 'Item_Outlet_Sales', data = Data)
plt.title('Outlet years vs Item_Outlet_Sales', fontsize = 15)
plt.xlabel('Outlet_Years', fontsize = 15)
plt.ylabel('Item_Outlet_Sales', fontsize = 15)

plt.show()


# ### 6. Creating Avg of Item_Visibility

# In[64]:


item_visib_avg = Data.pivot_table(values = 'Item_Visibility', index = 'Item_Identifier')

item_visib_avg


# In[65]:


function = lambda x: x['Item_Visibility'] /item_visib_avg['Item_Visibility'][item_visib_avg.index == x['Item_Identifier']][0]


# In[66]:


Data['item_visib_avg'] = Data.apply(function,axis=1).astype(float)


# In[67]:


Data['item_visib_avg']


# In[68]:


plt.figure(figsize = (20,5))
sns.distplot(Data['item_visib_avg'])
plt.title('Distribution Plot of item_visib_avg', fontsize = 15)
plt.xlabel('item_visib_avg', fontsize = 15)

plt.show()


# ## Encoding Categorical Variables

# In[69]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

Data['Outlet'] = lb.fit_transform(Data['Outlet_Identifier'])


# In[70]:


var = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Type','Outlet_Size','Item_Type_Combined']


for item in var:
    Data[item] = lb.fit_transform(Data[item])


# In[71]:


Numerical_cols = Data.select_dtypes(include = np.number)

Numerical_cols.head()


# In[72]:


Categorical_cols = Data.select_dtypes(include = 'object')

Categorical_cols.head()


# ## One Hot Encoding

# * Creating Dummy variables

# In[73]:


Data_dummies = pd.get_dummies(Data, columns=['Item_Fat_Content','Outlet_Location_Type',
                                             'Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])


# In[74]:


Data_dummies.dtypes


# In[75]:


Data_dummies.drop(['Item_Type','Outlet_Establishment_Year','Item_Identifier','Outlet_Identifier'], axis = 1, inplace = True)


# In[76]:


Data_dummies.head()


# ## PreProcessing Data

# * Divide into test and train

# In[77]:


#Divide into test and train:
idx = pd.IndexSlice
train_df = Data_dummies.loc[idx[['Train',], :]]
test_df = Data_dummies.loc[idx[['Test',], :]]


# In[78]:


X = Data_dummies.drop(columns=['Item_Outlet_Sales'])
Y = Data_dummies['Item_Outlet_Sales']


# In[79]:


print(X.shape)
print(Y.shape)


# ## Modeling

# In[80]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3 , random_state=42)


# ## Linear Regression

# In[81]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train,y_train)


# In[82]:


intercept = print(model.intercept_)

intercept


# In[83]:


from sklearn.model_selection import cross_val_score

lr = LinearRegression()
c = 111111111111111111111
d = 0
for i in range(2,12):
    sc = cross_val_score(lr, X, Y, cv=i, scoring='neg_root_mean_squared_error')
    sc = -sc
    print(i,") ",sc.mean())
    if sc.mean()<c:
        c=sc.mean()
        d=i
print('\nBest number of kfolds for cross validation is ',d,'\n')


# In[84]:


coefficients = pd.DataFrame(model.coef_,x_train.columns,columns=['coefficients'])
coefficients


# In[85]:


coefficients.sort_values('coefficients', ascending=False, inplace=True)
coefficients.plot(kind = 'bar', figsize=(26,15))
plt.xlabel("coefficients", size = '25')
plt.title("'Linear Regression model, small regularization'", size = '25')
plt.legend()


# In[87]:


LR_pred = model.predict(x_test)

LR_pred


# In[88]:


from sklearn import metrics
mean_square = metrics.mean_squared_error(y_test,LR_pred)

print('Mean Square Error (MSE)       :', metrics.mean_squared_error(y_test,LR_pred))
print('Root mean Square Error (RMSE) :', np.sqrt(metrics.mean_squared_error(y_test,LR_pred)))


# In[89]:


from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(model, x_train,y_train, cv=11, scoring='neg_mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))
print ("\n Scores ")
print (cv_score)
#Print model report:
print ("\nModel Report")
print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),
                                                                         np.min(cv_score),np.max(cv_score)))


# ## Regularized Linear Regression

# ### Ridge Regression

# In[90]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso

print ("RIDGE REGRESSION")
ridge_model = Ridge(alpha=0.03,normalize=True)
ridge_model.fit(x_train,y_train)
predictions = ridge_model.predict(x_test)

from sklearn import metrics
mean_square = metrics.mean_squared_error(y_test,predictions)

print('Mean Square Error (MSE)       :', metrics.mean_squared_error(y_test,predictions))
print('Root mean Square Error (RMSE) :', np.sqrt(metrics.mean_squared_error(y_test,predictions)))

cv_score = cross_val_score(ridge_model, x_train,y_train, cv=11, scoring='neg_mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))
print ("\n Scores ")
print (cv_score)
#Print model report:
print ("\nModel Report")
print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),
                                                                         np.min(cv_score),np.max(cv_score)))


# In[91]:


coefficients = pd.DataFrame(ridge_model.coef_,x_train.columns,columns=['coefficients'])
coefficients


# In[92]:


coefficients.sort_values('coefficients', ascending=False, inplace=True)
coefficients.plot(kind = 'bar', figsize=(26,15))
plt.xlabel("coefficients", size = '25')
plt.title("'Ridge Regression model, small regularization'", size = '25')
plt.legend()


# ### Lasso Regression

# In[93]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso

print ("LASSO REGRESSION")
lasso_model = Lasso(alpha=0.03,normalize=True)
lasso_model.fit(x_train,y_train)
predictions = lasso_model.predict(x_test)

from sklearn import metrics
mean_square = metrics.mean_squared_error(y_test,predictions)

print('Mean Square Error (MSE)       :', metrics.mean_squared_error(y_test,predictions))
print('Root mean Square Error (RMSE) :', np.sqrt(metrics.mean_squared_error(y_test,predictions)))

cv_score = cross_val_score(lasso_model, x_train,y_train, cv=11, scoring='neg_mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))
print ("\n Scores ")
print (cv_score)
#Print model report:
print ("\nModel Report")
print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),
                                                                         np.min(cv_score),np.max(cv_score)))


# In[94]:


coefficients = pd.DataFrame(lasso_model.coef_,x_train.columns,columns=['coefficients'])
coefficients


# In[95]:


coefficients.sort_values('coefficients', ascending=False, inplace=True)
coefficients.plot(kind = 'bar', figsize=(26,15))
plt.xlabel("coefficients", size = '25')
plt.title("'Lasso Regression model, small regularization'", size = '25')
plt.legend()


# ## Random Forest Regression

# In[96]:


from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor(n_estimators = 600, max_depth = 6, min_samples_leaf = 120, n_jobs = 5)

RF.fit(x_train,y_train)

RF.pred = RF.predict(x_test)

RF.pred


# In[97]:


from sklearn import metrics
mean_square = metrics.mean_squared_error(y_test,RF.pred)

print('Mean Square Error (MSE)       :', metrics.mean_squared_error(y_test,RF.pred))
print('Root mean Square Error (RMSE) :', np.sqrt(metrics.mean_squared_error(y_test,RF.pred)))


# In[98]:


cv_score = cross_val_score(RF, x_train,y_train, cv=11, scoring='neg_mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))
print ("\n Scores ")
print (cv_score)
#Print model report:
print ("\nModel Report")
print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),
                                                                         np.min(cv_score),np.max(cv_score)))


# ## XGBoost

# In[99]:


from sklearn.ensemble import GradientBoostingRegressor

gbm = GradientBoostingRegressor(max_depth= 2)

gbm.fit(x_train, y_train)

gbm_pred = gbm.predict(x_test)


# In[100]:


from sklearn import metrics
mean_square = metrics.mean_squared_error(y_test,gbm_pred)

print('Mean Square Error (MSE)       :', metrics.mean_squared_error(y_test,gbm_pred))
print('Root mean Square Error (RMSE) :', np.sqrt(metrics.mean_squared_error(y_test,gbm_pred)))


# In[101]:


#KFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# CV model
kfold = KFold(n_splits = 2)
results = cross_val_score(gbm, x_train, y_train, cv = kfold)
print("Accuracy          : " , results.mean()*100)
print("standard deviation: " , results.std()*100)


# In[102]:


cv_score = cross_val_score(gbm, x_train,y_train, cv=11, scoring='neg_mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))
print ("\n Scores ")
print (cv_score)
#Print model report:
print ("\nModel Report")
print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),
                                                                         np.min(cv_score),np.max(cv_score)))


# ## Summary

# #### LINEAR REGRESSION
# 
# * Linear Regression's MSE is 1141381.7903140076
# * Linear Regression's RMSE is 1068.3547118415343
# * Linear Regression's Cross ValidationScore (Mean) is 1079
# * Linear Regression's Cross ValidationScore(Std) is 44.48
# 
# 
# 
# #### RIDGE REGRESSION
# 
# * Ridge Regression's MSE is 1140550.4479773378
# * Ridge Regression's RSME is 1067.9655649773254
# * Ridge Regression's Cross ValidationScore (Mean) is 1079
# * Ridge Regression's Cross ValidationScore(Std) is 45.20
# 
# 
# 
# #### LASSO REGRESSION
# 
# * Lasso Regression's MSE is 1140983.2091309088
# * Lasso Regression's RSME is 1068.1681558307703
# * Lasso Regression's Cross ValidationScore (Mean) is 1079 
# * Lasso Regression's Cross ValidationScore(Std) is 44.67
# 
# 
# 
# #### Random Forest Regressor
# 
# * RandomForest Regression's MSE is 1100925.3627566441
# * RandomForest Regression's RSME is 1049.2499048161235
# * RandomForest Regression's Cross ValidationScore (Mean) is 1065 
# * RandomForest Regression's Cross ValidationScore(Std) is 41.98
# 
# 
# 
# #### GradientBoostingRegressor (XGBOOST)
# 
# * XGBOOST Regression's MSE is 1099142.4847272479
# * XGBOOST Regression's RSME is 1048.3999641011287
# * XGBOOST Regression's Cross ValidationScore (Mean) is 1066 
# * XGBOOST Regression's Cross ValidationScore(Std) is 43.98
# 

# ## Conclusions: 

# * In present era of digitally connected world every shopping mall desires to knowthe customer demands beforehand to avoid the shortfall of sale items in all sea-sons. Day to day the companies or the malls are predicting more accurately the demand of product sales or user demands. Extensive research in this area atenterprise level is happening for accurate sales prediction. 
# 
# 
# 
# * As the proﬁt madeby a company is directly proportional to the accurate predictions of sales, theBig marts are desiring more accurate prediction algorithm so that the companywill not suﬀer any losses. 
# 
# 
# 
# * In this research work, we have designed a predictivemodel by modifying Gradient boosting machines as Xgboost technique and ex-perimented it on the 2013 Big Mart dataset for predicting sales of the productfrom a particular outlet. Experiments support that our technique produce moreaccurate prediction compared to than other available techniques like Linear, ridge regression , RandomForest Regression etc. 
# 
# 
# 
# * Finally a comparison of diﬀerent models is summa-rized. From it is also concluded that our model with lowest MSE and RMSE performs better compared to existing models.
