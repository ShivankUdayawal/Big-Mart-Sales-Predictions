# Technocolabs
Data Analyst Intern at Technocolabs

## Big Mart Sales Predictions 

### Problem Statement
* The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim of this data science project is to build a predictive model and find out the sales of each product at a particular store.
* Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.
* The data has missing values as some stores do not report all the data due to technical glitches. Hence, it will be required to treat them accordingly.


### Hypothesis Generation
#### Store Level Hypotheses:
* City type: Stores located in urban or Tier 1 cities should have higher sales because of the higher income levels of people there.
* Population Density: Stores located in densely populated areas should have higher sales because of more demand.
* Store Capacity: Stores which are very big in size should have higher sales as they act like one-stop-shops and people would prefer getting everything from one place
* Competitors: Stores having similar establishments nearby should have less sales because of more competition.
* Marketing: Stores which have a good marketing division should have higher sales as it will be able to attract customers through the right offers and advertising.
* Location: Stores located within popular marketplaces should have higher sales because of better access to customers.
* Customer Behavior: Stores keeping the right set of products to meet the local needs of customers will have higher sales.
* Ambiance: Stores which are well-maintained and managed by polite and humble people are expected to have higher footfall and thus higher sales.

#### Product Level Hypotheses:
* Brand: Branded products should have higher sales because of higher trust in the customer.
* Packaging: Products with good packaging can attract customers and sell more.
* Utility: Daily use products should have a higher tendency to sell as compared to the specific use products.
* Display Area: Products which are given bigger shelves in the store are likely to catch attention first and sell more.
* Visibility in Store: The location of product in a store will impact sales. Ones which are right at entrance will catch the eye of customer first rather than the ones in back.
* Advertising: Better advertising of products in the store will should higher sales in most cases.
* Promotional Offers: Products accompanied with attractive offers and discounts will sell more.

### We need to predict the sales for test data set.
* Item_Identifier: Unique product ID
* Item_Weight: Weight of product
* Item_Fat_Content: Whether the product is low fat or not
* Item_Visibility: The % of total display area of all products in a store allocated to the particular product
* Item_Type: The category to which the product belongs
* Item_MRP: Maximum Retail Price (list price) of the product
* Outlet_Identifier: Unique store ID
* Outlet_Establishment_Year: The year in which store was established
* Outlet_Size: The size of the store in terms of ground area covered
* Outlet_Location_Type: The type of city in which the store is located
* Outlet_Type: Whether the outlet is just a grocery store or some sort of supermarket
* Item_Outlet_Sales: Sales of the product in the particulat store. This is the outcome variable to be predicted.
