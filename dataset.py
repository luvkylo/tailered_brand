#operating system functions
import os
import time
import re
import random
import itertools
from datetime import datetime

#data science, linear algebra functions
import pandas as pd
import numpy as np
import scipy

#data visualization
import seaborn as sbn
import matplotlib.pyplot as plt

#machine learning
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics.cluster import fowlkes_mallows_score

#Importing tqdm for the progress bar
from tqdm import tnrange, tqdm_notebook

os.chdir('C://Users//studentv14//Documents//files_to_transfer/')

product_data=pd.read_csv('product_data.csv', usecols=['productid', 'quantitysold_last12months', 'lasttransactiondate', \
                                                      'color', 'lasttransactiondate', \
                                                      'status', 'availability', 'c_divisionname', \
                                                      'c_subdivname', 'firsttransactiondate', 'totalrevenue', \
                                                      'totaltransactioncount'])
product_data.set_index('productid')
trans_data=pd.read_csv('transaction_data.csv', nrows=10000, usecols=['mastercustomerid', 'salerevenue', 'discount', \
                                                                     'quantity', 'productid'])
cust_data=pd.read_csv('customer_data.csv', \
                      usecols=['mastercustomerid', 'age', 'c_estimatedcurrenthomevalue','c_esthhincomeamountv5'])
                      
#to eliminate extra values in mastercustomerid and prep for trans_data join with customer data
bool_mask = [x.startswith('FTP') for x in trans_data['mastercustomerid']]
trans_data=trans_data[bool_mask]
trans_data['mastercustomerid']=[x[14:] for x in trans_data['mastercustomerid'] if x.startswith('FTP')]

## 'inner' will remove row with no matching customerid or productid
cust_trans_data = trans_data.merge(cust_data, how='inner', left_on='mastercustomerid', right_on='mastercustomerid')

product_data['lasttransactiondate'] = pd.to_datetime(product_data['lasttransactiondate']).dt.date
product_data['firsttransactiondate'] = pd.to_datetime(product_data['firsttransactiondate']).dt.date

purchase_dic = {}
purchase_list = cust_trans_data.groupby(by='mastercustomerid')['productid'].apply(list)
for x in range(0,len(purchase_list)):
    purchase_dic[purchase_list.keys()[x]] = purchase_list[x]
## for each customer
## store all past purchase product id
## find latest purchase
## run through all demographic to find the best five from associate division, subdivision name and color 
## (quantitysold_13_24months, first transaction date, quantitysold_last12months)

cust_purchase = pd.DataFrame(purchase_dic.items(),columns = ["mastercustomerid", "purchase"])
cust_purchase.set_index('mastercustomerid')
cust_purchase['top_lasttransactiondate'] = ''
cust_purchase['top_firsttransactiondate'] = ''
cust_purchase['top_quantitysold_last12months'] = ''
cust_purchase_temp = pd.DataFrame()

for index, row in cust_purchase.iterrows():
    for y in row['purchase']:
        divisionName = product_data[product_data['productid'] == y]['c_divisionname'].item()
        subdivName = product_data[product_data['productid'] == y]['c_subdivname'].item()
        color = product_data[product_data['productid'] == y]['color'].item()
        
        if product_data[product_data['productid'] == y]['c_divisionname'].item() == "SERVICES":
            continue
        
        ## in the market for 1-2 years now
        top5qt24 = pd.concat([product_data[(product_data['status'] == 'Active') & \
                                           (product_data['availability'] == 'available') &\
                                (product_data['c_divisionname'] == divisionName) & \
                                (product_data['lasttransactiondate'] > datetime(2017, 1, 1).date()) & \
                                (product_data['lasttransactiondate'] <= datetime(2018,12, 31).date())], \
                    product_data[(product_data['status'] == 'Active') & (product_data['availability'] == 'available') &\
                                  (product_data['color'] == color) & \
                                  (product_data['lasttransactiondate'] > datetime(2017, 1, 1).date()) & \
                                  (product_data['lasttransactiondate'] <= datetime(2018,12, 31).date())], \
                    product_data[(product_data['status'] == 'Active') & (product_data['availability'] == 'available') &\
                                  (product_data['c_subdivname'] == subdivName) &\
                                  (product_data['lasttransactiondate'] > datetime(2017, 1, 1).date()) & \
                                  (product_data['lasttransactiondate'] <= datetime(2018,12, 31).date())]])\
        .sort_values(by=['lasttransactiondate'], ascending=False)['productid'][0:5]
        top5transdate = pd.concat([product_data[(product_data['status'] == 'Active') & \
                                                (product_data['availability'] == 'available') &\
                                (product_data['c_divisionname'] == divisionName)], \
                    product_data[(product_data['status'] == 'Active') & (product_data['availability'] == 'available') &\
                                  (product_data['color'] == color)], \
                    product_data[(product_data['status'] == 'Active') & (product_data['availability'] == 'available') &\
                                  (product_data['c_subdivname'] == subdivName)]])\
        .sort_values(by=['firsttransactiondate'], ascending=False)['productid'][0:5]
        top5qt12 = pd.concat([product_data[(product_data['status'] == 'Active') & \
                                           (product_data['availability'] == 'available') &\
                                           (product_data['c_divisionname'] == divisionName)], \
                    product_data[(product_data['status'] == 'Active') & (product_data['availability'] == 'available') &\
                                  (product_data['color'] == color)], \
                    product_data[(product_data['status'] == 'Active') & (product_data['availability'] == 'available') &\
                                  (product_data['c_subdivname'] == subdivName)]])\
        .sort_values(by=['quantitysold_last12months'], ascending=False)['productid'][0:5]
        for i in range(0, len(top5qt24)):
            cust_purchase_temp = cust_purchase_temp.append(pd.DataFrame([[cust_purchase.iloc[index, 0], y, top5qt24.values[i], top5transdate.values[i], top5qt12.values[i]]],\
                                              columns=["mastercustomerid", "purchase", 'top_lasttransactiondate',\
                                                      'top_firsttransactiondate', 'top_quantitysold_last12months']), ignore_index=True)
            
    print(index/len(cust_purchase)*100)
cust_purchase = cust_purchase_temp
cust_purchase = cust_purchase.dropna(subset=['top_lasttransactiondate', 'top_firsttransactiondate', 'top_quantitysold_last12months'])

## run through product with similar division/subdivision name and color that is active and available
## if last transaction date is far, suggest discount on the best five items from the 15 items
## store it in the dataframe

## discount association with salerevenue 
## remove outliars
## find the trend
## suggest a discount on the (totalrevenue/totaltransactioncount)

discount = np.corrcoef(trans_data['salerevenue'], trans_data['discount'])[0, 1]
print(discount)

model = LinearRegression()

col_to_use_for_pred = ['discount']
target_col = ['salerevenue']

trans_data = trans_data.dropna(subset=target_col)
trans_data = trans_data[(trans_data['salerevenue'] >= -300) & (trans_data['salerevenue'] <= 700)]

for x in col_to_use_for_pred:
    trans_data[x] = trans_data[x].fillna(trans_data[x].median()) 

# this section of the code is responsible for cross-validation. Set the number of chunks you want to break your data into, and the model will train/test based on each of the chunks.
temp = list(range(5))
for i in list(range(5)): #iterate over the 5 cross validation segments

    # this section breaks up the dataset into a training and test dataset
    temp = list(range(5))
    temp.remove(i)
    test = trans_data[i::5]
    training_setup = [trans_data[temp[0]::5], trans_data[temp[1]::5], trans_data[temp[2]::5], \
                      trans_data[temp[3]::5]]
    training = pd.concat(training_setup)

    fit = model.fit(training[col_to_use_for_pred], training[target_col]) #fit the model
    prediction = fit.predict(test[col_to_use_for_pred]) #make the prediction

print('Feature Weights')

# this code will tell you which columns were significant in making the prediction. One works for linear regression, the other
# for decision tree types of models
for i in range(0, len(test[col_to_use_for_pred].keys())): print(fit.coef_[0][i], test[col_to_use_for_pred].keys()[i])

print(fit.intercept_[0], "intercept")

trans_data['model_discount'] = ''
trans_data['model_salesrevenue'] = ''
for index, row in product_data.iterrows():
    trans_data[trans_data['productid'] == index]['model_discount'] = (product_data.at[index, 'totalrevenue']/product_data.at[index, 'totaltransactioncount'])*discount
    trans_data[trans_data['productid'] == index]['model_salesrevenue'] = fit.intercept_[0] + (trans_data[trans_data['productid'] == index]['model_discount']*fit.coef_[0][0])

trans_data.to_csv('smaller_transaction_data.csv', index=False) 
#create a smaller transaction file, with only meaningful mastercustomerid values (recommended)
cust_data.to_csv('smaller_customer_data.csv', index=False)
product_data.to_csv('smaller_product_data.csv', index=False)
# 3112 rows out of 10,000 cutomers
cust_trans_data.to_csv('smaller_cust_tran_data.csv', index=False)
cust_purchase.to_csv('smaller_cust_purchase.csv', index=False)
