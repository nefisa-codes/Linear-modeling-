#!/usr/bin/env python
# coding: utf-8

# Nefisa Hassen 
# D208 PA task1 

# Nefisa Hassen D208 PA task1
# Part I: Research Question
# A1. What factors influence TotalCharge during a hospital stay for a patient?
# A2. Goal of the Data Analysis:
# The goal of this analysis is to understand the relationships between the dependent variable TotalCharge and independent variables. It is also to identify the factors that influence the total charges patients pay during their hospital stay. The analysis will provide insight into what affects healthcare costs to healthcare administrators.
# Part II: Method Justification
# B1. Summarize the four assumptions of a multiple linear regression model:
# Homogeneity of variance or homoscedasticity: constant variance of the residuals.
# Multicollinearity: two or more of the predictors correlate strongly with each other.
# Normality: normal distribution of the error.
# Linearity: the regression line should represent the points.
# B2. Describe two benefits of using Python for various phases of the analysis: For this analysis, Python will be used. Python has many libraries and packages needed for analysis and offers greater visualization tools.
# B3. Explain why multiple linear regression an appropriate technique is to use for analyzing the research question summarized in Part I:
# The research question is investigating factors contributing to TotalCharge. We have one dependent variable, which is TotalCharge, and we are examining multiple independent variables that can potentially influence the total amount patients pay during their hospital stay. Independent variables include 'VitD_levels,' 'Doc_visits,' 'Full_meals_eaten,' 'vitD_supp,' 'Soft_drink,' 'Initial_admin,' 'HighBlood,' 'Stroke,' 'Complication_risk,' 'Overweight,' 'Arthritis,' 'Diabetes,' 'Hyperlipidemia,' 'BackPain,' 'Anxiety,' 'Allergic_rhinitis,' 'Reflux_esophagitis,' 'Asthma,' 'Services,' and 'Initial_days.'
# Part III: Data Preparation
# C1. On this analysis the file "medical_data" is used. After importing, the data was explored to identify and address any missing values in each variable. There were no missing values detected in any of the columns. Subsequently, the std() function was used to assess outliers. outliers were found in four variables: TotalCharge, Additional_charges, VitD_levels, and Initial_days. The zscore method was applied to treat outliers in all four columns. Also, categorical variables were transformed into numeric values through dummy variables. For categorical variables with more than two options, such as 'Services,' 'Complication_risk,' and 'Initial_admin,' one-hot encoding was implemented to generate numeric values. Lastly, the following variables are not needed to answer the research question and were removed. 'CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 
# 

# In[ ]:





# In[ ]:





# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

import seaborn as sns
from statistics import stdev
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
import statsmodels.api as sm


# In[2]:


med_data = pd.read_csv('medical_clean.csv')


# In[3]:


med_data
 


# In[4]:


med_data.describe()


# In[5]:


med_data.shape


# In[6]:


med_data.columns


# In[7]:


# dropping irevelvant columns. 
med_data = med_data.drop(columns=['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 'State',
       'County', 'Zip', 'Lat', 'Lng', 'Population', 'Area', 'TimeZone', 'Job',
       'Children', 'Age', 'Income', 'Marital', 'Gender','Item1', 'Item2', 'Item3', 'Item4',
       'Item5', 'Item6', 'Item7', 'Item8' ])


# In[8]:


med_data.columns # checking to see of the columns are removed from the file. 


# In[9]:


med_data.isnull().sum() # cheking for missing data


# In[10]:


med_data.info 


# In[11]:


categorical_columns = [
    'ReAdmis', 'VitD_levels', 'Doc_visits', 'Full_meals_eaten', 'vitD_supp',
    'Soft_drink', 'HighBlood', 'Stroke', 'Overweight', 'Arthritis', 'Diabetes',
    'Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis',
    'Reflux_esophagitis', 'Asthma'
]

for column in categorical_columns:
    med_data[column] = med_data[column].astype('category').cat.codes


# In[12]:


med_data.dtypes


# In[13]:


med_data = pd.get_dummies(med_data, columns=['Services','Complication_risk','Initial_admin'], drop_first=True)


# In[15]:


med_data.dtypes


# In[16]:


med_data = med_data.astype(int)  #changing the bool into intiger 


# In[31]:


med_data.dtypes


# In[17]:


# checking for outliers.
med_data.std()


# In[18]:


# as showen above the columuns  inital_days. TotalCharge,AddtionalCharge have outliners and  theses columnes are treated using the Zscare method as below. 


# In[19]:


med_data ['VitD_levels_z']=stats.zscore(med_data['VitD_levels'])


# In[20]:


med_data_outliers_VitD_levels = med_data.query('VitD_levels_z > 3 | VitD_levels_z< -3')


# In[21]:


med_data ['Initial_days_z']=stats.zscore(med_data['Initial_days'])


# In[22]:


med_data_outliers_Initial_days = med_data.query('Initial_days_z > 3 | Initial_days_z< -3')


# In[23]:


med_data ['TotalCharge_z']=stats.zscore(med_data['TotalCharge'])


# In[24]:


med_data_outliers_TotalCharg = med_data.query('TotalCharge_z > 3 | TotalCharge_z< -3')


# In[25]:


med_data['Additional_charges_z'] = stats.zscore(med_data['Additional_charges']) # z varaible


# In[26]:


med_data_outliers_addCharg = med_data.query('Additional_charges_z > 3 | Additional_charges_z < -3') # new dataset created 


# In[28]:


med_data.std() # no more outliners 


# C2. Describe the dependent variable and all independent variables using summary
# statistics
# 
# **The following columns have been excluded from the summary statistics as they are unnecessary for addressing 
# the research question and have been removed in a previous step. I do have a summary statistics code that was 
# executed prior to their deletion. 
# ('CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 'State', 'County', 'Zip', 'Lat', 'Lng', 'Population',
#  'Area', 'TimeZone', 'Job', 'Children', 'Age', 'Income', 'Marital', 'Gender','Item1', 'Item2', 'Item3', 'Item4', 
#  'Item5', 'Item6', 'Item7', 'Item8')
# 
# The data includes 50 variables and 10,000 rows, with TotalCharge as the dependent variable.
# 
# The independent variables used for the analysis are 'ReAdmis', 'VitD_levels', 'Doc_visits', 'Full_meals_eaten',
# 
# 'vitD_supp', 'Soft_drink', 'Initial_admin', 'HighBlood', 'Stroke',
# 
# 'Complication_risk', 'Overweight', 'Arthritis', 'Diabetes',
# 
# 'Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis',
# 
# 'Reflux_esophagitis', 'Asthma', 'Services', 'Initial_days',
# 
# 'TotalCharge', 'Additional_charges'.
# 
# The average patient is 53 years old with 2 children and has approximately a $40,000 yearly income.
# 
# The average doctor visits 5 times a year, and the average total cost is around $5,312, with an average 
# additional charge of $12,934 per patient. The minimum age of a patient is 18 years old, and the maximum is 89 years old.
# The minimum income is $154, the maximum is $207,249. The minimum doctor visit is once a year, and the maximum is 9 times
# a year. Looking at the total charge, on average, a patient pays $5,312, and the maximum is $9,180.

# In[29]:


#Describtibve ananlysis 
med_data.describe()


# In[30]:


med_data.columns #for the following columns, more summary statistics  will be explored below


# In[31]:


med_data['Initial_admin_Observation Admission'].value_counts()


# In[32]:


med_data['Initial_admin_Emergency Admission'].value_counts()


# In[33]:


med_data['Anxiety'].value_counts()


# In[34]:


med_data['Allergic_rhinitis'].value_counts()


# In[35]:


med_data['Reflux_esophagitis'].value_counts()


# In[ ]:





# In[36]:


med_data['Services_CT Scan'].value_counts()


# In[37]:


med_data['Services_Intravenous'].value_counts()


# In[38]:


med_data['Services_MRI'].value_counts()


# In[39]:


med_data['Asthma'].value_counts()


# In[40]:


med_data['Initial_days'].describe()


# In[41]:


med_data['Soft_drink'].value_counts()


# In[42]:


med_data['ReAdmis'].value_counts()


# In[43]:


med_data['VitD_levels'].describe()


# In[44]:


med_data['Full_meals_eaten'].describe()


# In[45]:


med_data['vitD_supp'].value_counts()


# In[46]:


med_data['HighBlood'].value_counts()


# In[47]:


med_data['Stroke'].value_counts()


# In[48]:


med_data['Complication_risk_Medium'].value_counts()
  


# In[49]:


med_data['Complication_risk_Low'].value_counts()


# In[50]:


med_data['Overweight'].value_counts()


# In[51]:


med_data['Arthritis'].value_counts()


# In[52]:


med_data['Diabetes'].value_counts()


# In[53]:


med_data['Hyperlipidemia'].value_counts()


# In[54]:


med_data['BackPain'].value_counts()


# In[55]:


med_data['Additional_charges'].describe()


# In[56]:


med_data['TotalCharge'].describe()


# In[71]:


# C3. Univariate and Bivariate visualizations 


# In[57]:


med_data.info


# In[ ]:





# In[59]:


med_data.columns # checking for new columns such as 'Services_MRI', 'Complication_risk_Low','Complication_risk_Medium', 'Initial_admin_Emergency Admission',
      


# In[60]:


med_data.dtypes #  all the varalbes have numberical varaialbes here. 


# In[61]:


#univariate visualization


# In[62]:


med_data.std()


# In[73]:


sns.countplot(x='ReAdmis', data =med_data)


# In[74]:


sns.boxplot(x='VitD_levels', data =med_data)


# In[75]:


sns.countplot(x='Doc_visits', data =med_data)


# In[76]:


sns.countplot(x='Full_meals_eaten', data =med_data)


# In[77]:


sns.countplot(x='vitD_supp', data =med_data)


# In[78]:


sns.countplot(x='Soft_drink', data =med_data)


# In[79]:


sns.boxplot(x='Initial_days', data =med_data)


# In[80]:


sns.countplot(x= 'HighBlood', data = med_data)


# In[81]:


sns.countplot(x= 'Stroke', data = med_data)


# In[82]:


sns.countplot(x= 'Overweight', data = med_data)


# In[83]:


sns.countplot(x= 'Hyperlipidemia', data = med_data)


# In[84]:


sns.countplot(x= 'Arthritis', data = med_data)


# In[85]:


sns.countplot(x= 'Allergic_rhinitis', data = med_data)


# In[86]:


sns.countplot(x= 'Reflux_esophagitis', data = med_data)


# In[87]:


sns.countplot(x= 'Diabetes', data = med_data)


# In[88]:


sns.countplot(x= 'BackPain', data = med_data)


# In[89]:


sns.countplot(x= 'Anxiety', data = med_data)


# In[90]:


sns.countplot(x= 'Allergic_rhinitis', data = med_data)


# In[91]:


sns.countplot(x= 'Asthma', data = med_data)


# In[92]:


selected_columns = ['Complication_risk_Low', 'Complication_risk_Medium']  
category_counts = med_data[selected_columns].sum()
plt.figure(figsize=(10, 6))  
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.xticks(rotation=45)
plt.xlabel('Categories')
plt.ylabel('Count')
plt.title('Count')
plt.show()


# In[93]:


selected_columns = ['Services_CT Scan', 'Services_Intravenous',
       'Services_MRI',] 

category_counts = med_data[selected_columns].sum()
plt.figure(figsize=(10, 6)) 
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.xticks(rotation=45)
plt.xlabel('Categories')
plt.ylabel('Count')
plt.title('Count')
plt.show()


# In[94]:


selected_columns = ['Initial_admin_Emergency Admission','Initial_admin_Observation Admission'] 
category_counts = med_data[selected_columns].sum()
plt.figure(figsize=(10, 6))  
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.xticks(rotation=45)
plt.xlabel('Categories')
plt.ylabel('Count')
plt.title('Count')
plt.show()


# In[95]:


# bivarent visualization


# In[96]:


sns.boxplot(x=med_data['Asthma'], y=med_data['TotalCharge'])
plt.show()


# In[97]:


sns.boxplot(x=med_data['HighBlood'], y=med_data[ 'TotalCharge'])
plt.show();


# In[98]:


sns.boxplot(x=med_data['Stroke'], y=med_data['TotalCharge'])
plt.show();


# In[99]:


sns.boxplot(x=med_data['Overweight'], y=med_data['TotalCharge'])
plt.show();


# In[100]:


sns.boxplot(x=med_data['Arthritis'], y=med_data['TotalCharge'])
plt.show();


# In[101]:


sns.boxplot(x=med_data['Diabetes'], y=med_data['TotalCharge'])
plt.show();


# In[102]:


sns.boxplot(x=med_data['Hyperlipidemia'], y=med_data['TotalCharge'])
plt.show();


# In[103]:


sns.boxplot(x=med_data['BackPain'], y=med_data['TotalCharge'])
plt.show();


# In[104]:


sns.boxplot(x=med_data['Anxiety'], y=med_data['TotalCharge'])
plt.show();


# In[105]:


sns.boxplot(x=med_data['Allergic_rhinitis'], y=med_data['TotalCharge'])
plt.show();


# In[106]:


sns.boxplot(x=med_data['Reflux_esophagitis'], y=med_data['TotalCharge'])
plt.show();


# In[107]:


#bivarent visualizatio for one hot encoded variavles. 


# In[108]:


sns.boxplot(x=med_data['Complication_risk_Low'], y=med_data['TotalCharge'])
plt.show();


# In[109]:


sns.boxplot(x=med_data['Complication_risk_Medium'], y=med_data['TotalCharge'])
plt.show();


# In[110]:


sns.boxplot(x=med_data['Initial_admin_Emergency Admission'], y=med_data['TotalCharge'])
plt.show();


# In[111]:


sns.boxplot(x=med_data['Initial_admin_Observation Admission'], y=med_data['TotalCharge'])
plt.show();


# In[112]:


sns.boxplot(x=med_data['Services_CT Scan'], y=med_data['TotalCharge'])
plt.show();


# In[113]:


sns.boxplot(x=med_data['Services_Intravenous'], y=med_data['TotalCharge'])
plt.show();


# In[114]:


sns.boxplot(x=med_data['Services_MRI'], y=med_data['TotalCharge'])


# In[121]:


#C4. Describe your data transformation goals 
  


# In[58]:


#C5. Provide the prepared data set as a CSV file.

med_data.to_csv('MSDA208_Task1_prepared_Data.csv')


# # Part IV: Model Comparison and Analysis

# In[108]:


# D1. Initial model 


# In[59]:


y = med_data['TotalCharge']

independent_vars = ['ReAdmis', 'VitD_levels', 'Doc_visits', 'Full_meals_eaten', 
    'vitD_supp','Soft_drink', 'HighBlood', 'Stroke', 'Overweight', 'Arthritis', 
    'Diabetes','Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis',
    'Reflux_esophagitis', 'Asthma','Services_CT Scan',
       'Services_Intravenous', 'Services_MRI', 'Complication_risk_Low',
       'Complication_risk_Medium', 'Initial_admin_Emergency Admission',
       'Initial_admin_Observation Admission']


X = med_data[independent_vars]


# In[60]:


X = sm.add_constant(X)


# In[61]:


model = sm.OLS(y, X)
results = model.fit()


# In[62]:


print(results.summary())


# In[113]:


#D2.Justify a statistically based feature selection procedure or a model evaluation metric to reduce the initial model in a way that aligns with the research question.


# In the initial model, we have a total of 24 independent variables. Using the feature selection method, we will be 
# removing variables that have a p-value of 0.05 or greater because that value is not statistically significant.  
# After removing the variables, there are 13 independent variables that have p-values less than 0.05 which will be 
# used in the reduced mutiple regression model.  

# In[114]:


#D3 a reduced linear regression model that follows the feature selection


#  second model after removing columes with p value of greater than 0.05

# In[63]:


y = med_data['TotalCharge']

independent_vars = ['ReAdmis', 'HighBlood', 'Arthritis', 
    'Diabetes','Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis',
    'Reflux_esophagitis','Services_CT Scan', 'Complication_risk_Low',
       'Complication_risk_Medium', 'Initial_admin_Emergency Admission']


X = med_data[independent_vars]


# In[64]:


X = sm.add_constant(X)


# In[65]:


model = sm.OLS(y, X)
results = model.fit()


# In[66]:


print(results.summary())


# In[124]:


#E1.  Explain your data analysis process by comparing the initial multiple linear regression model and reduced linear regression model, including the following element:


# In[120]:


#E2


# In[67]:


residuals = results.resid


# In[68]:


mse = np.mean(residuals**2)  
sre = np.sqrt(mse) 


# In[69]:


print("Standard Residual Error (SRE):", sre)


# In[70]:


fig=plt.figure(figsize=[16,8])
sm.graphics.plot_regress_exog(results,'ReAdmis', fig=fig);


# In[71]:


fig=plt.figure(figsize=[16,8])
sm.graphics.plot_regress_exog(results,'HighBlood', fig=fig);


# In[136]:


fig=plt.figure(figsize=[16,8])
sm.graphics.plot_regress_exog(results,'Arthritis', fig=fig);


# In[137]:


fig=plt.figure(figsize=[16,8])
sm.graphics.plot_regress_exog(results,'Diabetes', fig=fig);


# In[138]:


fig=plt.figure(figsize=[16,8])
sm.graphics.plot_regress_exog(results,'Hyperlipidemia', fig=fig);


# In[139]:


fig=plt.figure(figsize=[16,8])
sm.graphics.plot_regress_exog(results,'BackPain', fig=fig);


# In[140]:


fig=plt.figure(figsize=[16,8])
sm.graphics.plot_regress_exog(results,'Anxiety', fig=fig);


# In[141]:


fig=plt.figure(figsize=[16,8])
sm.graphics.plot_regress_exog(results,'Allergic_rhinitis', fig=fig);


# In[142]:


fig=plt.figure(figsize=[16,8])
sm.graphics.plot_regress_exog(results,'Reflux_esophagitis', fig=fig);


# In[143]:


fig=plt.figure(figsize=[16,8])
sm.graphics.plot_regress_exog(results,'Services_CT Scan', fig=fig);


# In[144]:


fig=plt.figure(figsize=[16,8])
sm.graphics.plot_regress_exog(results,'Complication_risk_Low', fig=fig);


# In[145]:


fig=plt.figure(figsize=[16,8])
sm.graphics.plot_regress_exog(results,'Complication_risk_Medium', fig=fig);


# In[146]:


fig=plt.figure(figsize=[16,8])
sm.graphics.plot_regress_exog(results,'Initial_admin_Emergency Admission', fig=fig);


# In[72]:


#E3.Provide an executable error-free copy of the code used to support the implementation of the linear regression models using a Python or R file.


med_data.to_csv('MSDA208_PA_task1_E3.cvs')


# In[ ]:





# # Part V: Data Summary and Implications

# In[138]:


#F1. Discuss the results of your data analysis


# y = 3714.1800 +  3808.7579 (ReAdmis)+ 77.2108 (HighBlood)+ 127.0744 (Arthritis )+ 
# 75.5687 (Diabetes)+ 63.2353 (Hyperlipidemia) + 110.9397 (BackPain) + 131.7639Anxiety+ 93.5506 (Allergic_rhinitis)+  93.2684(Reflux_esophagitis)+ ((-83.3542- Services_CT Scan) + (-335.1634 Complication_risk_Low)+ ( -438.5156 Complication_risk_Medium)+  390.5600 (Initial_admin_Emergency Admission*)                                       

# In[144]:


#J


# In[145]:


#I


# In[146]:


#J


# In[ ]:




