#!/usr/bin/env python
# coding: utf-8

# Nefisa Hassen D208 PA Task 2 

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix ,classification_report,ConfusionMatrixDisplay,accuracy_score

from statsmodels.stats.outliers_influence import variance_inflation_factor 


# In[2]:


med_data = pd.read_csv('medical_clean.csv')


# In[ ]:


med_data.columns


# In[ ]:


med_data = med_data.drop(columns=['CaseOrder', 'Customer_id', 'Interaction', 'UID','Job', 'Children', 'Age', 'Income', 'Marital', 'Gender', 'City', 'State',
       'County', 'Zip', 'Lat', 'Lng','Population', 'Area', 'TimeZone' , 'Item1', 'Item2', 'Item3', 'Item4',
       'Item5', 'Item6', 'Item7', 'Item8' ])


# In[ ]:


med_data.columns


# In[ ]:


med_data.dtypes 


# In[ ]:


med_data.isnull().sum() # cheking for missing data


# In[ ]:


categorical_columns = [ 
    'ReAdmis', 'VitD_levels', 'Doc_visits', 'Full_meals_eaten', 'vitD_supp',
    'Soft_drink', 'HighBlood', 'Stroke', 'Overweight', 'Arthritis', 'Diabetes',
    'Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis',
    'Reflux_esophagitis', 'Asthma'
]

for column in categorical_columns:
    med_data[column] = med_data[column].astype('category').cat.codes


# In[ ]:


med_data = pd.get_dummies(med_data, columns=['Services','Complication_risk','Initial_admin'], drop_first=True)


# In[ ]:


# checking for outliers. 
med_data.std()


# In[ ]:


med_data ['TotalCharge_z']=stats.zscore(med_data['TotalCharge'])


# In[ ]:


med_data_outliers_TotalCharge = med_data.query('TotalCharge_z > 3 | TotalCharge_z< -3')


# In[ ]:


med_data ['Additional_charges_z']=stats.zscore(med_data['Additional_charges'])


# In[ ]:


med_data_outliers_Additional_charges  = med_data.query('Additional_charges_z > 3 | Additional_charges_z< -3')


# In[ ]:


med_data ['Initial_days_z'] = stats.zscore(med_data['Initial_days'])


# In[ ]:


med_data_outliers_Initial_days = med_data.query('Initial_days_z > 3 | Initial_days_z< -3')


# In[ ]:


med_data ['VitD_levels_z'] = stats.zscore(med_data['VitD_levels'])


# In[ ]:


med_data_outliers_VitD_levels = med_data.query('VitD_levels_z > 3 | VitD_levels_z< -3')


# In[ ]:


med_data.std() # checking to see if outliers were treated.


# In[ ]:


med_data['VitD_levels'].describe()


# In[ ]:


med_data['Doc_visits'].describe()


# In[ ]:


med_data['Full_meals_eaten'].describe()


# In[ ]:


med_data['vitD_supp'].value_counts()


# In[ ]:


med_data['Initial_admin_Emergency Admission'].value_counts()


# In[ ]:


med_data['Initial_admin_Observation Admission'].value_counts()


# In[ ]:


med_data['Complication_risk_Low'].value_counts()


# In[ ]:


med_data['Complication_risk_Medium'].value_counts()


# In[ ]:


med_data['Services_CT Scan'].value_counts()


# In[ ]:


med_data['Services_Intravenous'].value_counts()


# In[ ]:


med_data['Services_MRI'].value_counts()


# In[ ]:





# In[ ]:


med_data['Overweight'].value_counts()


# In[ ]:


med_data['Arthritis'].value_counts()


# In[ ]:


med_data['Diabetes'].value_counts()


# In[ ]:


med_data['Hyperlipidemia'].value_counts()


# In[ ]:


med_data['BackPain'].value_counts()


# In[ ]:


med_data['Anxiety'].value_counts()


# In[ ]:


med_data['Allergic_rhinitis'].value_counts()


# In[ ]:


med_data['Reflux_esophagitis'].value_counts()


# In[ ]:


med_data['Asthma'].value_counts()


# In[ ]:


med_data['TotalCharge'].describe()


# In[ ]:


med_data['Additional_charges'].describe()


# In[ ]:


med_data['Asthma'].value_counts()


# In[ ]:


#Describtibve ananlysis 
med_data.describe()


# # C3. Univariate and Bivariate visualizations
# before ploting the graphs, catagorical variables will converted to numerical values as follows

# In[ ]:


med_data.dtypes


# In[ ]:


med_data = med_data.astype(int) 


# In[ ]:


med_data.dtypes


# In[ ]:


sns.countplot(x='ReAdmis', data =med_data)


# In[ ]:


sns.countplot(x='VitD_levels', data =med_data)


# In[ ]:


sns.countplot(x='Doc_visits', data =med_data)


# In[ ]:


sns.countplot(x='Full_meals_eaten', data =med_data)


# In[ ]:


sns.countplot(x='vitD_supp', data =med_data)


# In[ ]:


sns.countplot(x='Soft_drink', data =med_data)


# In[ ]:


sns.countplot(x='HighBlood', data =med_data)


# In[ ]:


sns.countplot(x='Stroke', data =med_data)


# In[ ]:


sns.countplot(x='Overweight', data =med_data)


# In[ ]:


sns.countplot(x='Arthritis', data =med_data)


# In[ ]:


sns.countplot(x='BackPain', data =med_data)


# In[ ]:


sns.countplot(x='Hyperlipidemia', data =med_data)


# In[ ]:


sns.countplot(x='Anxiety', data =med_data)


# In[ ]:


sns.countplot(x='Allergic_rhinitis', data =med_data)


# In[ ]:


sns.countplot(x='Reflux_esophagitis', data =med_data)


# In[ ]:


sns.countplot(x='Asthma', data =med_data)


# In[ ]:


sns.countplot(x='Diabetes', data =med_data)


# In[ ]:


med_data.columns


# In[ ]:


sns.countplot(x='Overweight', data =med_data)


# In[ ]:


sns.countplot(x='Initial_days', data =med_data)


# In[ ]:


sns.countplot(x='TotalCharge', data =med_data)


# In[ ]:


sns.countplot(x='Additional_charges', data =med_data)


# In[ ]:


selected_columns = ['Complication_risk_Low', 'Complication_risk_Medium']  
category_counts = med_data[selected_columns].sum()
plt.figure(figsize=(10, 6))  
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.xticks(rotation=45)
plt.xlabel('Categories')
plt.ylabel('Count')
plt.title('Count')
plt.show()


# In[ ]:


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


# In[ ]:


selected_columns = ['Initial_admin_Emergency Admission','Initial_admin_Observation Admission'] 
category_counts = med_data[selected_columns].sum()
plt.figure(figsize=(10, 6))  
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.xticks(rotation=45)
plt.xlabel('Categories')
plt.ylabel('Count')
plt.title('Count')
plt.show()


# In[ ]:


#Barient visiulization


# In[ ]:


sns.histplot(binwidth=0.5, x= "Full_meals_eaten", hue="ReAdmis", data=med_data, stat="count", multiple="stack")


# In[ ]:


sns.histplot(binwidth=0.5, x= "VitD_levels", hue="ReAdmis", data=med_data, stat="count",multiple="stack")


# In[ ]:


sns.histplot(binwidth=0.5, x= "vitD_supp", hue="ReAdmis", data=med_data, stat="count",multiple="stack")


# In[ ]:


sns.histplot(binwidth=0.5, x= "Soft_drink", hue="ReAdmis", data=med_data, stat="count", multiple="stack")


# In[ ]:


sns.histplot(binwidth=0.5, x= "HighBlood", hue="ReAdmis", data=med_data, stat="count", multiple="stack")


# In[ ]:


sns.histplot(binwidth=0.5, x= "Stroke", hue="ReAdmis", data=med_data, stat="count", multiple="stack")


# In[ ]:


sns.histplot(binwidth=0.5, x= "Overweight", hue="ReAdmis", data=med_data, stat="count",multiple="stack")


# In[ ]:


sns.histplot(binwidth=0.5, x= "Arthritis", hue="ReAdmis", data=med_data, stat="count", multiple="stack")


# In[ ]:


sns.histplot(binwidth=0.5, x= "Diabetes", hue="ReAdmis", data=med_data, stat="count", multiple="stack")


# In[ ]:


sns.histplot(binwidth=0.5, x= "Hyperlipidemia", hue="ReAdmis", data=med_data, stat="count",multiple="stack")


# In[ ]:


sns.histplot(binwidth=0.5, x= "BackPain", hue="ReAdmis", data=med_data, stat="count", multiple="stack")


# In[ ]:


sns.histplot(binwidth=0.5, x= "Anxiety", hue="ReAdmis", data=med_data, stat="count", multiple="stack")


# In[ ]:


sns.histplot(binwidth=0.5, x= "Allergic_rhinitis", hue="ReAdmis", data=med_data, stat="count", multiple="stack")


# In[ ]:


sns.histplot(binwidth=0.5, x= "Reflux_esophagitis", hue="ReAdmis", data=med_data, stat="count", multiple="stack")


# In[ ]:


sns.histplot(binwidth=0.5, x= "Asthma", hue="ReAdmis", data=med_data, stat="count", multiple="stack")


# In[ ]:


sns.histplot(binwidth=0.5, x= "Hyperlipidemia", hue="ReAdmis", data=med_data, stat="count", multiple="stack")


# In[ ]:


med_data.boxplot(column ='TotalCharge', by= 'ReAdmis', figsize = ( 5,6))


# In[ ]:


med_data.boxplot(column ='Additional_charges', by= 'ReAdmis', figsize = ( 5,6))


# In[ ]:


sns.histplot(binwidth=0.5, x= "Complication_risk_Low", hue="ReAdmis", data=med_data, stat="count", multiple="stack")
plt.show();


# In[ ]:


sns.histplot(binwidth=0.5, x= "Complication_risk_Medium", hue="ReAdmis", data=med_data, stat="count", multiple="stack")
plt.show();


# In[ ]:


sns.histplot(binwidth=0.5, x= "Services_Intravenous", hue="ReAdmis", data=med_data, stat="count", multiple="stack")
plt.show();



# In[ ]:


sns.histplot(binwidth=0.5, x= "Services_MRI", hue="ReAdmis", data=med_data, stat="count", multiple="stack")
plt.show();




# In[ ]:


sns.histplot(binwidth=0.5, x= "Services_CT Scan", hue="ReAdmis", data=med_data, stat="count", multiple="stack")
plt.show();




# In[ ]:


sns.histplot(binwidth=0.5, x= "Initial_admin_Emergency Admission", hue="ReAdmis", data=med_data, stat="count", multiple="stack")
plt.show();


# In[ ]:


sns.histplot(binwidth=0.5, x= "Initial_admin_Observation Admission", hue="ReAdmis", data=med_data, stat="count", multiple="stack")
plt.show();


# In[ ]:


#C4 Describe your data transformation goals that align with your research question and the steps used totransform the data to achieve the goals, including the annotated code.


# 
# As stated in c2, the data is clean. It has no missing values. The columns with outliners are treated using the z-score
# method. Columns that are irrelevant to the research question are removed.
# A total of 24 independent variables are prepared to be used for the initial model. All the catagorical variables are 
# converted to a numerical value. for catagorical variables with more than two potions(i.e'Services','Complication_risk','Initial_admin'),
# one hot encoding is used to give variables numberical values.

# In[ ]:


#C5. Provide the prepared data set as a CSV file.

med_data.to_csv('MSDA208_Task2_prepared_Data.csv')


# # Part IV: Model Comparison and Analysis

# In[ ]:


# D1. Initial model 


# The initial model has  total of 24 varableis   while the reduced model has  17 varableis. The reduced model is less complex, suggesting a more parsimonious representation.
# Goodness of Fit.
# Both models have high Pseudo R-squared values (around 0.949), indicating a good fit to the data. 
# 
# Log-Likelihood:
# The log-likelihood values  shows that both models have negative log-likelihood values, but the reduced model has a slightly lower value, suggesting a marginal improvement in fit.
# 
# The LLR p-value is 0.000 for both models, indicating that both models provide a statistically significant improvement 
# over the null model (LL-Null).
# 
# The reduced model, despite having fewer predictors, performs similarly in terms of goodness of fit and statistical 
# significance. This suggests that the removed predictors in the reduced model may not contribute significantly to
# explaining the variability in the dependent variable.

# In[ ]:


y = med_data['ReAdmis']

independent_vars = ['VitD_levels',  'Doc_visits',
    'Full_meals_eaten', 'vitD_supp','Soft_drink', 'HighBlood', 'Stroke', 'Overweight', 'Arthritis', 
    'Diabetes','Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis',
    'Reflux_esophagitis', 'Asthma','Services_CT Scan','TotalCharge' ,
       'Services_Intravenous', 'Services_MRI', 'Complication_risk_Low',
       'Complication_risk_Medium', 'Initial_admin_Emergency Admission',
       'Initial_admin_Observation Admission']                           

X = med_data[independent_vars]
X = sm.add_constant(med_data[independent_vars])


# In[ ]:


model = sm.Logit(y, X) 
results = model.fit()



# In[ ]:


print(results.summary())


# In[ ]:


y = med_data['ReAdmis']

 
independent_vars =['VitD_levels', 'Doc_visits', 'Full_meals_eaten', 
    'vitD_supp','Soft_drink', 'HighBlood', 'Stroke', 'Overweight', 'Arthritis', 
    'Diabetes','Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis',
    'Reflux_esophagitis', 'Asthma','Services_CT Scan','TotalCharge' ,
       'Services_Intravenous', 'Services_MRI', 'Complication_risk_Low',
       'Complication_risk_Medium', 'Initial_admin_Emergency Admission',
       'Initial_admin_Observation Admission']


# In[ ]:


X2 = sm.add_constant(med_data[independent_vars])



# In[ ]:


y_pred = results.predict(X2)


# In[ ]:


y = y.astype(int)
y_pred = y_pred.astype(int)


# In[ ]:


print("Unique values in y:", np.unique(y))
print("Unique values in y_pred:", np.unique(y_pred))


# In[ ]:


y_pred_binary = (y_pred > 0.5).astype(int)


# In[ ]:


classification_rep = classification_report(y, y_pred_binary)


# In[ ]:


confusion_matrix1 = confusion_matrix(y, y_pred_binary)
print(confusion_matrix1)


# In[ ]:


print("Confusion Matrix:")
print(confusion_matrix)

print("\nClassification Report:")
print(classification_rep)


# In[ ]:


disp = ConfusionMatrixDisplay(confusion_matrix1, display_labels=['Not Readmitted', 'Readmitted'])
disp.plot(cmap=plt.cm.Blues, values_format='d')

plt.title("Confusion Matrix")
plt.show()


# In[ ]:


#D2.Justify a statistically based feature selection procedure or a model evaluation metric to reduce the initial model in a way that aligns with the research question.


#  using the Backward Elimination variables that are contributing the least to the model 
# are removed.  In this case the variable that has p value  grater than.05  
# Overweight ,Services_Intravenous, VitD_levels,Doc_visits ,Full_meals_eaten ,vitD_supp
# ,Soft_drink.  these varavles will be exlused  for the reduced model. 

# In[ ]:


#D3. reduced model  after   removing the columns. 


# In[ ]:


y = med_data['ReAdmis']

independent_vars = ['HighBlood', 'Stroke', 'Arthritis', 
    'Diabetes','Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis',
    'Reflux_esophagitis', 'Asthma','Services_CT Scan','TotalCharge' ,
       'Services_MRI', 'Complication_risk_Low',
       'Complication_risk_Medium', 'Initial_admin_Emergency Admission',
       'Initial_admin_Observation Admission']


X = med_data[independent_vars]
X = sm.add_constant(med_data[independent_vars])




# In[ ]:


model = sm.Logit(y, X) 
results = model.fit()



# In[ ]:


print(results.summary())


# In[ ]:


y = med_data['ReAdmis']


# In[ ]:


reduced_model_vars  = ['HighBlood', 'Stroke', 'Arthritis', 
   'Diabetes','Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis',
   'Reflux_esophagitis', 'Asthma','Services_CT Scan','TotalCharge' ,
      'Services_MRI', 'Complication_risk_Low',
      'Complication_risk_Medium', 'Initial_admin_Emergency Admission',
      'Initial_admin_Observation Admission']

   
   


# In[ ]:


X3 = sm.add_constant(med_data[reduced_model_vars])


# In[ ]:


y_pred = results.predict(X3)


# In[ ]:


y = y.astype(int)
y_pred = y_pred.astype(int)


# In[ ]:


print("Unique values in y:", np.unique(y))
print("Unique values in y_pred:", np.unique(y_pred))


# In[ ]:


y_pred_binary = (y_pred > 0.5).astype(int)


# In[ ]:


classification_rep = classification_report(y, y_pred_binary)


# In[ ]:


confusion_matrix2 = confusion_matrix(y, y_pred_binary)


# In[ ]:


print(confusion_matrix2)


# In[ ]:


print("Confusion Matrix:")
print(confusion_matrix2)

print("\nClassification Report:")
print(classification_rep)


# In[ ]:


disp = ConfusionMatrixDisplay(confusion_matrix2, display_labels=['Not Readmitted', 'Readmitted'])
disp.plot(cmap=plt.cm.Blues, values_format='d')

plt.title("Confusion Matrix")
plt.show()


# In[ ]:


accuracy = accuracy_score(y, y_pred_binary)


# In[ ]:


print("Accuracy:", accuracy)


# In[ ]:


vif_data= ['HighBlood', 'Stroke', 'Arthritis', 
   'Diabetes','Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis',
   'Reflux_esophagitis', 'Asthma','Services_CT Scan','TotalCharge' ,
      'Services_MRI', 'Complication_risk_Low',
      'Complication_risk_Medium', 'Initial_admin_Emergency Admission',
      'Initial_admin_Observation Admission']


# In[ ]:


vif_data = pd.DataFrame() 
vif_data["feature"] = X.columns 
vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))] 
  
print(vif_data)


# Most predictor variables have low VIF values, suggesting that multicollinearity is not a significant concern for individual predictors. 
# These variables have VIF values close to 1, indicating low multicollinearity. 
# HighBlood, Stroke, Arthritis, Diabetes, Hyperlipidemia, BackPain, Anxiety, Allergic_rhinitis, Reflux_esophagitis, Asthma, Services_CT Scan, TotalCharge, Services_MRI:
# 
# 
# The follwoing  variables have somewhat higher VIF values, but they are still below 5. While they may suggest a moderate level of multicollinearity, it is not severe.
# 
# Complication_risk_Low, Complication_risk_Medium, Initial_admin_Emergency Admission, Initial_admin_Observation Admission

# 

# In[ ]:


#E3.  Provide an executable error-free copy of the code 


# In[ ]:


med_data.to_csv('MSDA208_PA_task2_E3.cvs')


# # Part V: Data Summary and Implications

# In[ ]:


#F1.


# In[ ]:


# a regression equation for the reduced model


# In(p/(1-p)) = −121.6056−1.1079×HighBlood+1.6548×Stroke−2.6233×Arthritis−0.8999×Diabetes−
# 1.4398×Hyperlipidemia−1.2672×BackPain−2.5751×Anxiety−1.4335×Allergic_rhinitis−1.4700×Reflux_esophagitis−1.3596×
# Asthma+1.5662×Services_CT_Scan+0.0180×TotalCharge+2.6915×Services_MRI+5.5986×Complication_risk_Low+7.0666×
# Complication_risk_Medium−6.7370×Initial_admin_Emergency_Admission+0.7856×Initial_admin_Observation_Admission

# In[ ]:


#.an interpretation of the coefficients of the reduced model


# **Interpretation: Holding all other variables constant
# 
# HighBlood:
# Coefficient: -1.1079
# Interpretation:  readmission decrease by 1.1079 for individuals with High Blood compared to those without High Blood.
# 
# Stroke:
# Coefficient: 1.6548
#  readmission increase by 1.6548 for individuals with a history of Stroke.
#  
# 
# Arthritis:
# Coefficient: -2.6233
# Interpretation:  readmission decrease by 2.6233 for individuals with Arthritis compared to those without Arthritis.
# 
# Diabetes:
# Coefficient: -0.8999
# Interpretation: readmission decrease by 0.8999 for individuals with Diabetes compared to those without Diabetes.
# 
# Hyperlipidemia:
# Coefficient: -1.4398
# Interpretation:   readmission decrease by 1.4398 for individuals with Hyperlipidemia compared to those without Hyperlipidemia.
# 
# BackPain:
# Coefficient: -1.2672
# Interpretation:  readmission decrease by 1.2672 for individuals with Back Pain compared to those without Back Pain.
# 
# Anxiety:
# Coefficient: -2.5751
# Interpretation:   readmission decrease by 2.5751 for individuals with Anxiety compared to those without Anxiety.
# 
# Allergic_rhinitis:
# Coefficient: -1.4335
# Interpretation: readmission decrease by 1.4335 for individuals with Allergic Rhinitis compared to those without Allergic Rhinitis.
# 
# Reflux_esophagitis:
# Coefficient: -1.4700
# Interpretation: readmission decrease by 1.4700 for individuals with Reflux Esophagitis compared to those without Reflux Esophagitis.
# 
# Asthma:
# Coefficient: -1.3596
# Interpretation: readmission decrease by 1.3596 for individuals with Asthma compared to those without Asthma.
# 
# Services_CT Scan:
# Coefficient: 1.5662
# Interpretation:  readmission increase by 1.5662 for individuals who had a CT Scan compared to those who did not.
# 
# TotalCharge:
# Coefficient: 0.0180
# Interpretation:  a one-unit increase in Total Charge is associated with an increase in readmission by 0.0180.
# 
# Services_MRI:
# Coefficient: 2.6915
# Interpretation: readmission increase by 2.6915 for individuals who had an MRI compared to those who did not.
# 
# Complication_risk_Low:
# Coefficient: 5.5986
# Interpretation:  readmission increase by 5.5986 for individuals with a Low Complication Risk compared to those with a Medium Complication Risk.
# 
# Complication_risk_Medium:
# Coefficient: 7.0666
# Interpretation:readmission increase by 7.0666 for individuals with a Medium Complication Risk compared to those with a Low Complication Risk.
# 
# Initial_admin_Emergency Admission:
# Coefficient: -6.7370
# Interpretation:  readmission decrease by 6.7370 for individuals admitted through Emergency Admission compared to those with other initial admission types.
# 
# Initial_admin_Observation Admission:
# Coefficient: 0.7856
# Interpretation: readmission increase by 0.7856 for individuals admitted through Observation Admission compared to those with other initial admission types.

# In[ ]:


# the statistical and practical significance of the reduced model


# Likelihood Ratio Test (LLR) p-value is 0.0 which indicates the model is statstically significant. 
# The model is also practically significant because we have columns such as Initial_admin_Observation Admission, 
# Complication_risk_Medium, Complication_risk_Low, Services_MRI, TotalCharge, Services_CT Scan that are
# suggesting the likelihood of hospital readmission.

# In[ ]:


#the limitations of the data analysis


# The analysis relies on a dataset of 10,000 patients, which might be inadequate for making precise outcome predictions.
# Another limitation is the complexity of medical conditions and the use of coefficients to interpret readmission rates.
# For instance, Diabetes has a coefficient of -0.8999, suggesting a decrease in readmission by 0.8999 for individuals
# with Diabetes compared to those without Diabetes. However, in reality, having diabetes likely increases a patient's readmission rate.

# In[ ]:


#F2 Recommend a course of action based on your results. 


# Based on the analysis, we recommend that stakeholders examine the 17 independent variables that impact readmission rates.
# We have identified variables such as Complication_risk_Medium, Services_MRI, Services_CT Scan, and Stroke that are 
# highly associated with an increase in readmission rates. We advise stakeholders to provide education and resources for 
# patients with a history of stroke and complication risk. Additionally, consider implementing an outreach program to 
# ensure patients are connected to a primary care provider and actively engaged in managing their health.

# # Part VI: Demonstration

# In[ ]:


#G


# In[ ]:


#HGeeksforGeeks. (n.d.). Detecting Multicollinearity with VIF in Python. GeeksforGeeks. https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/
    
Smith, J. (2021, October 15). Building a Logistic Regression in Python: Step by Step. Towards Data Science. https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
        
ResidentMario. (2021, July 12). Bivariate Plotting with Pandas. Kaggle. https://www.kaggle.com/code/residentmario/bivariate-plotting-with-pandas/notebook.
    


# GeeksforGeeks. (n.d.). Detecting Multicollinearity with VIF in Python. GeeksforGeeks. https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/
#     
# Smith, J. (2021, October 15). Building a Logistic Regression in Python: Step by Step. Towards Data Science. https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
#         
# ResidentMario. (2021, July 12). Bivariate Plotting with Pandas. Kaggle. https://www.kaggle.com/code/residentmario/bivariate-plotting-with-pandas/notebook.
#     

# In[ ]:


#I


# In[ ]:


#J


# In[ ]:




