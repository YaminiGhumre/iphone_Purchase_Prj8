#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve,auc


# In[2]:


#### Generating Synthetic data
data = pd.read_csv("C:/Users/Admin/Documents/Python Scripts/Assignments_18-01-2024/Assignments/Decision Tree Project8\Decision Tree  - Assignment -4/iphone_purchase_records.csv")
print(data.head(5))


# In[3]:


data_types = pd.DataFrame(data.dtypes, columns = ['Data Type'])
data_types


# In[4]:


data.shape


# In[5]:


missing_values = pd.DataFrame(data.isna().sum(), columns = ['Missing Values'])
missing_values


# In[6]:


data.describe()


# In[7]:


print(data.nunique())
print("Gender having unique values - ",data['Gender'].unique())
print("Purchase IPhone having unique values - ",data['Purchase Iphone'].unique())


# In[10]:


data.dtypes


# In[11]:


## any outliers in dataset
# numeric columns in data
ncol= data.columns[data.dtypes!='O']
ncol


# In[14]:


for i in ncol:
    plt.figure()
    sns.boxplot(x=data[i])


# In[15]:


data['Gender'].value_counts(normalize=True)*100


# In[16]:


(data['Gender'].value_counts(normalize=True)*100).plot(kind='bar')
plt.title('Gender wise contribution')
plt.show()


# In[18]:


genderwise_avg_age = round(data.groupby('Gender')['Age'].mean(),0)
print("Female average age is - ",genderwise_avg_age[0])
print("Male avergae age is - ",genderwise_avg_age[1])


# In[19]:


genderwise_avg_age.plot(kind='pie')
plt.title('gender wise avg age')
plt.show()


# In[21]:


genderwise_max_age = round(data.groupby('Gender')['Age'].max(),0)
print("Female max age is - ",genderwise_max_age[0])
print("Male max age is - ",genderwise_max_age[1])


# In[22]:


genderwise_min_age = round(data.groupby('Gender')['Age'].min(),0)
print("Female min age is - ",genderwise_min_age[0])
print("Male min age is - ",genderwise_min_age[1])


# In[23]:


genderwise_avg_salary = round(data.groupby('Gender')['Salary'].mean(),0)
print("Female average salary is - ",genderwise_avg_salary[0])
print("Male avergae salary is - ",genderwise_avg_salary[1])


# In[24]:


genderwise_avg_salary.plot(kind='barh')
plt.title('Gender wise avg salary')
plt.show()


# In[25]:


genderwise_max_salary = round(data.groupby('Gender')['Salary'].max(),0)
print("Female max salary is - ",genderwise_max_salary[0])
print("Male max salary is - ",genderwise_max_salary[1])


# In[26]:


print("Total Iphone purchased are - ",data['Purchase Iphone'].value_counts()[1])


# In[27]:


data.groupby('Gender')['Purchase Iphone'].sum()


# In[28]:


## Iphone purchased data filtered from dataset as df1

data1= data[data['Purchase Iphone']==1]
data1


# In[29]:


sns.lineplot(data=data1,x='Age',y='Purchase Iphone',estimator='sum',hue='Gender')
plt.title('Age wise Iphone Purchased')
plt.show()


# In[30]:


agewise_salary=pd.DataFrame(data1.groupby(['Age','Gender'])['Salary'].sum(),columns=['Salary'])


# In[31]:


sns.scatterplot(data=agewise_salary,x='Age',y='Salary',hue='Gender')


# ### There is no any correlation in salary and age in case of both male and female.

# In[ ]:





# In[34]:


correlation = data.corr()
sns.heatmap(correlation,annot=True,cmap='rainbow')
plt.show()


# ### As per correlation table, only Age is having good correlation with Purchase Iphone.

# In[35]:


data.head(5)


# In[37]:


## convert categorical column gender into numerical using get dummies data.
data_dummy=pd.get_dummies(data['Gender'])


# In[38]:


data_final=pd.concat([data,data_dummy],axis=1)


# In[39]:


data_final


# ### Training & Testing the Model

# In[41]:


X = data_final[['Female','Male','Age','Salary']]
y = data_final['Purchase Iphone']


# In[42]:


#### Splitting the data -
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 123)


# In[43]:


X_train


# In[44]:


X_test


# In[45]:


y_train


# In[46]:


y_test


# In[47]:


X_train.shape


# In[48]:


### two different methods for modelling
logistics_model = LogisticRegression(random_state = 1234)
logistics_model.fit(X_train, y_train)
decisiontree_model = DecisionTreeClassifier(criterion = 'entropy', random_state=1234)
decisiontree_model.fit(X_train,y_train)
print("Training the Models")


# In[49]:


logistics_model.classes_


# In[62]:


def decision_tree_model(X_train, X_test, y_train, y_test):
    model= DecisionTreeClassifier(criterion='entropy')
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    results = confusion_matrix(y_test,pred)
    accuracy = accuracy_score(y_test,pred)
    return accuracy


# In[63]:


accuracy = decision_tree_model(X_train, X_test, y_train, y_test)
print("Accuracy of Decision Tree Model -", accuracy)


# In[56]:


## to save model, importing joblib
import joblib


# In[57]:


filename='iphone_Purchase_Prj8.sav'


# In[ ]:




