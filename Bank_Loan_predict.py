

import pandas as pd
import numpy as np
import os


# In[2]:


train=pd.read_csv('./Loan_Data/train.csv')
train.Loan_Status=train.Loan_Status.map({'Y':1,'N':0})


# In[3]:


train.isnull().sum()


# In[4]:


Loan_status=train.Loan_Status
train.drop('Loan_Status',axis=1,inplace=True)
test=pd.read_csv('./Loan_Data/test.csv')
Loan_ID=test.Loan_ID
data=train.append(test)
data.head()


# In[5]:


data.shape


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns




# In[9]:


## Label encoding for gender
data.Gender=data.Gender.map({'Male':1,'Female':0})
data.Gender.value_counts()


# In[10]:


## Labelling 0 & 1 for Marrital status
data.Married=data.Married.map({'Yes':1,'No':0})
data.Married.value_counts()


# In[11]:


## Labelling 0 & 1 for Dependents
data.Dependents=data.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
data.Dependents.value_counts()


# In[12]:


## Labelling 0 & 1 for Education Status
data.Education=data.Education.map({'Graduate':1,'Not Graduate':0})
data.Education.value_counts()


# In[13]:


## Labelling 0 & 1 for Employment status
data.Self_Employed=data.Self_Employed.map({'Yes':1,'No':0})
data.Self_Employed.value_counts()


# In[14]:


data.Property_Area.value_counts()


# In[15]:


## Labelling 0 & 1 for Property area
data.Property_Area=data.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})
data.Property_Area.value_counts()


# In[16]:


data.head()


# # Filling missing values

# In[17]:


data.Credit_History.fillna(np.random.randint(0,2),inplace=True)


# In[18]:


data.isnull().sum()


# In[19]:


data.Married.fillna(np.random.randint(0,2),inplace=True)


# In[20]:


## Filling with median
data.LoanAmount.fillna(data.LoanAmount.median(),inplace=True)


# In[21]:


## Filling with mean
data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.mean(),inplace=True)


# In[22]:


data.Gender.value_counts()


# In[23]:


## Filling Gender with random number between 0-2
from random import randint 
data.Gender.fillna(np.random.randint(0,2),inplace=True)


# In[24]:


data.Gender.value_counts()


# In[25]:


## Filling Dependents with median
data.Dependents.fillna(data.Dependents.median(),inplace=True)


# In[26]:


data.isnull().sum()


# In[27]:


data.Self_Employed.fillna(np.random.randint(0,2),inplace=True)
data.isnull().sum()


# In[28]:


data.head()


# In[29]:


## Dropping Loan ID from data
data.drop('Loan_ID',inplace=True,axis=1)


# # Splitting the dataset for training and testing

# In[30]:


train_X=data.iloc[:614,] 
train_y=Loan_status  


# In[31]:


from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(train_X,train_y,random_state=0)


# In[32]:


train_X.head()


# # Different machine learning model

# In[33]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[34]:


models=[]
models.append(("Logistic Regression",LogisticRegression()))
models.append(("Decision Tree",DecisionTreeClassifier()))
models.append(("Linear Discriminant Analysis",LinearDiscriminantAnalysis()))
models.append(("Random Forest",RandomForestClassifier()))
models.append(("Support Vector Classifier",SVC()))


# In[35]:


scoring='accuracy'


# In[36]:


from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
result=[]
names=[]


# In[37]:


for name,model in models:
    kfold=KFold(n_splits=10,random_state=0)
    cv_result=cross_val_score(model,train_X,train_y,cv=kfold,scoring=scoring)
    result.append(cv_result)
    names.append(name)
    #print(model)
    #print("%s %f" % (name,cv_result.mean()))


# In[48]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

LR=LogisticRegression()
#LR = SVC()
LR.fit(train_X,train_y)
pred=LR.predict(test_X)
#print("Model Accuracy:- ",accuracy_score(test_y,pred))
#print(confusion_matrix(test_y,pred))
#print(classification_report(test_y,pred))


# In[49]:


print(pred)


# In[50]:


X_test=data.iloc[614:,] 
# X_test[sc_f]=SC.fit_transform(X_test[sc_f])


# In[51]:


prediction = LR.predict(X_test)
#print(prediction)


# In[52]:


## TAken data from the dataset
t = LR.predict([[0.0,	0.0,	0.0,	1,	0.0,	1811,	1666.0,	54.0,	360.0,	1.0,	2]])


# In[53]:


#print(t)


# In[54]:


import pickle
import sklearn
svc = sklearn.svm.SVC()

file = './ML_Model1.pkl'
with open(file, 'wb') as f:
    pickle.dump(svc, f)


# In[64]:



prediction = LR.predict([[0, 0, 0, 0, 0, 7666, 8787, 5767, 60, 0, 0]])
#print(prediction)


import streamlit as st
from PIL import Image
import pickle
import pandas as pd


model = pickle.load(open('./ML_Model1.pkl', 'rb'))


def run():
    img1 = Image.open('bank1.JPG')
    #img1 = img1.resize((156,145))
    st.image(img1,use_column_width=False)
    st.title("Bank Loan Prediction using Machine Learning")

    ## Account No
    account_no = st.text_input('Account number')

    ## Full Name
    fn = st.text_input('Full Name')

    ## For gender
    gen_display = ('Female','Male')
    gen_options = list(range(len(gen_display)))
    gen = st.selectbox("Gender",gen_options, format_func=lambda x: gen_display[x])

    ## For Marital Status
    mar_display = ('No','Yes')
    mar_options = list(range(len(mar_display)))
    mar = st.selectbox("Marital Status", mar_options, format_func=lambda x: mar_display[x])

    ## No of dependets
    dep_display = ('No','One','Two','More than Two')
    dep_options = list(range(len(dep_display)))
    dep = st.selectbox("Dependents",  dep_options, format_func=lambda x: dep_display[x])

    ## For edu
    edu_display = ('Not Graduate','Graduate')
    edu_options = list(range(len(edu_display)))
    edu = st.selectbox("Education",edu_options, format_func=lambda x: edu_display[x])

    ## For emp status
    emp_display = ('Job','Business')
    emp_options = list(range(len(emp_display)))
    emp = st.selectbox("Employment Status",emp_options, format_func=lambda x: emp_display[x])

    ## For Property status
    prop_display = ('Rural','Semi-Urban','Urban')
    prop_options = list(range(len(prop_display)))
    prop = st.selectbox("Property Area",prop_options, format_func=lambda x: prop_display[x])

    ## For Credit Score
    cred_display = ('Between 300 to 500','Above 500')
    cred_options = list(range(len(cred_display)))
    cred = st.selectbox("Credit Score",cred_options, format_func=lambda x: cred_display[x])

    ## Applicant Monthly Income
    mon_income = st.number_input("Applicant's Monthly Income($)",value=0)

    ## Co-Applicant Monthly Income
    co_mon_income = st.number_input("Co-Applicant's Monthly Income($)",value=0)

    ## Loan AMount
    loan_amt = st.number_input("Loan Amount",value=0)

    ## loan duration
    dur_display = ['2 Month','6 Month','8 Month','1 Year','16 Month']
    dur_options = range(len(dur_display))
    dur = st.selectbox("Loan Duration",dur_options, format_func=lambda x: dur_display[x])

    if st.button("Submit"):
        duration = 0
        if dur == 0:
            duration = 60
        if dur == 1:
            duration = 180
        if dur == 2:
            duration = 240
        if dur == 3:
            duration = 360
        if dur == 4:
            duration = 480
        features = [[gen, mar, dep, edu, emp, mon_income, co_mon_income, loan_amt, duration, cred, prop]]
        #prediction = predict(gen, mar, dep, edu, emp, mon_income, co_mon_income, loan_amt, duration, cred, prop)
        #st.write(features)
        prediction = LR.predict(features)
        #st.write(prediction)
        lc = [str(i) for i in prediction]
        ans = int("".join(lc))
        
        if ans == 0:
            st.error(
                "Hello: " + fn +" || "
                "Account number: "+account_no +' || '
                'According to our Calculations, you will not get the loan from Bank'
            )
        else:
            st.success(
                "Hello: " + fn +" || "
                "Account number: "+account_no +' || '
                'Congratulations!! you will get the loan from Bank'
            )

run()

st.write("Made By Nishant Singh")
