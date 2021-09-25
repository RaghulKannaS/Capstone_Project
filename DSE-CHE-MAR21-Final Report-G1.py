#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


# import 'Pandas' 
import pandas as pd 

# import 'Numpy' 
import numpy as np

# import subpackage of Matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# import 'Seaborn' 
import seaborn as sns
import scipy.stats as stats

# to suppress warnings 
from warnings import filterwarnings
filterwarnings('ignore')

# display all columns of the dataframe
pd.options.display.max_columns = None

# import train-test split 
from sklearn.model_selection import train_test_split

# import various functions from statsmodels
import statsmodels
import statsmodels.api as sm

# import StandardScaler to perform scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder 
from imblearn.over_sampling import SMOTE

# import various functions from sklearn 
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,cohen_kappa_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import KFold

# import function to perform feature selection
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

import  warnings
warnings.filterwarnings("ignore")


# In[2]:


from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)


# # Title: Bank Marketing

# ### Business Problem
# There has been a revenue decline for the Portuguese bank and they would like to know what actions to take. After investigation, they found out that the root cause is that their clients are not depositing as frequently as before. Knowing that term deposits allow banks to hold onto a deposit for a specific amount of time, so banks can invest in higher gain financial products to make a profit. In addition, banks also hold better chance to persuade term deposit clients into buying other products such as funds or insurance to further increase their revenues. As a result, the Portuguese bank would like to identify existing clients that have higher chance to subscribe for a term deposit and focus marketing effort on such clients.
# 
# To resolve the proble, we suggest a classification approach to predict which clients are more likely to subscribe for term deposits.

# The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required,in order to access if the product (bank term deposit) would be (or not) subscribed.
# The classification goal is to predict if the client will subscribe a term deposit (variable y).
# 
# Age group: 
# 
# 10 - 19 = 1
# 
# 20 - 29 = 2
# 
# 30 - 39 = 3
# 
# 40 - 49 = 4
# 
# 50 - 59 = 5
# 
# 60 - 69 = 6
# 
# 70 - 79 = 7
# 
# 80 - 89 = 8
# 
# 90 - 99 = 9
# 
# 1 - age (numeric)
# 
# 2 - job : type of job (categorical:"admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services") 
#  
# 3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
#  
# 4 - education (categorical: "unknown","secondary","primary","tertiary")
#  
# 5 - default: has credit in default? (binary: "yes","no")
#    
# 6 - balance: average yearly balance, in euros (numeric)
#    
# 7 - housing: has housing loan? (binary: "yes","no")
#    
# 8 - loan: has personal loan? (binary: "yes","no")
#    
# #related with the last contact of the current campaign:
#     
# 9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
# 
# 10 - day: last contact day of the month (numeric)
#   
# 11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
#   
# 12 - duration: last contact duration, in seconds (numeric).
# 
# Important note:  This attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
#   
# #other attributes:
# 
# 13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 
# 14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
# 
# 15 - previous: number of contacts performed before this campaign and for this client (numeric)
#   
# 16 - poutcome: outcome of the previous marketing campaign (categorical:"unknown","other","failure","success")
# 
# Output variable (desired target):
#   
# 17 - y - has the client subscribed a term deposit? (binary: "yes","no")
# 
# Missing Attribute Values: There are several missing values in some categorical attributes, all coded with the "unknown" label. These missing values can be treated as a possible class label or using deletion or imputation techniques. 

# In[196]:


df = pd.read_csv('bank-marketing.csv')
df.head()


# In[4]:


df.sample(10)


# In[5]:


df.info()


# In[197]:


df.drop('marital-education',1,inplace=True)


# In[198]:


df.drop('response',1,inplace=True)


# In[8]:


df.shape


# In[9]:


df.isnull().sum()


# In[10]:


df.describe()


# In[ ]:





# In[11]:


df.describe(include=np.object).T


# In[12]:


for i in df.select_dtypes(include=np.object).columns:
    print(df[i].value_counts())
    print()


# In[13]:


df['y'].value_counts()


# In[14]:


df['y'].value_counts(normalize=True)*100


# In[ ]:





# In[15]:


df.head()


# In[7]:


num_col = ['age','salary','balance','day','duration','campaign','pdays','previous']
cat_col = ['age group', 'eligible', 'job', 'marital', 'education','targeted', 'default', 'housing',
           'loan', 'contact', 'month', 'poutcome']


# In[17]:


len(num_col),len(cat_col)


# In[18]:


for i in num_col:
    plt.figure(figsize=(18,7))
    plt.subplot(1,2,1)
    sns.distplot(df[i])
    plt.subplot(1,2,2)
    sns.boxplot(df[i])
    plt.show()


# In[ ]:





# In[19]:


for i in num_col:
    plt.figure(figsize=(18,7))
    plt.subplot(1,2,1)
    sns.barplot(df['y'],df[i])
    plt.subplot(1,2,2)
    sns.boxplot(df['y'],df[i])
    plt.show()


# In[ ]:





# In[20]:


for i in cat_col:
    sns.countplot(y=df[i],hue=df['y'])
    plt.show()


# In[ ]:





# In[21]:


for i in num_col:
    for j in cat_col:
        sns.barplot(df[j],df[i],hue=df['y'])
        plt.xticks(rotation=90)
        plt.show()


# In[ ]:





# In[199]:


df['y'].replace({'no':0,'yes':1},inplace=True)


# In[23]:


plt.figure(figsize=(18,7))
sns.heatmap(df.corr(),annot=True,fmt='.2f');


# ### Multi Collinearity Check:

# In[9]:


num_col


# In[12]:


v = df[num_col]
vif = [VIF(v.values,i) for i in range(v.shape[1])]
vif_df = pd.DataFrame()
vif_df['numeric_features'] = v.columns
vif_df['VIF'] = vif
vif_df.sort_values('VIF',ascending=False)


# In[ ]:





# In[ ]:





# In[24]:


sns.pairplot(df,hue='y');


# In[ ]:





# In[25]:


for i in num_col:
    sns.scatterplot(df[i],df['y'])
    plt.show()


# In[ ]:





# # Treating 'Unknown' Values

# In[151]:


pd.crosstab(df['job'],df['education'])


# In[200]:


ind = df[(df['job']=='unknown') & (df['education']=='primary')]['job'].index
df.iloc[ind,3] = 'blue-collar'


# In[201]:


ind = df[(df['job']=='unknown') & (df['education']=='secondary')]['job'].index
df.iloc[ind,3] = 'blue-collar'


# In[202]:


ind = df[(df['job']=='unknown') & (df['education']=='tertiary')]['job'].index
df.iloc[ind,3] = 'management'


# In[203]:


pd.crosstab(df['job'],df['education'])


# In[204]:


ind = df[(df['education']=='unknown') & (df['job']=='admin.')]['education'].index
df.iloc[ind,6] = 'secondary'


# In[205]:


ind = df[(df['education']=='unknown') & (df['job']=='blue-collar')]['education'].index
df.iloc[ind,6] = 'secondary'


# In[206]:


ind = df[(df['education']=='unknown') & (df['job']=='entrepreneur')]['education'].index
df.iloc[ind,6] = 'tertiary'


# In[207]:


ind = df[(df['education']=='unknown') & (df['job']=='housemaid')]['education'].index
df.iloc[ind,6] = 'primary'


# In[208]:


ind = df[(df['education']=='unknown') & (df['job']=='management')]['education'].index
df.iloc[ind,6] = 'tertiary'


# In[209]:


ind = df[(df['education']=='unknown') & (df['job']=='retired')]['education'].index
df.iloc[ind,6] = 'secondary'


# In[210]:


ind = df[(df['education']=='unknown') & (df['job']=='self-employed')]['education'].index
df.iloc[ind,6] = 'tertiary'


# In[211]:


ind = df[(df['education']=='unknown') & (df['job']=='services')]['education'].index
df.iloc[ind,6] = 'secondary'


# In[212]:


ind = df[(df['education']=='unknown') & (df['job']=='student')]['education'].index
df.iloc[ind,6] = 'secondary'


# In[213]:


ind = df[(df['education']=='unknown') & (df['job']=='technician')]['education'].index
df.iloc[ind,6] = 'secondary'


# In[214]:


ind = df[(df['education']=='unknown') & (df['job']=='unemployed')]['education'].index
df.iloc[ind,6] = 'secondary'


# In[115]:


pd.crosstab(df['job'],df['education'])


# In[215]:


ind = df[(df['education']=='unknown') & (df['job']=='unknown')].index
df.drop(index=ind,inplace=True)


# In[216]:


pd.crosstab(df['job'],df['education'])


# In[217]:


df['contact'].value_counts()


# In[218]:


df['contact'].mode()[0]


# In[219]:


df['contact'].replace({'unknown':df['contact'].mode()[0]},inplace=True)


# In[220]:


df['poutcome'].value_counts()


# In[221]:


df.drop('poutcome',1,inplace=True)


# In[123]:


sns.distplot(df['pdays']);


# In[174]:


df['pdays'].value_counts()


# In[222]:


def pdays(x):
    if (x<=0):
        return 'Not.Previously.Contacted'
    elif (x>0 and x<=150):
        return '1-150 days'
    elif (x>150 and x<=300):
        return '151-300 days'
    else:
        return '>300 days'


# In[223]:


df['pdays'] = df['pdays'].apply(pdays)


# ### Outliers Treatment

# In[155]:


df.head()


# In[156]:


df.shape


# In[37]:


for i in df.select_dtypes(include=np.number).columns:
    sns.boxplot(df[i],whis=3)
    plt.show()


# In[58]:


df['previous'].value_counts()


# In[224]:


ind = df[df['previous']>8].index
df.drop(index=ind,inplace=True)


# In[225]:


df['previous'].value_counts()


# In[226]:


df.columns


# In[227]:


num_out = ['salary','balance','duration','campaign']


# In[228]:


for i in num_out:
    q1 = df[i].quantile(0.25)
    q3 = df[i].quantile(0.75)
    iqr = q3 - q1
    ll = q1 - (3*iqr)
    ul = q3 + (3*iqr)
    df[i] = df[(df[i]>=ll) & (df[i]<=ul)][i]


# In[229]:


df.isnull().sum()/len(df)*100


# In[230]:


df.dropna(inplace=True)


# In[231]:


df.shape


# ### Statistical Tests

# In[232]:


df.head()


# In[233]:


df.columns


# In[234]:


num_col = ['age', 'salary', 'balance', 'duration', 'campaign', 'previous']
cat_col = ['loan','job','default','day','month','age group','housing','marital','pdays','contact',
           'targeted','education','eligible']


# In[187]:


num_col


# In[188]:


cat_col


# In[236]:


# Chi-sqr Test of Independence
# Hypothesis Formation
# Ho : Variables are Independant (NO relation)
# Ha : Variables are Not independant (Relation)

def chi(obs):
    chi_stat,pval,df,exp_tab = stats.chi2_contingency(obs)
    return pval

not_sig_features = []
sig_features = []

for i in cat_col:
    obs = pd.crosstab(df[i],df['y'])
    pval = chi(obs)
    if (pval > 0.05):
        not_sig_features.append(i)   # Accept H0
    else:
        sig_features.append(i)       # Reject Ho


# In[238]:


not_sig_features


# In[239]:


sig_features


# In[242]:


for i in num_col:
    stat,pval = stats.shapiro(df[i])
    print(i,':',pval)


# In[ ]:





# ### Encoding

# In[165]:


df.head()


# In[166]:


df['month'].replace({'may':5,'jun':6,'jul':7,'aug':8,'oct':10,'nov':11,'dec':12,'jan':1,'feb':2,'mar':3,
                     'apr':4,'sep':9},inplace=True)


# In[167]:


cat_le = ['eligible','job','marital','education','targeted','default','housing',
          'loan','contact','pdays']


# In[168]:


df[cat_le].head()


# In[169]:


for i in cat_le:
    print(i,'',df[i].unique())
    print()


# In[170]:


le = LabelEncoder()


# In[171]:


for i in cat_le:
    df[i] = le.fit_transform(df[i])


# In[172]:


df.dtypes


# In[173]:


df.drop('age',1,inplace=True)


# In[174]:


df.head()


# ### Handling Imbalanced Data

# In[175]:


df['y'].value_counts()


# In[176]:


df['y'].value_counts(normalize=True)*100


# In[177]:


x = df.drop('y',1)
y = df['y']


# In[178]:


smote = SMOTE(sampling_strategy=0.5,random_state=10)
x_sm,y_sm = smote.fit_resample(x,y)


# In[179]:


df_sm = pd.DataFrame(x_sm,columns=x.columns)
df_sm['y']=y_sm
df_sm.head()


# In[180]:


df_sm['y'].value_counts()


# In[181]:


df_sm['y'].value_counts(normalize=True)*100


# In[182]:


df.to_csv('Capstone_df.csv')
df_sm.to_csv('Capstone_df_sm.csv')


# In[183]:


df.head()


# In[184]:


df_sm.head()


# In[ ]:





# # Assumptions

# # For Logistic Regression

# ## From GL Pdf
# ### Assumption 1
# Independence of error, whereby all sample group outcomes are
# separate from each other (i.e., there are no duplicate responses)
# ### Assumption 2
# Linearity in the logit for any continuous independent variables
# ### Assumption 3
# Absence of multicollinearity
# ### Assumption 4
# lack of strongly influential outliers

# ## From Internet
# ### Assumption 1: Appropriate dependent variable structure
# This assumption simply states that a binary logistic regression requires your dependent variable to be dichotomous and an ordinal logistic regression requires it to be ordinal.
# In addition, the dependent variable should neither be an interval nor ratio scale.
# ### Assumption 2: There is a linear relationship between the logit of the outcome and each independent variable.
# The logit function is given by:
# logit(p) = log(p/(1-p)), where p is the probability of an outcome
# To check this assumption, you can do it visually by plotting each independent variable and the logit values on a scatterplot.The Y axes are the independent variables while the X axis shows the logit values. Then look at the equation of the curve to see if it meets the linearity assumption.
# Remember that linearity is in the parameters. As long as the equation meets the linear equation form stated above, it meets the linearity assumption.
# ### Assumption 3: No Multicollinearity
# As with the assumption for OLS regression, the same can be said here.
# ### Assumption 4: No Influential Outliers
# Influential outliers are extreme data points that affect the quality of the logistic regression model.
# Not all outliers are influential.
# You will need to check for which points are the influential ones before removing or transforming them for analysis.
# To check for outliers, you can run Cook’s Distance on the data values. A high Cook’s Distance value indicates outliers.
# A rule of thumb for flagging out an influential outlier is when Cook’s Distance > 1.
# ### Assumption 5: Observation Independence
# This assumption requires logistic regression observations to be independent of each other.
# That is, observations should not come from a repeated measure design.
# A repeated measure design refers to multiple measures of the same variable taken for the same person under different experimental conditions or across time.
# A good example of repeated measures is longitudinal studies — tracking progress of a subject over years.

# # For KNN
# ### Assumptions in KNN
# Before using KNN, let us revisit some of the assumptions in KNN.
# 
# KNN assumes that the data is in a feature space. More exactly, the data points are in a metric space. The data can be scalars or possibly even multidimensional vectors. Since the points are in feature space, they have a notion of distance – This need not necessarily be Euclidean distance although it is the one commonly used.
# 
# Each of the training data consists of a set of vectors and class label associated with each vector. In the simplest case , it will be either + or – (for positive or negative classes). But KNN , can work equally well with arbitrary number of classes.
# 
# We are also given a single number "k" . This number decides how many neighbors (where neighbors is defined based on the distance metric) influence the classification. This is usually a odd number if the number of classes is 2. If k=1 , then the algorithm is simply called the nearest neighbor algorithm.

# # For Naive Bayes
# It is a classification technique based on Bayes' Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.

# # For Tree-based Models
# For tree-based models such as Decision Trees, Random Forest & Gradient Boosting there are no model assumptions to validate.
# Unlike OLS regression or logistic regression, tree-based models are robust to outliers and do not require the dependent variables to meet any normality assumptions.

# In[ ]:





# In[4]:


df = pd.read_csv('Capstone_df.csv')
df_sm = pd.read_csv('Capstone_df_sm.csv')


# In[5]:


df.head()


# In[6]:


df_sm.head()


# In[7]:


df.drop('Unnamed: 0',1,inplace=True)
df_sm.drop('Unnamed: 0',1,inplace=True)


# In[8]:


df.head()


# In[9]:


df_sm.head()


# In[218]:


df['y'].value_counts(normalize=True)*100


# In[219]:


df_sm['y'].value_counts(normalize=True)*100


# # Splitting

# In[10]:


x = df.drop('y',1)
y = df['y']

x_sm = df_sm.drop('y',1)
y_sm = df_sm['y']


# In[11]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=10)

x_sm_train,x_sm_test,y_sm_train,y_sm_test = train_test_split(x_sm,y_sm,test_size=0.3,random_state=10)


# # Scaling

# In[12]:


ss = StandardScaler()


# In[13]:


sc = ['salary','balance','duration','campaign','previous']


# In[14]:


# Only used for Cross Validation Score

x_scaled = x.copy(deep=True)
x_sm_scaled = x_sm.copy(deep=True)

x_scaled[sc] = ss.fit_transform(x_scaled[sc])
x_sm_scaled[sc] = ss.fit_transform(x_sm_scaled[sc])


# In[15]:


# For Model Building

x_train[sc] = ss.fit_transform(x_train[sc])
x_test[sc] = ss.fit_transform(x_test[sc])

x_sm_train[sc] = ss.fit_transform(x_sm_train[sc])
x_sm_test[sc] = ss.fit_transform(x_sm_test[sc])


# # 1) Logistic Regression:

# ### Base Model (without SMOTE)

# In[11]:


log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)


# In[12]:


y_test_pred = log_reg.predict(x_test)


# In[13]:


y_test_prob_1 = log_reg.predict_proba(x_test)[:,1]


# In[229]:


cm = confusion_matrix(y_test,y_test_pred)


# In[230]:


sns.heatmap(cm,annot=True,fmt='.0f');


# In[231]:


print(classification_report(y_test,y_test_pred))


# In[232]:


fpr, tpr, threshold = roc_curve(y_test,y_test_prob_1)
roc_df = pd.DataFrame([fpr,tpr,threshold],index=['FPR','TPR','Threshold']).T


# In[233]:


plt.plot(fpr,tpr)
plt.plot([[0,0],[1,1]])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC Curve')
plt.show()


# In[234]:


roc_auc_score(y_test,y_test_prob_1)


# ### For Cross Validation

# In[14]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=log_reg,X=x_scaled,y=y,cv=k,scoring='recall')
np.mean(scores)


# In[15]:


recall_log_reg_without_SMOTE = scores


# In[16]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=log_reg,X=x_scaled,y=y,cv=k,scoring='f1_weighted')
np.mean(scores)


# In[17]:


f1_weighted_log_reg_without_SMOTE = scores


# In[18]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=log_reg,X=x_scaled,y=y,cv=k,scoring='roc_auc')
scores


# In[19]:


roc_auc_log_reg_without_SMOTE = scores


# In[238]:


roc_auc = np.mean(scores)
roc_auc


# In[239]:


bias_error = 1 - np.mean(scores)
bias_error


# In[240]:


variance_error = np.std(scores,ddof=1) / np.abs(np.mean(scores))
variance_error


# In[ ]:





# ### Base Model (with SMOTE)

# ### Using Logit Function

# In[16]:


x_sm_scaled_const = sm.add_constant(x_sm_scaled)
logit_model = sm.Logit(y_sm,x_sm_scaled_const).fit()


# In[17]:


logit_model.summary()


# ### Using Sklearn

# In[22]:


log_reg = LogisticRegression()
log_reg.fit(x_sm_train,y_sm_train)


# In[242]:


y_sm_test_pred = log_reg.predict(x_sm_test)


# In[243]:


y_sm_test_prob_1 = log_reg.predict_proba(x_sm_test)[:,1]


# In[244]:


cm = confusion_matrix(y_sm_test,y_sm_test_pred)


# In[245]:


sns.heatmap(cm,annot=True,fmt='.0f');


# In[246]:


print(classification_report(y_sm_test,y_sm_test_pred))


# In[247]:


fpr, tpr, threshold = roc_curve(y_sm_test,y_sm_test_prob_1)
roc_df = pd.DataFrame([fpr,tpr,threshold],index=['FPR','TPR','Threshold']).T


# In[248]:


plt.plot(fpr,tpr)
plt.plot([[0,0],[1,1]])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC Curve')
plt.show()


# In[249]:


roc_auc_score(y_sm_test,y_sm_test_prob_1)


# ### For Cross Validation

# In[23]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=log_reg,X=x_sm_scaled,y=y_sm,cv=k,scoring='recall')
np.mean(scores)


# In[25]:


recall_log_reg_with_SMOTE = scores


# In[26]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=log_reg,X=x_sm_scaled,y=y_sm,cv=k,scoring='f1_weighted')
np.mean(scores)


# In[27]:


f1_weighted_log_reg_with_SMOTE = scores


# In[28]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=log_reg,X=x_sm_scaled,y=y_sm,cv=k,scoring='roc_auc')
np.mean(scores)


# In[29]:


roc_auc_log_reg_with_SMOTE = scores


# In[253]:


bias_error = 1 - np.mean(scores)
bias_error


# In[254]:


variance_error = np.std(scores,ddof=1) / np.abs(np.mean(scores))
variance_error


# In[ ]:





# # 2) KNN Classifier:

# ### Base Model (with SMOTE)

# In[ ]:





# In[30]:


knn = KNeighborsClassifier()
knn.fit(x_sm_train,y_sm_train)


# In[309]:


y_sm_test_pred = knn.predict(x_sm_test)


# In[310]:


y_sm_test_prob_1 = knn.predict_proba(x_sm_test)[:,1]


# In[311]:


cm = confusion_matrix(y_sm_test,y_sm_test_pred)


# In[312]:


sns.heatmap(cm,annot=True,fmt='.0f');


# In[313]:


print(classification_report(y_sm_test,y_sm_test_pred))


# In[263]:


fpr, tpr, threshold = roc_curve(y_sm_test,y_sm_test_prob_1)
roc_df = pd.DataFrame([fpr,tpr,threshold],index=['FPR','TPR','Threshold']).T


# In[264]:


plt.plot(fpr,tpr)
plt.plot([[0,0],[1,1]])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC Curve')
plt.show()


# In[265]:


roc_auc_score(y_sm_test,y_sm_test_prob_1)


# ### For Cross Validation

# In[31]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=knn,X=x_sm_scaled,y=y_sm,cv=k,scoring='recall')
np.mean(scores)


# In[32]:


recall_knn = scores


# In[33]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=knn,X=x_sm_scaled,y=y_sm,cv=k,scoring='f1_weighted')
np.mean(scores)


# In[34]:


f1_weighted_knn = scores


# In[35]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=knn,X=x_sm_scaled,y=y_sm,cv=k,scoring='roc_auc')
np.mean(scores)


# In[36]:


roc_auc_knn = scores


# In[269]:


bias_error = 1 - np.mean(scores)
bias_error


# In[270]:


variance_error = np.std(scores,ddof=1) / np.abs(np.mean(scores))
variance_error


# In[ ]:





# # 3) Random Forest:

# ### Base Model

# In[272]:


rf = RandomForestClassifier(random_state=10)
params = {'max_depth':np.arange(1,18)}

k = KFold(n_splits=5,shuffle=True,random_state=10)
grid_cv = GridSearchCV(estimator=rf,param_grid=params,cv=k,scoring='roc_auc')
grid_cv.fit(x_sm_train,y_sm_train)


# In[273]:


grid_cv.best_params_


# In[37]:


rf = RandomForestClassifier(max_depth=17,random_state=10)
rf.fit(x_sm_train,y_sm_train)


# In[275]:


y_sm_test_pred = rf.predict(x_sm_test)


# In[276]:


y_sm_test_prob_1 = rf.predict_proba(x_sm_test)[:,1]


# In[277]:


cm = confusion_matrix(y_sm_test,y_sm_test_pred)


# In[278]:


sns.heatmap(cm,annot=True,fmt='.0f');


# In[279]:


print(classification_report(y_sm_test,y_sm_test_pred))


# In[280]:


fpr, tpr, threshold = roc_curve(y_sm_test,y_sm_test_prob_1)
roc_df = pd.DataFrame([fpr,tpr,threshold],index=['FPR','TPR','Threshold']).T


# In[281]:


plt.plot(fpr,tpr)
plt.plot([[0,0],[1,1]])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC Curve')
plt.show()


# In[282]:


roc_auc_score(y_sm_test,y_sm_test_prob_1)


# ### For Cross Validation

# In[38]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=rf,X=x_sm_scaled,y=y_sm,cv=k,scoring='recall')
np.mean(scores)


# In[39]:


recall_rf = scores


# In[40]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=rf,X=x_sm_scaled,y=y_sm,cv=k,scoring='f1_weighted')
np.mean(scores)


# In[41]:


f1_weighted_rf = scores


# In[42]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=rf,X=x_sm_scaled,y=y_sm,cv=k,scoring='roc_auc')
np.mean(scores)


# In[43]:


roc_auc_rf = scores


# In[286]:


bias_error = 1 - np.mean(scores)
bias_error


# In[287]:


variance_error = np.std(scores,ddof=1) / np.abs(np.mean(scores))
variance_error


# In[ ]:





# # 4) AdaBoosting:

# In[ ]:





# In[44]:


ada = AdaBoostClassifier(n_estimators=100,random_state=10)
ada.fit(x_sm_train,y_sm_train)


# In[289]:


y_sm_test_pred = ada.predict(x_sm_test)


# In[290]:


y_sm_test_prob_1 = ada.predict_proba(x_sm_test)[:,1]


# In[291]:


cm = confusion_matrix(y_sm_test,y_sm_test_pred)


# In[292]:


sns.heatmap(cm,annot=True,fmt='.0f');


# In[293]:


print(classification_report(y_sm_test,y_sm_test_pred))


# In[294]:


fpr, tpr, threshold = roc_curve(y_sm_test,y_sm_test_prob_1)
roc_df = pd.DataFrame([fpr,tpr,threshold],index=['FPR','TPR','Threshold']).T


# In[295]:


plt.plot(fpr,tpr)
plt.plot([[0,0],[1,1]])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC Curve')
plt.show()


# In[296]:


roc_auc_score(y_sm_test,y_sm_test_prob_1)


# ### For Cross Validation

# In[45]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=ada,X=x_sm_scaled,y=y_sm,cv=k,scoring='recall')
np.mean(scores)


# In[46]:


recall_ada = scores


# In[47]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=ada,X=x_sm_scaled,y=y_sm,cv=k,scoring='f1_weighted')
np.mean(scores)


# In[48]:


f1_weighted_ada = scores


# In[49]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=ada,X=x_sm_scaled,y=y_sm,cv=k,scoring='roc_auc')
np.mean(scores)


# In[50]:


roc_auc_ada = scores


# In[300]:


bias_error = 1 - np.mean(scores)
bias_error


# In[301]:


variance_error = np.std(scores,ddof=1) / np.abs(np.mean(scores))
variance_error


# In[ ]:





# # 5) Gradient Boosting

# In[51]:


gb = GradientBoostingClassifier(random_state=10)
gb.fit(x_sm_train,y_sm_train)


# In[316]:


y_sm_test_pred = gb.predict(x_sm_test)


# In[317]:


y_sm_test_prob_1 = gb.predict_proba(x_sm_test)[:,1]


# In[318]:


cm = confusion_matrix(y_sm_test,y_sm_test_pred)


# In[319]:


sns.heatmap(cm,annot=True,fmt='.0f');


# In[321]:


print(classification_report(y_sm_test,y_sm_test_pred))


# In[322]:


fpr, tpr, threshold = roc_curve(y_sm_test,y_sm_test_prob_1)
roc_df = pd.DataFrame([fpr,tpr,threshold],index=['FPR','TPR','Threshold']).T


# In[323]:


plt.plot(fpr,tpr)
plt.plot([[0,0],[1,1]])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC Curve')
plt.show()


# In[324]:


roc_auc_score(y_sm_test,y_sm_test_prob_1)


# ### For Cross Validation

# In[52]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=gb,X=x_sm_scaled,y=y_sm,cv=k,scoring='recall')
np.mean(scores)


# In[53]:


recall_gb = scores


# In[54]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=gb,X=x_sm_scaled,y=y_sm,cv=k,scoring='f1_weighted')
np.mean(scores)


# In[55]:


f1_weighted_gb = scores


# In[56]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=gb,X=x_sm_scaled,y=y_sm,cv=k,scoring='roc_auc')
np.mean(scores)


# In[57]:


roc_auc_gb = scores


# In[328]:


bias_error = 1 - np.mean(scores)
bias_error


# In[329]:


variance_error = np.std(scores,ddof=1) / np.abs(np.mean(scores))
variance_error


# In[ ]:





# # 6) XGB

# In[366]:


xgb = XGBClassifier(verbosity=0,random_state=100)
params = {'max_depth':np.arange(1,18)}

k = KFold(n_splits=5,shuffle=True,random_state=10)
grid_cv = GridSearchCV(estimator=xgb,param_grid=params,cv=k,scoring='roc_auc')
grid_cv.fit(x_sm_train,y_sm_train)


# In[367]:


grid_cv.best_params_


# In[11]:


xgb = XGBClassifier(max_depth=11,verbosity=0,random_state=100)
xgb.fit(x_sm_train,y_sm_train)


# In[16]:


y_sm_test_pred = xgb.predict(x_sm_test)


# In[17]:


y_sm_test_prob_1 = xgb.predict_proba(x_sm_test)[:,1]


# In[18]:


cm = confusion_matrix(y_sm_test,y_sm_test_pred)


# In[19]:


sns.heatmap(cm,annot=True,fmt='.0f');


# In[20]:


print(classification_report(y_sm_test,y_sm_test_pred))


# In[374]:


fpr, tpr, threshold = roc_curve(y_sm_test,y_sm_test_prob_1)
roc_df = pd.DataFrame([fpr,tpr,threshold],index=['FPR','TPR','Threshold']).T


# In[375]:


plt.plot(fpr,tpr)
plt.plot([[0,0],[1,1]])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC Curve')
plt.show()


# In[376]:


roc_auc_score(y_sm_test,y_sm_test_prob_1)


# ### For Cross Validation

# In[59]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=xgb,X=x_sm_scaled,y=y_sm,cv=k,scoring='recall')
np.mean(scores)


# In[60]:


recall_xgb = scores


# In[61]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=xgb,X=x_sm_scaled,y=y_sm,cv=k,scoring='f1_weighted')
np.mean(scores)


# In[62]:


f1_weighted_xgb = scores


# In[63]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=xgb,X=x_sm_scaled,y=y_sm,cv=k,scoring='roc_auc')
np.mean(scores)


# In[64]:


roc_auc_xgb = scores


# In[380]:


bias_error = 1 - np.mean(scores)
bias_error


# In[381]:


variance_error = np.std(scores,ddof=1) / np.abs(np.mean(scores))
variance_error


# In[ ]:





# # 7) Stacking Classifier:

# In[65]:


rf = RandomForestClassifier(max_depth=17,random_state=10)
gb = GradientBoostingClassifier(random_state=10)

est = [('randomforest',rf),('gradientboost',gb)]

xgb = XGBClassifier(max_depth=11,verbosity=0,random_state=100)


# In[66]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
stack_cls  = StackingClassifier(estimators=est,final_estimator=xgb,cv=k)
stack_cls.fit(x_sm_train,y_sm_train)


# In[395]:


y_sm_test_pred = stack_cls.predict(x_sm_test)


# In[396]:


y_sm_test_prob_1 = stack_cls.predict_proba(x_sm_test)[:,1]


# In[397]:


cm = confusion_matrix(y_sm_test,y_sm_test_pred)


# In[398]:


sns.heatmap(cm,annot=True,fmt='.0f');


# In[399]:


print(classification_report(y_sm_test,y_sm_test_pred))


# In[400]:


fpr, tpr, threshold = roc_curve(y_sm_test,y_sm_test_prob_1)
roc_df = pd.DataFrame([fpr,tpr,threshold],index=['FPR','TPR','Threshold']).T


# In[401]:


plt.plot(fpr,tpr)
plt.plot([[0,0],[1,1]])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC Curve')
plt.show()


# In[402]:


roc_auc_score(y_sm_test,y_sm_test_prob_1)


# ### For Cross Validation

# In[67]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=stack_cls,X=x_sm_scaled,y=y_sm,cv=k,scoring='recall')
np.mean(scores)


# In[68]:


recall_stack_cls = scores


# In[69]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=stack_cls,X=x_sm_scaled,y=y_sm,cv=k,scoring='f1_weighted')
np.mean(scores)


# In[70]:


f1_weighted_stack_cls = scores


# In[71]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=stack_cls,X=x_sm_scaled,y=y_sm,cv=k,scoring='roc_auc')
np.mean(scores)


# In[72]:


roc_auc_stack_cls = scores


# In[406]:


bias_error = 1 - np.mean(scores)
bias_error


# In[407]:


variance_error = np.std(scores,ddof=1) / np.abs(np.mean(scores))
variance_error


# In[ ]:





# # 8) Voting Classifier:

# In[73]:


rf = RandomForestClassifier(max_depth=17,random_state=10)
gb = GradientBoostingClassifier(random_state=10)
xgb = XGBClassifier(max_depth=11,verbosity=0,random_state=100)

est = [('XGB',xgb),('randomforest',rf),('gradientboost',gb)]


# In[74]:


vc = VotingClassifier(estimators=est,voting='soft')
vc.fit(x_sm_train,y_sm_train)


# In[412]:


y_sm_test_pred = vc.predict(x_sm_test)
y_sm_test_prob_1 = vc.predict_proba(x_sm_test)[:,1]


# In[413]:


cm = confusion_matrix(y_sm_test,y_sm_test_pred)


# In[414]:


print(classification_report(y_sm_test,y_sm_test_pred))


# In[415]:


fpr, tpr, threshold = roc_curve(y_sm_test,y_sm_test_prob_1)
roc_df = pd.DataFrame([fpr,tpr,threshold],index=['FPR','TPR','Threshold']).T


# In[416]:


plt.plot(fpr,tpr)
plt.plot([[0,0],[1,1]])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC Curve')
plt.show()


# In[417]:


roc_auc_score(y_sm_test,y_sm_test_prob_1)


# ### For Cross Validation

# In[75]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=vc,X=x_sm_scaled,y=y_sm,cv=k,scoring='recall')
np.mean(scores)


# In[76]:


recall_vc = scores


# In[77]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=vc,X=x_sm_scaled,y=y_sm,cv=k,scoring='f1_weighted')
np.mean(scores)


# In[78]:


f1_weighted_vc = scores


# In[79]:


k = KFold(n_splits=5,shuffle=True,random_state=10)
scores = cross_val_score(estimator=vc,X=x_sm_scaled,y=y_sm,cv=k,scoring='roc_auc')
np.mean(scores)


# In[80]:


roc_auc_vc = scores


# In[421]:


bias_error = 1 - np.mean(scores)
bias_error


# In[422]:


variance_error = np.std(scores,ddof=1) / np.abs(np.mean(scores))
variance_error


# In[ ]:





# # recall_score comparison

# In[94]:


names = ['log_reg_without_SMOTE','log_reg_with_smote','knn','rf','ada','gb','xgb','stack_cls','vc']


# In[103]:


results_recall_1 = pd.DataFrame()

results_recall_1['recall'] = recall_log_reg_without_SMOTE
results_recall_1['Models'] = 'log_reg_without_SMOTE'


results_recall_1.reset_index(drop=True,inplace=True)


# In[104]:


results_recall_1


# In[106]:


results_recall_2 = pd.DataFrame()

results_recall_2['recall'] = recall_log_reg_with_SMOTE
results_recall_2['Models'] = 'log_reg_with_SMOTE'

results_recall_3 = pd.DataFrame()
results_recall_3['recall'] = recall_knn
results_recall_3['Models'] = 'knn'

results_recall_4 = pd.DataFrame()
results_recall_4['recall'] = recall_rf
results_recall_4['Models'] = 'rf'

results_recall_5 = pd.DataFrame()
results_recall_5['recall'] = recall_ada
results_recall_5['Models'] = 'ada'

results_recall_6 = pd.DataFrame()
results_recall_6['recall'] = recall_gb
results_recall_6['Models'] = 'gb'

results_recall_7 = pd.DataFrame()
results_recall_7['recall'] = recall_xgb
results_recall_7['Models'] = 'xgb'

results_recall_8 = pd.DataFrame()
results_recall_8['recall'] = recall_stack_cls
results_recall_8['Models'] = 'stack_cls'

results_recall_9 = pd.DataFrame()
results_recall_9['recall'] = recall_vc
results_recall_9['Models'] = 'vc'


# In[108]:


results_box = pd.concat([results_recall_1,results_recall_2,results_recall_3,results_recall_4,
                        results_recall_5,results_recall_6,results_recall_7,results_recall_8,
                        results_recall_9])


# In[113]:


plt.figure(figsize=(18,7))
sns.boxplot(x=results_box['Models'],y=results_box['recall']);


# # Overall Inference

# In[3]:


results_df = pd.DataFrame()


# In[4]:


results_df['recall_score'] = [0.22,0.76,0.81,0.87,0.81,0.82,0.88,0.86,0.87]
results_df['f1_weighted'] = [0.88,0.86,0.88,0.92,0.90,0.91,0.93,0.92,0.93]
results_df['roc_auc (%)'] = [87.28,92.80,93.58,97.65,95.89,96.72,98.07,97.45,97.91]
results_df['bias_error %(roc_auc)'] = [12.72,7.20,6.42,2.35,4.11,3.28,1.93,2.55,2.09]
results_df['variance_error %(roc_auc)'] = [0.83,0.33,0.23,0.16,0.27,0.17,0.11,0.17,0.13]


# In[5]:


results_df.index = ['Logistic Regression (without SMOTE)','Logistic Regression (with SMOTE)',
                   'KNN Classifier','Randm Forest','AdaBoost','Gradient Boost',
                    'XGB','Stacking Classifier','Voting Classifier']


# In[6]:


results_df


# In[15]:


sns.lineplot(x=results_df.index,y=results_df['recall_score'],marker='o',label='recall_score')
sns.lineplot(x=results_df.index,y=results_df['f1_weighted'],marker='o',label='f1_weighted')
plt.ylabel('score (0-1)')
plt.xticks(rotation=80);


# In[14]:


sns.lineplot(x=results_df.index,y=results_df['roc_auc (%)'],marker='o',label='roc_auc (%)')
plt.xticks(rotation=80);


# In[13]:


sns.lineplot(x=results_df.index,y=results_df['bias_error %(roc_auc)'],marker='o',label='bias_error')
sns.lineplot(x=results_df.index,y=results_df['variance_error %(roc_auc)'],marker='o',label='variance_error')
plt.ylabel('(%)')
plt.xticks(rotation=80);


# # Feature Importance:

# In[12]:


features_df = pd.DataFrame()
features_df['Importance'] = xgb.feature_importances_
features_df.index = x_sm.columns

features_df = features_df.sort_values('Importance',ascending=False)


# In[13]:


features_df


# In[16]:


plt.figure(figsize=(18,7))
sns.barplot(x=features_df['Importance'],y=features_df.index);


# # Done !!! (Refer Report and PPT for Overall Inference)

# In[ ]:





# In[ ]:




