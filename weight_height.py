#project for predicting weight according to height of a person grouped by gender
import pandas as pd
df=pd.read_csv("project\weight-height.csv")
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.info())
print(df['Gender'].nunique()) #number of unique category column gender has
print(df['Gender'].unique()) #unique values of column gender

import matplotlib.pyplot as plt
plt.style.use('ggplot')

#histogram of height
df.Height.plot(kind='hist', color='purple', edgecolor='pink', figsize=(5,4))
plt.title('Distribution of height', size=15)
plt.xlabel('Height(inches)', size=10)
plt.ylabel('Frequency', size=10)


#histogram of weight
df.Weight.plot(kind='hist', color='teal', edgecolor='pink', figsize=(6,5))
plt.title('Distribution of weight', size=15)
plt.xlabel('Weight(pounds)', size=12)
plt.ylabel('Frequency',size=12)
plt.show()

#histogram of the height- males and females
df[df['Gender']=='Male'].Height.plot(kind='hist', color='pink', edgecolor='red',alpha=0.9, figsize=(5,4))
df[df['Gender']=='Female'].Height.plot(kind='hist', color='teal', edgecolor='pink',alpha=0.9, figsize=(5,4))
plt.legend(labels=['Males','Females'])
plt.title('Distribution of height', size=15)
plt.xlabel('Height(inches)', size=10)
plt.ylabel('Frequency', size=10)
#histogram of weight male and females
df[df['Gender']=='Female'].Weight.plot(kind='hist', color='blue', edgecolor='black', alpha=0.8, figsize=(6,5))
df[df['Gender']=='Male'].Weight.plot(kind='hist', color='magenta', edgecolor='black', alpha=0.8, figsize=(6,5))
plt.title('Distribution of weight', size=15)
plt.xlabel('Weight(pounds)',size=10)
plt.ylabel('Frequency',size=10)
plt.legend(labels=['Female','Male'])

plt.show()

#descriptive statistic male
sm=df[df['Gender']=='Male'].describe()
sm.rename(columns={'Height':'Height_male','Weight':'Weight_male'},inplace=True)
print(sm)

#descriptive statistic female
sf=df[df['Gender']=='Female'].describe()
sf.rename(columns={'Height':'Height_female','Weight':'Weight_female'},inplace=True)
print(sf)

#dataframe that concats statistics for both male and female
statistics=pd.concat([sm,sf], axis=1)
print(statistics)

#scatter plot of height and weight
ax1=df[df['Gender']=='Male'].plot(kind='scatter', x='Height', y='Weight', color='blue', alpha=0.5, figsize=(6,5))
df[df['Gender']=='Female'].plot(kind='scatter',x='Height', y='Weight', color='magenta', alpha=0.5, figsize=(6,5), ax=ax1)
plt.legend(labels=['Males','females'])
plt.title('Relationship b/w Height and Weight',size=15)
plt.xlabel('Height',size=10)
plt.ylabel('Weight',size=10)

#scatter plot of 500 females
s=df[df['Gender']=='Female'].sample(500)
s.plot(kind='scatter', x='Height', y='Weight', color='magenta',alpha=0.5,figsize=(6,5))
plt.title('Relationship b/w Height and Weight (sample of 500 females)', size=12)
plt.xlabel('height(inches)', size=12)
plt.ylabel('Weight(pounds)',size=12)
plt.show()

#scatter plot of 500 males
s=df[df['Gender']=='Male'].sample(500)
s.plot(kind='scatter', x='Height', y='Weight', color='red',alpha=0.5,figsize=(6,5))
plt.title('Relationship b/w Height and Weight (sample of 500 females)', size=12)
plt.xlabel('height(inches)', size=12)
plt.ylabel('Weight(pounds)',size=12)
plt.show()

#df containing only female
dfF=df[df['Gender']=='Female']

#correlation coefficient
print(dfF['Height'].corr(dfF['Weight']))

#df containing only male
df_Male=df[df['Gender']=='Male']

#correlation coefficient
print(df_Male['Height'].corr(df_Male['Weight']))

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
import seaborn as sns
x=df_Male[['Height']]
y=df_Male[['Weight']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=123)
print('Training feature set size:',x_train.shape)
print('Test feature set size:',x_test.shape)
print('Training feature set size:',y_train.shape)
print('Test feature set size:',y_test.shape)

#create lr obj
lr_male=LinearRegression() #model
lr_male.fit(x_train,y_train)
LinearRegression()

#get slope and intercept of the line best fit
print("intercept: ",lr_male.intercept_)
print("slope: ",lr_male.coef_)

#Prediction
predictions=lr_male.predict(x_test)

#visualize
plt.figure(figsize=(8,6))
plt.title('Actual VS Predicted Male Values')
plt.xlabel('Height',size=18)
plt.ylabel('Weight',size=18)
plt.scatter(x=x_test,y=y_test)
plt.plot(x_test, predictions, color='Blue', linewidth=3)
plt.show()

#regression evaluation metrics
print('Mean abs error(MAE): ',metrics.mean_absolute_error(y_test,predictions))
print('Mean Square Error (MSE): ',metrics.mean_squared_error(y_test,predictions))
print('Root mean square error(RMSE)',metrics.root_mean_squared_error(y_test,predictions))
#r-squared value
print('R-squared value of predictions:',round(metrics.r2_score(y_test,predictions),2))

#cross check first few actual values and predicted values
df_check=pd.DataFrame({'Actual':y_test['Weight'][:10].values,'Predicted':predictions[:10].ravel()})
print(df_check)

#plot residuals
r=np.subtract(y_test,predictions)
#r.plot(kind='hist',edgecolor='blue')
sns.displot(r)
plt.title('Histogram of residuals')
plt.xlabel('Residual value')
plt.ylabel('count')
plt.show()

h=[58.3,60,61.2,64.8,67.2]
fam=np.array([h]).reshape(5,1)
print(fam)

print(lr_male.predict(fam))

#training model for female
df_Female=df[df['Gender']=='Female']
a=df_Female[['Height']]
b=df_Female[['Weight']]
a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=0.3,random_state=123)
print(a_train.shape)
lr_female=LinearRegression()
lr_female.fit(a_train,b_train)
LinearRegression()
p=lr_female.predict(a_test)
plt.title('Actual vs predicted female weight')
plt.scatter(x=a_test,y=b_test)
plt.plot(a_test,p, color='teal', linewidth=1)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()

f=pd.DataFrame({'Actual':df_Female['Weight'][:10].values,'Predicted':p[:10].ravel()})
print(f)
print('Mean Abs error:',round(metrics.mean_absolute_error(b_test,p),2))
print('MSE: ',round((metrics.mean_squared_error(b_test,p)),2))
print('RMSE: ',round((metrics.root_mean_squared_error(b_test,p)),2))

print(lr_female.predict(fam))





