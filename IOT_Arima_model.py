
# coding: utf-8

# IOT ARIMA MODEL BUILDING (https://www.analyticsvidhya.com/blog/2018/08/auto-arima-time-series-modeling-python-r/)

# In[1]:


import pandas as pd
from datetime import datetime
from matplotlib import pyplot


# In[2]:


temp = pd.read_csv("TempData.csv") 


# In[3]:


temp


# In[4]:


df1=temp.loc[0:1247,:]


# In[5]:


df1


# In[6]:


df2=temp.loc[1248:,:]
df2


# In[7]:


df1.dtypes


# In[8]:


df2.dtypes


# In[9]:


df1['Date'] = pd.to_datetime(df1['Date'])
df2['Date'] = pd.to_datetime(df2['Date'])

 


# In[10]:


df1.dtypes


# In[11]:


df2.dtypes


# In[12]:


df1.head()


# In[13]:


df2.head()


# In[38]:


result=df1.append(df2)
result.head()


# In[ ]:





# In[39]:


dff=result['Date'].apply(lambda x: x.strftime('%Y-%b'))
dff.head()


# In[41]:


resultfinal = pd.concat([dff,result['Avg Temp.']],axis=1)
resultfinal.head()


# In[17]:


#Check for null values
resultfinal.isnull().sum()


# In[18]:


#Replace null values by mean
resultfinal['Avg Temp.'] = resultfinal['Avg Temp.'].fillna(resultfinal['Avg Temp.'].mean())
resultfinal.isnull().sum()


# In[19]:


import matplotlib.pyplot as plt

#Got last 9 years data(108 records)
x=resultfinal['Date'].loc[2504:2612:]       
y=resultfinal['Avg Temp.'].loc[2504:2612:]

plt.figure(figsize=(20, 8))
plt.plot(x,y )
plt.xticks(rotation=90)    #Rotate xlabels vertically
plt.xlabel("Month")
plt.ylabel("Avg Temperature monthly")


plt.show()


# In[20]:


from statsmodels.tsa.stattools import adfuller

result = adfuller(resultfinal['Avg Temp.'].loc[2504:2612:])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# p-value>0.05 so data is not stationary


# In[21]:


#divide into train and validation set
train = resultfinal.loc[2504:2612:][:int(0.7*(len(resultfinal.loc[2504:2612:])))]
valid = resultfinal.loc[2504:2612:][int(0.7*(len(resultfinal.loc[2504:2612:]))):]
valid.head()


# In[22]:



plt.figure(figsize=(20, 8))
plt.plot(train['Date'],train['Avg Temp.'] )
plt.plot(valid['Date'],valid['Avg Temp.'] )

plt.xticks(rotation=90)    #Rotate xlabels vertically
plt.xlabel("Month")
plt.ylabel("Avg Temperature monthly")


plt.show()


# In[23]:


#building the model
from pyramid.arima import auto_arima
model = auto_arima(train['Avg Temp.'], trace=True, error_action='ignore', suppress_warnings=True)


# In[24]:


model.fit(train['Avg Temp.'])


# In[25]:


forecast = model.predict(n_periods=len(valid['Avg Temp.']))
forecast = pd.DataFrame(forecast,index = valid['Avg Temp.'].index,columns=['Prediction'])
forecast.head()


# In[26]:




#plot the predictions for validation set
plt.figure(figsize=(20, 8))
plt.plot(train['Date'],train['Avg Temp.'])
plt.plot(valid['Date'],valid['Avg Temp.'])
plt.plot(valid['Date'],forecast['Prediction'])

plt.xticks(rotation=90)    #Rotate xlabels vertically
plt.xlabel("Month")
plt.ylabel("Avg Temperature monthly")


plt.show()


# In[27]:


#calculate rmse
from math import sqrt
from sklearn.metrics import mean_squared_error

rms = sqrt(mean_squared_error(valid['Avg Temp.'],forecast))
print(rms)


# PREDICT FROM 2013-oct to 2013-May
# 
# Data is from this website. typed into csv file and imported it
# https://www.worldweatheronline.com/colombo-weather-averages/western/lk.aspx 
# 
# 

# In[28]:


tempfrom2013sepTo2019Apr=pd.read_csv("TempUntilApril.csv")
tempfrom2013sepTo2019Apr.head()


# In[29]:


tempfrom2013sepTo2019Apr.dtypes


# In[30]:


tempfrom2013sepTo2019Apr['Avg Temp.'] = tempfrom2013sepTo2019Apr['Avg Temp.'].astype('float64')
tempfrom2013sepTo2019Apr['Date'] = pd.to_datetime(tempfrom2013sepTo2019Apr['Date'])

tempfrom2013sepTo2019Apr.dtypes


# In[31]:


tempfrom2013sepTo2019Apr.head()


# In[44]:


test = tempfrom2013sepTo2019Apr['Date'].apply(lambda x: x.strftime('%Y-%b'))
test.head()


# In[46]:


testfinal = pd.concat([test,tempfrom2013sepTo2019Apr['Avg Temp.']],axis=1)
testfinal.head()


# In[51]:


# predict from 2013-oct to 2019-May  

forecastTest = model.predict(n_periods=68)
forecastTest = pd.DataFrame(forecastTest,columns=['PredictionUntilThisMonth'])
forecastTest.head()


# In[55]:


#Compare test results with predicted results from 2013-sep to 2019-Apr 
plt.figure(figsize=(20, 8))
plt.plot(testfinal['Date'],testfinal['Avg Temp.'])
plt.plot(testfinal['Date'],forecastTest[0:67:])

plt.xticks(rotation=90)    #Rotate xlabels vertically
plt.xlabel("Month")
plt.ylabel("Avg Temperature monthly")


plt.show()


# In[56]:


#calculate rmse
from math import sqrt
from sklearn.metrics import mean_squared_error

rms = sqrt(mean_squared_error(testfinal['Avg Temp.'],forecastTest[0:67:]))
print(rms)


# In[57]:


#Compare predicted 2019-May result with sensor data
mayAvgTempPredicted = forecastTest[67:68:]
mayAvgTempPredicted

