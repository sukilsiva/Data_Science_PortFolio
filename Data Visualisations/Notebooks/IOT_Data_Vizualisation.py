#!/usr/bin/env python
# coding: utf-8

# # DATA VIZUALISATION

# ### IOT TEMPERATURE DATA READINGS

# #### 1.Importing Neccesary Libraries 

# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# In[2]:


###Read the file and take a first look at the various columns and the values contained therein. Also do check the size of the matrix/dataframe imported that you are going to work on.
df=pd.read_csv('IOTtemp.csv')
df.head(5)


# In[3]:


df.rename(columns = {'room_id/id':'room_id', 'out/in':'out_in'}, inplace = True)
df.head()


# In[4]:


def get_df_summary(df):
    
    '''This function is used to summarise especially unique value count and data type for variable'''
    
    unq_val_cnt_df = pd.DataFrame(df.nunique(), columns = ['unq_val_cnt'])
    unq_val_cnt_df.reset_index(inplace = True)
    unq_val_cnt_df.rename(columns = {'index':'variable'}, inplace = True)
    unq_val_cnt_df = unq_val_cnt_df.merge(df.dtypes.reset_index().rename(columns = {'index':'variable', 0:'dtype'}),
                                          on = 'variable')
    unq_val_cnt_df = unq_val_cnt_df.sort_values(by = 'unq_val_cnt', ascending = False)
    
    return unq_val_cnt_df


# In[5]:


unq_val_cnt_df = get_df_summary(df)
unq_val_cnt_df


# In[7]:


df.drop(columns = 'room_id', inplace = True)
print('No. of duplicate records in the data set : {}'.format(df.duplicated().sum()))


# In[8]:


df[df.duplicated()]


# In[10]:


# Drop duplicate records.
temp_iot_data = df.drop_duplicates()


# In[11]:


# Convert noted_date into date-time.
temp_iot_data['noted_date'] = pd.to_datetime(temp_iot_data['noted_date'], format = '%d-%m-%Y %H:%M')


# In the absence of seconds component from noted_date variable values, the given data set would give a perception of Data Duplicacy or Data Redundancy for the combination of noted_date, out_in & temp variables.
# 
# How do we handle this situation?

# In[12]:


#Find out a variable (if any) in the given data set that can be used to sort the data such that the sorted data set is in the order as if it is sorted by noted_date variable.

# Check data duplicacy based on noted_date variable.

temp_iot_data.groupby(['noted_date'])['noted_date'].count().sort_values(ascending = False).head()


# In[13]:


temp_iot_data.loc[temp_iot_data['noted_date'] == pd.to_datetime('2018-09-12 03:09:00', format = '%Y-%m-%d %H:%M:%S'), ].sort_values(by = 'id').head(10)


# In[14]:


temp_iot_data['id'].apply(lambda x : x.split('_')[6]).nunique() == temp_iot_data.shape[0]


# Yes, the numerical part of id can be used as a primary key to uniquely identify the observations in the given data set.
# 
# Let's further analyse numerical part of id in order to re-confirm if it can be used to sort the given data set.
# 

# In[15]:


temp_iot_data['id_num'] = temp_iot_data['id'].apply(lambda x : int(x.split('_')[6]))
temp_iot_data.head()


# In[16]:


temp_iot_data.loc[temp_iot_data['id_num'].isin(range(17003, 17007))].sort_values(by = 'id_num')


# There is no data for id_num = 17005.
# Also, id_num = 17004 observation has been recorded at "2018-09-12 03:08:00" while observation for id_num = 17003 has been recorded at "2018-09-12 03:09:00". However, it's expected to have noted_date for former later to id_num = 17003.

# In[17]:


temp_iot_data.loc[temp_iot_data['id_num'].isin(range(17006, 17010))].sort_values(by = 'id_num')


# In[18]:


temp_iot_data.loc[temp_iot_data['noted_date'] == pd.to_datetime('2018-09-09 16:24:00', format = '%Y-%m-%d %H:%M:%S'), ].sort_values(by = 'id_num').head(10)


# In[19]:


temp_iot_data.loc[temp_iot_data['id_num'].isin(range(4000, 4003))].sort_values(by = 'id_num')


# In[20]:


temp_iot_data.loc[temp_iot_data['id_num'].isin(range(4002, 4007))].sort_values(by = 'id_num')


# Observation with id_num = 4004 is expected to have noted_date same as that of 4002 and 4006.
# 
# Based on the above observations, we can conclude that id_num variable cannot be used to sort the data set to get the actual data in sorted order in spite of absence of seconds component in the noted_date variable.
# 
# We'll use id_num as primary key to identify the observations uniquely. Replace id variable values with id_num and drop id_num variable from the data set.
# 

# In[21]:


temp_iot_data.loc[:, 'id'] = temp_iot_data.loc[:, 'id_num']


# In[22]:


temp_iot_data.loc[:, 'id'] = temp_iot_data.loc[:, 'id_num']
# Drop id_num column from the data set.

temp_iot_data.drop(columns = 'id_num', inplace = True)


# In[23]:


print('No. of years data : {}'.format(temp_iot_data['noted_date'].dt.year.nunique()))


# In[24]:


print('No. of months data : {}'.format(temp_iot_data['noted_date'].dt.month.nunique()))


# In[25]:


sorted(temp_iot_data['noted_date'].dt.month.unique())


# We have got data for only second half of the year 2018.

# In[26]:


print('No. of days data : {}'.format(temp_iot_data['noted_date'].dt.day.nunique()))


# In[27]:


temp_iot_data['month'] = temp_iot_data['noted_date'].apply(lambda x : int(x.month))


# In[28]:


temp_iot_data['day'] = temp_iot_data['noted_date'].apply(lambda x : int(x.day))


# In[29]:


temp_iot_data['day_name'] = temp_iot_data['noted_date'].apply(lambda x : x.day_name())


# In[30]:


temp_iot_data['hour'] = temp_iot_data['noted_date'].apply(lambda x : int(x.hour))
print(sorted(temp_iot_data['hour'].unique()))


# In[31]:


temp_iot_data.head()


# Let's assume this data has been recorded in India. Based on this assumption, we can presume two very important things:
# 
# (A) Climatological Seasons:
#   India Meteorological Department (IMD) follows the international standard of four climatological seasons with some local adjustments:
# 
#   a. Winter (December, January and February).
#   b. Summer (March, April and May).
#   c. Monsoon means rainy season (June to September).
#   d. Post-monsoon period (October to November).
# 
# Accordingly, we will create another variable season to hold the season which we are going derive based on month variable value.
# 
# 
# (B) Unit of measurement used to measure temp**.
# 
#   As India follows SI units system of measurement, we assume that the temperature is recorded in degree celsius.
# 

# In[32]:


def map_month_to_seasons(month_val):
    if month_val in [12, 1, 2]:
        season_val = 'Winter'
    elif month_val in [3, 4, 5]:
        season_val = 'Summer'
    elif month_val in [6, 7, 8, 9]:
        season_val = 'Monsoon'
    elif month_val in [10, 11]:
        season_val = 'Post_Monsoon'
    
    return season_val
temp_iot_data['season'] = temp_iot_data['month'].apply(lambda x : map_month_to_seasons(x))
temp_iot_data['season'].value_counts(dropna = False)


# Name: season, dtype: int64
# Since, we have data for 2nd half of year 2018 only, we see Monsoon, Post_Monsoon and Winter in season variable.

# In[33]:


temp_iot_data['month_name'] = temp_iot_data['noted_date'].apply(lambda x : x.month_name())
# temp_iot_data['month_name'].value_counts(dropna = False)


# Let's bin the hour into four different timings i.e. Night, Morning, Afternoon and Evening.
# 
# Night : 2200 - 2300 Hours & 0000 - 0359 Hours
# Morning : 0400 - 1159 Hours
# Afternoon : 1200 - 1659 Hours
# Evening : 1700 - 2159 Hours

# In[34]:


def bin_hours_into_timing(hour_val):
    
    if hour_val in [22,23,0,1,2,3]:
        timing_val = 'Night (2200-0359 Hours)'
    elif hour_val in range(4, 12):
        timing_val = 'Morning (0400-1159 Hours)'
    elif hour_val in range(12, 17):
        timing_val = 'Afternoon (1200-1659 Hours)'
    elif hour_val in range(17, 22):
        timing_val = 'Evening (1700-2159 Hours)'
    else:
        timing_val = 'X'
        
    return timing_val


# In[35]:


temp_iot_data['timing'] = temp_iot_data['hour'].apply(lambda x : bin_hours_into_timing(x))
temp_iot_data['timing'].value_counts(dropna = False)


# ### 2.Exploratory Data Analysis 

# Do the Data Analysis for Inside and Outside temperatures, separately.
# 
# How is overall temperature variation across months inside and outside room?

# In[38]:


fig = px.box(temp_iot_data, x = 'out_in', y = 'temp', labels = {'out_in':'Outside/Inside', 'temp':'Temperature'})
fig.update_xaxes(title_text = 'In or Out')
fig.update_yaxes(title_text = 'Temperature (in degree celsius)')
fig.update_layout(title = 'Overall Temp. Variation Inside-Outside Room')
fig.show()


# Overall Temp. Variation Inside-Outside Room
# In or Out
# Temperature (in degree celsius)
# Observations: (assuming there are no outliers)
# 
# Temperature recorded inside room :
#  Min. temperature : 21°C.
#  Max. temperature : 41°C.
# Temperature recorded outside room :
#  Min. temperature : 24°C.
#  Max. temperature : 51°C.
# Average tempurature recorded inside the room < Average temperature recorded outside the room. This is an obvious thing to observe.
# Temperature has varied alot outside room when compared to inside.
# Outside room : Magnitude of temperature variation before and after 37°C is almost same. However, temperature has varied a lot after reaching 40°C in comparison to temperature variation upto 31°C.
# How temperature varies across seasons?

# In[39]:


fig = px.box(temp_iot_data, 
             x = 'season', 
             y = 'temp', 
             color = 'out_in', 
             labels = {'out_in':'Outside/Inside', 'temp':'Temperature', 'season':'Season'})
fig.update_xaxes(title_text = 'Inside/Outside - Season')
fig.update_yaxes(title_text = 'Temperature (in degree celsius)')
fig.update_layout(title = 'Season-wise Temp. Variation')
fig.show()


# Season-wise Temp. Variation
# Inside/Outside - Season
# Temperature (in degree celsius)
# Observations: (assuming there are no outliers)
# 
# Max. temperature of 51°C has been recorded in Monsoon season which is quite surprising and not expected in rainy season.
# Note: We have to yet see when was this temperature was recorded; Is it at the start of monsoon season or at the end of the season?
# As usual the lowest temperature of 21°C has been recorded in Winter season.
# Magnitude of temperature variation is observed inside room in Monsoon season is higher compared to Winter and Post Monsoon season.
# Similary, maximum temperature variation outside room is observed in Monsoon season.
# In comparison to average (median) tem

# In[40]:


fig = px.box(temp_iot_data, x = 'month_name', y = 'temp', 
             category_orders = {'month_name':['July', 'August', 'September', 'October', 'November', 'December']},
             color = 'out_in')
fig.update_xaxes(title_text = 'Inside/Outside Month')
fig.update_yaxes(title_text = 'Temperature (in degree celsius)')
fig.update_layout(title = 'Monthly Temp. Variation')
fig.show()


# Volume of data we have for July and August months is very very low compared to other months.
# Maximum temperature variations are observed in September month both inside and outside the room.
# Highest average temperature (median) of 39°C is observed in November months.
# Lowest temperature of 21°C is recorded in December month.
# Despite of Point No. 1, Minimum temperature variation is observed in July and August months.
# How temperature varies for different timings for all seasons?

# In[41]:


temp_iot_data.head()


# In[42]:


for in_out_val in ['In', 'Out']:

    fig = px.box(temp_iot_data.loc[temp_iot_data['out_in'] == in_out_val], x = 'month_name', y = 'temp', 
                 category_orders = {'month_name':['July', 'August', 'September', 'October', 'November', 'December'], 
                                    'timing':['Morning (0400-1159 Hours)', 'Afternoon (1200-1659 Hours)', 'Evening (1700-2159 Hours)', 'Night (2200-0359 Hours)']},
                 hover_data = ['hour'],
                 labels = {'timing':'Timing', 'hour':'Hour', 'month_name':'Month', 'temp':'Temperature'},
                 color = 'timing')
    fig.update_xaxes(title_text = 'Month-Day Timings')
    fig.update_yaxes(title_text = 'Temperature (in degree celsius)')
    fig.update_layout(title = 'Temperature Variation in a Day (' + in_out_val + ')')
    fig.show()


# Temperature Variation in a Day (Out)
# Month-Day Timings
# Temperature (in degree celsius)
# Observations: (assuming there are no outliers)
# (A) Inside room:
# 
# September month : Maximum temperature variation is observed in morning & evening.
# October month : Highest average (median) temperature of 33°C has been recorded during evening & night.
# Lowest temperature of 21°C is recorded in December month in the morning.
# Highest temperature of 41°C is recorded in September month in the afternoon between 1400-1500 hours.
# (B) Outside room:
# 
# September month : Maximum temperature variation is observed during afternoon.
# November month : Highest average (median) temperature of 42°C has been recorded in morning.
# Lowest temperature of 24°C is recorded in September month in the afternoon.
# Highest temperature of 51°C is recorded in September month in the evening between 1700-1800 hours.

# In[45]:


tmp_df = round(temp_iot_data.groupby(['out_in', 'month', 'month_name', 'hour'])['temp'].mean(), 1).reset_index()
tmp_df.head()


# In[46]:


for out_in_val in ['In', 'Out']:

    fig = go.Figure()
    
    for mth in range(9, 13):
    
        mth_name = pd.to_datetime('01' + str(mth) + '2019', format = '%d%m%Y').month_name()
        filter_cond = ((tmp_df['month'] == mth) & (tmp_df['out_in'] == out_in_val))

        fig.add_trace(go.Scatter(x = tmp_df.loc[filter_cond, 'hour'],
                                 y = tmp_df.loc[filter_cond, 'temp'],
                                 mode = 'lines+markers',
                                 name = mth_name))
    
    fig.update_xaxes(tickvals = list(range(0, 24)), ticktext = list(range(0, 24)), title = '24 Hours')
    fig.update_yaxes(title = 'Temperature (in degree Celsius)')
    fig.update_layout(title = 'Hourly Avg. Temperature for each month (' + out_in_val + ')')
    fig.show()


# Temperature (in degree Celsius)
# Observations:
# (A) Inside room:
# 
# October month : Saw Highest Average (Median) temperature of 33.6°C between 2000:2059 hours.
# December month : Saw Lowest Average (Median) temperature of 26.9°C between 0400:0459 hours.
# (B) Outside room:
# 
# October month : Saw the Highest Average (Median) temperature of 46.6°C between 0800:0859 hours.
# December month : Saw the Lowest Average (Median) temperature of 29.4°C between 1900:1959 hours.
# Compared to September, October & November months, December month's average temperature per hour has always been on lower side.

# In[48]:


tmp_df = round(temp_iot_data.groupby(['out_in', 'month', 'month_name', 'day_name'])['temp'].mean(), 1).reset_index()
tmp_df.head()


# In[49]:


for out_in_val in ['In', 'Out']:

    fig = go.Figure()
    
    for mth in range(9, 13):
    
        mth_name = pd.to_datetime('01' + str(mth) + '2019', format = '%d%m%Y').month_name()
        filter_cond = ((tmp_df['month'] == mth) & (tmp_df['out_in'] == out_in_val))

        fig.add_trace(go.Scatter(x = tmp_df.loc[filter_cond, 'day_name'],
                                 y = tmp_df.loc[filter_cond, 'temp'],
                                 mode = 'markers',
                                 name = mth_name,
                                 marker = dict(size = tmp_df.loc[filter_cond, 'temp'].tolist())                                 
                                ))
    
    fig.update_xaxes(title = 'Day', categoryarray = np.array(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']))
    fig.update_yaxes(title = 'Temperature (in degree Celsius)')
    fig.update_layout(title = 'Day-wise Avg. Temperature for each month (' + out_in_val + ')')
    fig.show()


# In[50]:


tmp_df = temp_iot_data.groupby(['noted_date', 'out_in'])['temp'].mean().round(1).reset_index()


# In[51]:


for out_in_val in ['In', 'Out']:

    filter_cond = (tmp_df['out_in'] == out_in_val)

    fig.add_trace(go.Scatter(x = tmp_df.loc[filter_cond, 'noted_date'],
                             y = tmp_df.loc[filter_cond, 'temp'],
                             mode = 'lines',
                             name = out_in_val))
    
fig.update_xaxes(title = 'Noted Date')
fig.update_yaxes(title = 'Temperature (in degree Celsius)')
fig.update_layout(title = 'Day-wise Temperature')
fig.show()


# ### 3.Conclusion

# Based on Data we can find the temperature of IOT Devices and control electricity Measures
