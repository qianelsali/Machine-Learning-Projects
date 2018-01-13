"""
Created on Sat Dec 23th 15:24:17 2017
@author: Qian Li, Email: onehorselee@gmail.com
"""

''' Questions:
- What promotion would you suggest?
- What target group should we use?
- What exactly should we measure in order to determine if the test is successful?
- building a classifier to predict which players will convert naturally
'''

import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import os
import datetime

## Load the data
os.chdir('C:/Users/oneho/Documents/PocketGems')
iaps = pd.read_csv('iaps_201712072201.csv')
sessions = pd.read_csv('sessions_201712072202.csv')
spendevents = pd.read_csv('spendevents_201712072203.csv')
users = pd.read_csv('users_201712072205.csv')

## explore the percentage of payers and non-payers
len(spendevents[['udid']].drop_duplicates())/len(users)
len(iaps[['udid']].drop_duplicates())/len(users)

''' 
80% users (18017) virtually "spend' in games, however, only 7%  of users (1526) acturally purchased
'''


''' Create a dataframe with the following columns

we need to define a number of variables to depict the gaming behavior of both payers and non-payers. 
Hopefully we can use these variables to build a classifier to predict payer and non-payers.  

##### some info about payers
 variable name: defination
0) udid: user ID
1) payer: whether a user is payer or not. True or False
2) install_date: install date
3) convert_date: first date to pay
4) convert_gap: integer, convert_date - install_date
5) first_payment_sess: at which session did the first payment occurr
6) first_pur_prod_type: the first purchased product name
7) first_pur_prod_name: the first purchased product type

7.1) pay frequency 
7.2) pay value 
    
    
##### some other info about gaming behaviors of all gamers
8) login_freq: daily Login frequency
9) spendInGame: True or False, whether a user spends in game
10) spend_stories: which stories a user likes to spend in game
11) spend_chapters: which chapter a user likes to spend in game
    
'''
# 0,1,2 udid, payer, install_date
df = users[['udid', 'install_date']]
df['payer'] = df['udid'].isin(iaps
  ['udid'])

# 3, 4, 6, 7 convert_date, convert_prod_name, convert_prod_type, convert_rev
iaps['ts'] = pd.to_datetime(iaps['ts'])
convert_date = iaps[['udid', 'ts']].groupby('udid').min()
convert_date = convert_date.reset_index()
convert_date = pd.merge(convert_date, iaps, on = ['udid', 'ts'], how = 'left')

convert_date.drop_duplicates(inplace = True)

sessions['ts'] = pd.to_datetime(sessions['ts'])
sessions['date'] = sessions['ts'].dt.date
sessions['hour'] = sessions['ts'].dt.hour

convert_date['date'] = convert_date['ts'].dt.date
convert_date['hour'] = convert_date['ts'].dt.hour

convert_date = pd.merge(convert_date, 
                   sessions[['udid', 'date', 'hour', 'session_num']], 
                   on = ['udid', 'date', 'hour'], 
                   how = 'left')  

convert_date = convert_date.sort_values('session_num', ascending = False).drop_duplicates(subset=['udid', 'ts'], keep='last')

df = pd.merge(df, convert_date, on = 'udid', how = 'left')

df = df.rename(columns = {'ts': 'convert_date',
                          'prod_name': 'convert_prod_name',
                          'prod_type': 'convert_prod_type',
                          'rev': 'convert_rev',
                          'session_num': 'convert_session'})

    
# 7.1 pay frequency of each payer
pay_freq = iaps[['udid', 'ts']].groupby('udid').count()
pay_freq.columns = ['pay_frequency']
pay_freq = pay_freq.reset_index()

df = pd.merge(df, pay_freq, how = 'left', on = 'udid')

# 7.2 pay total value of each payer
iaps['rev'] = iaps['rev'].str.replace(',', '')
iaps['rev']  = iaps['rev'].astype(float)
pay_value = iaps[['udid', 'rev']].groupby('udid').sum()
pay_value.columns = ['pay_value_total']
pay_value = pay_value.reset_index()

df = pd.merge(df, pay_value, how = 'left', on = 'udid')

# 4 convert_gap
df['install_date'] = pd.to_datetime(df['install_date'])
df['convert_gap'] = df['convert_date'] - df['install_date']
df['convert_gap_days'] =  df['convert_gap'].dt.days 


# 9 whether spend virtual money in games
df['spendInGame'] = df['udid'].isin(spendevents['udid'])



### detect when a user stop playing games.
### Active days = last login session date - install date
#sessions = sessions2

sessions['session_num'] = sessions['session_num'].str.replace(',', '')
sessions['session_num'] = sessions['session_num'].astype(int)
grouped = sessions[['udid', 'session_num']].groupby('udid')
last_session = grouped['session_num'].apply(lambda x: x.max())
last_session = last_session.reset_index()
last_session = pd.merge(last_session, sessions[['udid', 'session_num','ts']], 
                        how = 'left', on = ['udid', 'session_num'])
last_session.columns = ['udid', 'last_session_num', 'last_session_ts']
last_session['last_session_ts'] = pd.to_datetime(last_session['last_session_ts'])

df = pd.merge(df, last_session, how = 'left', on = 'udid')
df.drop_duplicates(inplace = True)
df['active_days'] = df['last_session_ts'] - df['install_date'] 
df['active_days'] = df['active_days'].dt.days

# 8 login frequency 
df['login_fre'] = df['last_session_num']/(df['active_days'] + 1)
df['pay_value_per_day'] = df['pay_value_total'] / (df['active_days'] + 1)

def plot_hist(x, xlabel, ylabel, height):
    plt.figure(figsize = (20,8))
    plt.xlabel(xlabel,fontsize=25)
    plt.ylabel(ylabel,fontsize=25)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.hist(x.dropna(), bins = 100)
    a = x.quantile(0.99)
    plt.vlines(a, ymin = 0, ymax= height, linestyles='dotted', colors = 'red')
    b = x.quantile(0.95)
    plt.vlines(b, ymin = 0, ymax= height, linestyles='dotted', colors = 'red')
    c = x.quantile(0.75)
    plt.vlines(c, ymin = 0, ymax= height, linestyles='dotted', colors = 'red')
    d = x.quantile(0.50)
    plt.vlines(d, ymin = 0, ymax= height, linestyles='dotted', colors = 'red')
    print('50%:', d, '\n',
          '75%:', c,'\n',
          '95%:', b,'\n',
          '99%:', a)
    plt.show()

# active days
plot_hist(df['active_days'][df['payer'] == True],
         xlabel = 'Active Days of Payers',
         ylabel = 'Observed Users',
         height = 140)

# convert days
plot_hist(df['convert_gap_days'],
          xlabel = 'Convert gap days',
          ylabel = 'Observed Users',
          height = 800)

# convert session
plot_hist(df['convert_session'],
          xlabel = 'Convert session number',
          ylabel = 'Observed Users',
          height = 800)

# purchase frequency 
plot_hist(df['pay_frequency'],
          xlabel = 'Pay frequency of payers',
          ylabel = 'Observed Users',
          height = 800)

# purchase value
plot_hist(df['pay_value_total']/100,
          xlabel = 'Paid value of payers ($)',
          ylabel = 'Observed Users',
          height = 500)

plot_hist(df['pay_value_per_day']/100,
          xlabel = 'Paid value per active day of payers ($)',
          ylabel = 'Observed Users',
          height = 900)


# login_frequency between payers and non-payers
plot_hist(df[df['payer'] == True]['login_fre'],
          xlabel = 'Login frequency of payers',
          ylabel = 'Observed Users',
          height = 250)

plot_hist(df[df['payer'] == False]['login_fre'],
          xlabel = 'Login frequency of non-payers',
          ylabel = 'Observed Users',
          height = 6000)

# check the difference
stats.stats.ttest_ind(df[df['payer'] == True]['login_fre'].dropna(),
                      df[df['payer'] == False]['login_fre'].dropna(),
                      equal_var=False
                      )
# there is a significant difference between login frequency of non-payers and payers,
# so this variable should be included as a predicting variable


# =============================================================================
# ### examine which stories payer and non-payers like to ‘spend’ in games
# =============================================================================
spendevents['payer'] = spendevents['udid'].isin(iaps['udid'])
spendevents[spendevents['payer'] == True]['spendtype'].unique()
spendevents[spendevents['payer'] == True]['story'].unique()

'''
 spend stories: 32 in total
 spend type: 5 in total
'''

# check the spend type between payers and non-payers
spendtype = spendevents[spendevents['payer'] == True].groupby(['spendtype']).count()
spendtype = spendtype[['udid']]
spendtype.columns = ['payer_spend_ty_freq']
spendtype['percent'] = spendtype['payer_spend_ty_freq']/spendtype['payer_spend_ty_freq'].sum()

spendtype2 = spendevents[spendevents['payer'] == False].groupby(['spendtype']).count()
spendtype2 = spendtype2[['udid']]
spendtype2.columns = ['nonpayer_spend_ty_freq']
spendtype2['percent'] = spendtype2['nonpayer_spend_ty_freq']/spendtype2['nonpayer_spend_ty_freq'].sum()


# check the spend stories between payers and non-payers
spendstories = spendevents[spendevents['payer'] == True].groupby(['story']).count()
spendstories = spendstories[['udid']]
spendstories.columns = ['payer_spend_st_freq']
spendstories['percent'] = spendstories['payer_spend_st_freq']/spendstories['payer_spend_st_freq'].sum()

spendstories2 = spendevents[spendevents['payer'] == False].groupby(['story']).count()
spendstories2 = spendstories2[['udid']]
spendstories2.columns = ['nonpayer_spend_st_freq']
spendstories2['percent'] = spendstories2['nonpayer_spend_st_freq']/spendstories2['nonpayer_spend_st_freq'].sum()


''' we find that there are 4 stories and 3 story types that majority of users like to virtually 'spend':
Spendtype: 99% of population 
    earnGemsCounter, premiumchoice, IAP
    
Spend stories: 95% of population
    Mean_Girls_Version_D, Demi_M_Closet, Demi_Master, Demi_M_EuroTour

We will use these top 3 SpendType and top 4 Spend stories as predict variables
'''

# spend type: top 3, spend frequency and spend amount
spendevents['amount'] = spendevents['amount'].str.replace(',', '')
spendevents['amount'] = spendevents['amount'].astype(float)
# Extreme outliers were identified. 
spendevents = spendevents[spendevents['amount'] != -999999]


# =============================================================================
# ### limit spendevents at three days of installation
# =============================================================================
spendevents2 = pd.DataFrame()
sessions2 = pd.DataFrame()

for i in users['install_date'].unique():
    user_temp = users[users['install_date'] == i]
    user_temp['install_date'] = pd.to_datetime(user_temp['install_date'])
    
    spendevent_temp = spendevents[spendevents['udid'].isin(user_temp['udid'])]
    sessions_temp = sessions[sessions['udid'].isin(user_temp['udid'])]
    
    spendevent_temp['date'] = pd.to_datetime(spendevent_temp['date'])
    sessions_temp['date'] = pd.to_datetime(sessions_temp['date'])
    
    spendevent_temp = spendevent_temp[spendevent_temp['date'] <= datetime.datetime.strptime(i, '%Y-%m-%d') + datetime.timedelta(days=4)]
    sessions_temp = sessions_temp[sessions_temp['date'] <= datetime.datetime.strptime(i, '%Y-%m-%d') + datetime.timedelta(days=4)]

    spendevents2 = spendevents2.append(spendevent_temp)
    sessions2 = sessions2.append(sessions_temp)

spendevents = spendevents2 

######### extract spend types and spend stories
spendTypes = ['earnGemsCounter','IAP','premiumChoice']
for spendType in spendTypes:
    # spend frequency
    temp = spendevents[spendevents['spendtype'] == spendType][['udid', 'spendtype']]
    grouped = temp.groupby(['udid']).count()
    grouped.columns = [spendType]
    grouped = grouped.reset_index()
    df = pd.merge(df, grouped, how = 'left', on = 'udid')
    df[spendType] = df[spendType].fillna(0)
    # spend amount
    spendevents = spendevents[spendevents['spendtype'] != -999999]
    temp = spendevents[spendevents['spendtype'] == spendType][['udid', 'amount']]
    grouped = temp.groupby(['udid']).sum()
    grouped.columns = [spendType + "_amount"]
    grouped = grouped.reset_index()
    df = pd.merge(df, grouped, how = 'left', on = 'udid')
    df[spendType + "_amount"] = df[spendType + "_amount"].fillna(0)    

# spend story: top 4    
stories = ['Mean_Girls_Version_D', 'Demi_M_Closet', 'Demi_Master', 'Demi_M_EuroTour']
for story in stories:
    # spend frequency
    temp = spendevents[spendevents['story'] == story][['udid', 'story']]
    grouped = temp.groupby(['udid']).count()
    grouped.columns = [story]
    grouped = grouped.reset_index()
    df = pd.merge(df, grouped, how = 'left', on = 'udid')
    df[story] = df[story].fillna(0)
    # spend amount
    temp = spendevents[spendevents['story'] == story][['udid', 'amount']]
    grouped = temp.groupby(['udid']).sum()
    grouped.columns = [story + '_amount']
    grouped = grouped.reset_index()
    df = pd.merge(df, grouped, how = 'left', on = 'udid')
    df[story + '_amount'] = df[story + '_amount'].fillna(0)


### check the different between non-payer and payres using boxplot and t-test
def plot_boxplot(var1, var2):
    f, ax = plt.subplots(figsize=(10, 8))
    fig = sns.boxplot(x = var1,
                      y = var2,
                      data = df)
    plt.xlabel('Payer vs Non-payer',fontsize=30)
    plt.ylabel(var2, fontsize=30)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
#    fig.axis(ymin=0, ymax=df[var2][df[var1] == True].quantile(0.97))
    plt.show()
    

def t_test(var):
    print('t-test results of ', var)
    print(stats.stats.ttest_ind(df[var][df['payer'] == True].dropna(),
                          df[var][df['payer'] == False].dropna(),
                          equal_var=False), '\n')

def print_percentile(x):
    print('25%:', x.quantile(0.25))
    print('50%:', x.quantile(0.50))
    print('75%:', x.quantile(0.75))
    print('99%:', x.quantile(0.99))
    
    
var = 'payer'
spendTypes = ['earnGemsCounter','IAP','premiumChoice']
for spendType in spendTypes:
    # frequency
    plot_boxplot(var, spendType)
    t_test(spendTypes)
    # value
    plot_boxplot(var, spendType + '_amount')
    t_test(spendType + '_amount')
    
stories = ['Mean_Girls_Version_D', 'Demi_M_Closet', 'Demi_Master', 'Demi_M_EuroTour']
for story in stories:
    # frequency
    plot_boxplot(var, story)
    t_test(story)
    # value
    plot_boxplot(var, story + '_amount')
    print('Non-payer:', story)
    print_percentile(df[story + '_amount'][df['payer'] == False])
    print('Payer:', story)
    print_percentile(df[story + '_amount'][df['payer'] == True])


plot_boxplot(var, 'login_fre')
plot_boxplot(var, 'active_days')


print_percentile(df['login_fre'][df['payer'] == True])
print_percentile(df['login_fre'][df['payer'] == False])

# =============================================================================
# Modeling- Build a classifier
# =============================================================================
#prepare the data
data = df[df['spendInGame'] == True][df['active_days'] >= 12]
data = data[['payer', 'login_fre', 
           'earnGemsCounter','IAP','premiumChoice',
           'Mean_Girls_Version_D', 'Demi_M_Closet', 'Demi_Master','Demi_M_EuroTour'
           ]]

print('The percentage of payer in data:',
      "{0:.2f}%".format(len(data[data['payer'] == True])/len(data)*100))

import pickle 
pickle.dump(data,open('data.p', 'wb'))
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

from sklearn import linear_model
from sklearn.model_selection import train_test_split

y = data['payer'].astype(int)
x = data.drop('payer',axis = 1)

# =============================================================================
# # Logistic Regression
# =============================================================================
accuracy = []
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(
                                                      x, y,
                                                      test_size=0.33,
                                                      shuffle = True)    
    model = linear_model.LogisticRegression()
    model.fit(X_train, y_train)
    #print(model.score(X_test, y_test))
    accuracy.append(model.score(X_test, y_test))

print('Accuracy of classifer:', sum(accuracy)/len(accuracy))


'''
Our goal is to identify those not-yet-payers in the initial stage, now let’s run a more rigid test on the users whose active days are less than 12 days. 
'''
for i in range(1,12):
    print('active days:', i)
    test_data2 = df[df['spendInGame'] == True][df['active_days'] < i]
    test_data2 = test_data2[['payer', 'login_fre', 
               'earnGemsCounter','IAP','premiumChoice',
               'Mean_Girls_Version_D', 'Demi_M_Closet', 'Demi_Master','Demi_M_EuroTour'
               ]]
    test_y = test_data2['payer'].astype(int)
    test_x = test_data2.drop('payer', axis = 1)
    
    print(model.score(test_x, test_y))
    

test_data3 = df[df['spendInGame'] == True]
test_data3 = test_data3[['payer', 'login_fre', 
           'earnGemsCounter','IAP','premiumChoice',
           'Mean_Girls_Version_D', 'Demi_M_Closet', 'Demi_Master','Demi_M_EuroTour'
           ]]

test_data3.dropna(inplace = True)

test_y = test_data3['payer'].astype(int)
test_x = test_data3.drop('payer', axis = 1)

print(model.score(test_x, test_y))
  

'''
The average accuracy of the classifier is quite satisfying. 
So we can use it to predict whether a user will naturally convert or not.
Next step is to design different promotions towards 1) highly-likely payers and 2) less-likely payers
'''

# check the popular convert product types
for i in df[df['payer'] == True]['convert_prod_type'].unique():
    print("Percentage of ", i,
          "{0:.2f}%".format(len(df[df['convert_prod_type'] == i]) * 100/len(df['convert_prod_type'].dropna()))
          )











