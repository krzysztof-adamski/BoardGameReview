#Board Game review using linear regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("https://raw.githubusercontent.com/Maduflavins/Board-Game-Review/master/games.csv")
df.columns


plt.hist(df['average_rating'])
plt.show()

print(df[df['average_rating']>0].iloc[0].values)

df=df[df['users_rated']>0]
df=df.dropna(axis=0)


corrmat=df.corr()
fig=plt.figure(figsize=(14,10))
sns.heatmap(corrmat,vmax=.8,square=True)



#X=df.iloc[:,:].values
X=pd.DataFrame(df.drop(columns=['bayes_average_rating','average_rating','type','name','id']))
Y=pd.DataFrame(df['average_rating'])
X=X.iloc[:,:].values
Y=Y.iloc[:,:].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size= 0.2,random_state=0)



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
regressor1=LinearRegression()
regressor1.fit(x_train,y_train)
y1_pred=regressor1.predict(x_test)
mean_squared_error(y1_pred,y_test)

r2_score(y_test,y1_pred)

from sklearn.ensemble import RandomForestRegressor
regressor2=RandomForestRegressor(n_estimators=10000, min_samples_leaf=10)
y1=y_train.ravel()
regressor2.fit(x_train,y_train)
y2_pred=regressor2.predict(x_test)
mean_squared_error(y2_pred,y_test)

r2_score(y_test,y2_pred)



