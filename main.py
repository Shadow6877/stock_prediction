#from pyexpat import model
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as df
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, LSTM
import streamlit as st

st.title('Stock Trend Prediction')


options = st.sidebar.selectbox(
     'Select the stock : ',
     ['GAIL.NS', 'SBIN.NS'])

st.write('You have selected : ',options)

end_date1 = st.sidebar.date_input(
     'Select the date for prediction : '
     )

end_date = end_date1-dt.timedelta(days=30)
start_date = end_date-dt.timedelta(days=365*8)

actual_stock = df.DataReader(options, 'yahoo', end_date+dt.timedelta(days=29), end_date1)

end_date1 = dt.datetime.strptime(str(end_date1), "%Y-%m-%d").date()
end_date = dt.datetime.strptime(str(end_date), "%Y-%m-%d").date()
start_date = dt.datetime.strptime(str(start_date), "%Y-%m-%d").date()

data = df.DataReader(options, 'yahoo', start_date, end_date)


st.subheader('Data from {0} - {1}'.format(start_date.year, end_date1.year))
st.write(data.describe())

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,8))
plt.plot(data.Close)
st.pyplot(fig)

d2=data.copy()

d2['open-high'] = d2['Open']-d2['High']
d2['open-low'] = d2['Open'] - d2['Low']
d2['close-high'] = d2['Close']-d2['High']
d2['close-low'] = d2['Close'] - d2['Low']
d2['high-low'] = d2['High'] - d2['Low']
d2['open-close'] = d2['Open'] - d2['Close']
d2=d2.drop(['Open','High','Low','Close','Adj Close'],axis=1)
d2.tail()


opn = data[['Open']] #It denotes the opening stock value
fig2 = plt.figure(figsize=(12,8))
plt.plot(opn)
st.pyplot(fig2)

ds = opn.values

#We are using MinMaxScaler for scaling (Subtracts the min value and divides by the range)
normalizer = MinMaxScaler(feature_range=(0,1))
ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))

#We are defining test and train data sizes
train_size = int(len(ds_scaled)*0.70)
test_size = len(ds_scaled) - train_size

#We are splitting dataset into train and test dataset
ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled)-1,:1]

#we created a function for making a time series dataset for the model
def create_ds(dataset,step):
    Xtrain, Ytrain = [], []
    for i in range(len(dataset)-step-1):
        a = dataset[i:(i+step), 0]
        Xtrain.append(a)
        Ytrain.append(dataset[i + step, 0])
    return np.array(Xtrain), np.array(Ytrain)

#We are taking 100 days price as one record for training
time_stamp = 100
X_train, y_train = create_ds(ds_train,time_stamp)
X_test, y_test = create_ds(ds_test,time_stamp)

#We are reshaping data to fit into LSTM model
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

model = load_model('keras.model.h5')

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

#Inverse transform to get actual value
train_predict = normalizer.inverse_transform(train_predict)
test_predict = normalizer.inverse_transform(test_predict)

#We are going to analyse the last 100 days records
fut_inp = ds_test[ds_test.size-100:]
fut_inp = fut_inp.reshape(1,-1)
tmp_inp = list(fut_inp)

#Creating list of the last 100 data
tmp_inp = tmp_inp[0].tolist()

#Predicting next 30 days price using the current data, It will predict in sliding window manner algorithm with stride 1
lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(tmp_inp)>100):
        fut_inp = np.array(tmp_inp[1:])
        fut_inp=fut_inp.reshape(1,-1)
        fut_inp = fut_inp.reshape((1, n_steps, 1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        tmp_inp = tmp_inp[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp = fut_inp.reshape((1, n_steps,1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1
#print(lst_output)

ds_new = ds_scaled.tolist()

#Entends helps us to fill the missing value with approx value
ds_new.extend(lst_output)
#plt.plot(ds_new[1200:])

#Creating final data for plotting
final_graph = normalizer.inverse_transform(ds_new).tolist()

#Plotting final results with predicted value after 30 Days
st.subheader("{0} prediction of next month open".format(options))
fig3 = plt.figure(figsize=(12,8))
plt.plot(final_graph,)
plt.ylabel("Price")
plt.xlabel("Days")
#plt.title("{0} prediction of next month open".format(options))
plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'NEXT 30D: {0}'.format(round(float(*final_graph[len(final_graph)-1]),2)))
plt.legend()
st.pyplot(fig3)

price = actual_stock['Open'].values[0]
st.write('Actual Stock Price : ',price)

final_prediction = round(float(*final_graph[len(final_graph)-1]),2)
st.write('Predicted Stock Price :',final_prediction)
