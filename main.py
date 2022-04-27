#from pyexpat import model
import datetime
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

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title('Stock Prediction')

st.sidebar.title("Rakesh Jhunjhunwala's Portfolio of companies")
link = 'https://www.moneycontrol.com/india-investors-portfolio/rakesh-jhunjhunwala-and-associates'
st.sidebar.write('Click here to view the portfolio :')
st.sidebar.markdown(link, unsafe_allow_html=True)

st.sidebar.write('')

options = st.sidebar.selectbox(
     'Select the stock : ',
     ['JUBLPHARMA.NS', 'CANBK.NS', 'IBULHSGFIN.BO', 
      'ANANTRAJ.NS', 'ATFL.BO', 'AUTOIND.BO',
      'DBREALTY.BO', 'EDELWEISS.NS', 'FEDERALBNK.BO',
      'FORTIS.BO',  'MANINFRA.BO', 'NATIONALUM.NS',
      'NCC.NS', 'ORIENTCEM.NS', 'RALLIS.NS',
      'TATACOMM.BO', 'TATAMOTORS.BO', 'WABAG.NS',
      'DCAL.NS', 'NAZARA.NS', 'JUBLINGREA.NS',
      'STARHEALTH.NS', 'METROBRAND.NS', 'CRISIL.NS',
      'DELTACORP.NS', 'INDHOTEL.NS', 'TITAN.NS',
      'APTECHT.NS', 'WOCKPHARMA.NS', 'TV18BRDCST.NS',
      'KARURVYSYA.BO' ])

st.write('You have selected : ',options)

end_date1 = st.sidebar.date_input(
     'Select the date for prediction : '
     )

end_date = end_date1-dt.timedelta(days=43)
start_date = end_date-dt.timedelta(days=365*5)

#actual_stock = df.DataReader(options, 'yahoo', end_date+dt.timedelta(days=42), end_date1)

end_date1 = dt.datetime.strptime(str(end_date1), "%Y-%m-%d").date()
end_date = dt.datetime.strptime(str(end_date), "%Y-%m-%d").date()
start_date = dt.datetime.strptime(str(start_date), "%Y-%m-%d").date()

data = df.DataReader(options, 'yahoo', start_date, end_date)

st.sidebar.write('')

if st.sidebar.button('Data of {0}'.format(options)):
    st.subheader('Data from {0} - {1}'.format(start_date.year, end_date1.year))
    st.write(data.describe())

if st.sidebar.button('Closing Price vs Time Chart'):
    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize=(12,8))
    plt.plot(data.Close)
    st.pyplot(fig)

opn = data[['Open']] #It denotes the opening stock value

if st.sidebar.button('Opening Stock Prices'):
    st.subheader('Opening price through years')
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

#Splitting data between train and test
ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]

#creating dataset in time series for LSTM model 
#X[100,120,140,160,180] : Y[200]
def create_ds(dataset,step):
    Xtrain, Ytrain = [], []
    for i in range(len(dataset)-step-1):
        a = dataset[i:(i+step), 0]
        Xtrain.append(a)
        Ytrain.append(dataset[i + step, 0])
    return np.array(Xtrain), np.array(Ytrain)

#Taking 100 days price as one record for training
time_stamp = 30
X_train, y_train = create_ds(ds_train,time_stamp)
X_test, y_test = create_ds(ds_test,time_stamp)

#Reshaping data to fit into LSTM model
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

model = load_model('keras.model.h5')

#Predicitng on train and test data
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

#Inverse transform to get actual value
train_predict = normalizer.inverse_transform(train_predict)
test_predict = normalizer.inverse_transform(test_predict)

look_back=30
trainPredictPlot = np.empty_like(ds_scaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(ds_scaled)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(ds_scaled)-1,:] = test_predict

test = np.vstack((train_predict,test_predict))

#Getting the last 100 days records
n=len(ds_test)-30
print(n)
fut_inp = ds_test[n:]
len(fut_inp)

fut_inp = fut_inp.reshape(1,-1)
tmp_inp = list(fut_inp)

#Creating list of the last 100 data
tmp_inp = tmp_inp[0].tolist()

#Predicting next 30 days price using the current data
#It will predict in sliding window manner (algorithm) with stride 1
lst_output=[]
n_steps=30
i=0
while(i<30):
    
    if(len(tmp_inp)>30):
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

#Creating a dummy plane to plot graph one after another
plot_new=np.arange(1,101)
plot_pred=np.arange(101,131)

ds_new = ds_scaled.tolist()

#Creating final data for plotting
final_graph = normalizer.inverse_transform(ds_new).tolist()

#Plotting final results with predicted value after 30 Days
st.subheader("{0} prediction on {1}".format(options, end_date1))
fig3 = plt.figure(figsize=(12,8))
plt.plot(final_graph,)
plt.ylabel("Price")
plt.xlabel("Days from {0}".format(start_date.year, end_date1.year))
#plt.title("{0} prediction of next month open".format(options))
plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'On {0} : {1}'.format(end_date1,round(float(*final_graph[len(final_graph)-1]),2)))
plt.legend()
st.pyplot(fig3)

#price = actual_stock['Open'].values[0]
#st.write('Actual Stock Price : ',price)

final_prediction = round(float(*final_graph[len(final_graph)-1]),2)
st.write('Predicted Stock Price :',final_prediction)
