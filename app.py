#Use streamlit run Directory/app.py to run the application
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st 
import matplotlib.pyplot as plt

model = load_model('/Users/yashi/Documents/Stock_Market/Stck_price.keras')

st.header = 'Stock Price Predictor'
stock = st.text_input('Enter the Stock Symbol','AAPL')
start = '2015-01-01'
end = '2024-06-25'
data = yf.download(stock,start,end)
st.subheader('Stock Price')

st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

past_100 = data_train.tail(100)
data_test = pd.concat([past_100,data_test],ignore_index= True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Stock Price vs Moving Avg of 50 days')
ma_50 = data.Close.rolling(50).mean()

fig1 = plt.figure(figsize=(14,7))
plt.title('Moving Avg of 50 days')
plt.plot(data.Close, 'g',label='Close Price')
plt.plot(ma_50, 'r',label='MA 50')
plt.legend()
plt.show()
st.pyplot(fig1)

st.subheader('Stock Price vs MA 50 days vs MA 100 days ')
ma_100 = data.Close.rolling(100).mean()

fig2 = plt.figure(figsize=(14,7))
plt.title('Price vs MA_50 vs MA_100')
plt.plot(data.Close, 'g',label='Close Price')
plt.plot(ma_50, 'r',label='MA 50')
plt.plot(ma_100, 'b',label='MA 100')
plt.legend()
plt.show()
st.pyplot(fig2)

st.subheader('Stock Price vs MA 100 days vs MA 200 days')
ma_200 = data.Close.rolling(200).mean()

fig3 = plt.figure(figsize=(14,7))
plt.title('Price vs MA_200 vs MA_100')
plt.plot(data.Close, 'g',label='Close Price')
plt.plot(ma_100, 'r',label='MA 100')
plt.plot(ma_200, 'b',label='MA 200')
plt.legend()
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
x, y = np.array(x), np.array(y)

predict = model.predict(x)
scale = 1/scaler.scale_
predict = predict*scale
y= y*scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)
