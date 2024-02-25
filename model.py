import math
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import yfinance as yf
import time
from datetime import datetime, timedelta


plt.style.use('fivethirtyeight')
# Get the stock quote
yf.pdr_override()
current_date = datetime.now().strftime('%Y-%m-%d')
df = pdr.get_data_yahoo('AAPL', start='2020-10-10', end=current_date)
#print(df) #check

#get the number of rows and columns in the data set
#print(df.shape)

'''#visualize the closing price 
plt.figure(figsize=(16,9))
plt.title('Close price history')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close price USD',fontsize=18)
#plt.show()'''

#create a new data frame with close price
data = df.filter(['Close'])
dataset = data.values #numpy array

#get the number of rows to train the model
training_data_len = math.ceil(len(dataset) * .8)
#print (training_data_len)

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset) 
#print(scaled_data)

#Creating the training data 
train_data = scaled_data[:training_data_len, :]
#split the data into x_train and y_train data sets
x_train = []
y_train = [] #target values
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    '''if i<=60:
         print(x_train)
         print(y_train)
         print()'''
    
#convert the x_train and y_train in numpy arrays   
x_train, y=train = np.array(x_train), np.array(y_train)

#reshape the data to 3D for the model (expecting 3D objects)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
#print(x_train.shape)

#Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile the model
model.compile(optimizer='adam',loss='mean_squared_error')

# Convert lists to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

# Then fit the model
model.fit(x_train, y_train, batch_size=1, epochs=1)
#model = joblib.load('apple-stock.joblib')

#Create the testing data set
test_data = scaled_data[training_data_len-60: , :]

x_test=[]
y_test=dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])

#convert the data to a numpy array
x_test = np.array(x_test)
#reshape the data
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

#get the models predicted prices values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Evaluate our model (RMSE - root mean suared error)
rmse = np.sqrt(np.mean((predictions - y_test)**2))
#print(rmse)

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#here

#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close price USD',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train','Val','Predictions'], loc='lower right')
plt.show()

#get the last 60 day closing price 
last_60_days = data[-60:].values
#scale the data to be values between 0 and 1
last_60_days_scaled = scaler.fit_transform(last_60_days)
last_60_days_scaled = np.array(last_60_days_scaled)
#reshape the data
last_60_days_scaled = np.reshape(last_60_days_scaled, (last_60_days_scaled.shape[0], last_60_days_scaled.shape[1], 1))
#get the predicted scaled price
predicted_price = model.predict(last_60_days_scaled)
#undo the scaling
predicted_price = scaler.inverse_transform(predicted_price)

# Create a date range for the next month
from datetime import datetime, timedelta

print(df.index[-1])
last_date = df.index[-1]
next_month_dates = [last_date + timedelta(days=i) for i in range(1, 61)]

# Plot the historical and predicted prices for the next month
plt.figure(figsize=(16,8))
plt.title('Historical and Predicted Close Prices')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close price USD',fontsize=18)
plt.plot(df.index, df['Close'], label='Historical Prices')

plt.plot(next_month_dates, predicted_price, label='Predicted Prices')  # plot predicted_prices instead of predictions

plt.legend(loc='lower right')
plt.show()
