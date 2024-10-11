# # BHARAT INTERN
# 
# # NAME-SHAIK TABASSUM SHABANA
# 
# # TASK1-STOCK PREDICTION 
# In this we will use the Nestle India -Historical Stock Price Dataset for STOCK PREDICTION

# # Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense 

# # Loading data from a CSV file

df=pd.read_csv(r"C:\Users\dell\Downloads\nestle.csv")
df.head()
df.tail()

# # SHAPE OF THE DATA
df.shape

# # Gathering information about data
df.info()
df.describe()
df.dtypes


# # cleaning the data
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date', ascending=True)
df = df[['Date', 'Close Price']]

df.columns
df.head()
df.tail()


# # Normalize the Close prices
scaler = MinMaxScaler()
df['Close Price'] = scaler.fit_transform(df['Close Price'].values.reshape(-1, 1))


# # split the data into train and test sets
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]
print(train_data)
print(test_data)


# # create sequences and labels for training and testing
# Function to create sequences and labels
def create_sequences(df, seq_length):
    sequences, labels = [], []
    for i in range(len(df) - seq_length):
        seq = df['Close Price'].values[i:i+seq_length]
        label = df['Close Price'].values[i+seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Define sequence length
seq_length = 10 
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)
print(X_train,y_train,X_test,y_test)


# # Reshape the data for LSTM
X_train = X_train.reshape(X_train.shape[0], seq_length, 1)
X_test = X_test.reshape(X_test.shape[0], seq_length, 1)
print(X_train.shape)
print(X_test.shape)


# # Build and train the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)


# # Make predictions
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))


# # Calculating RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")


# # Plot the true vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='True Price', color='green')
plt.plot(y_pred, label='Predicted Price', color='red')
plt.legend()
plt.show()
