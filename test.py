import yfinance  as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#user input
ticker=input("Enter stock name (eg RELIANCE.NS)\nuse suffix like .NS or .BO: ")
start_date = input("Enter start date (YYYY-MM-DD): ")
end_date = input("Enter end date (YYYY-MM-DD): ")

#loading data...
data = yf.download(ticker, start=start_date, end=end_date)
data.reset_index(inplace=True)

#prepping data..
data['Days']=(data['Date']-data['Date'].min()).dt.days
X=data[['Days']]
y=data['Close']

#split dataa..
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#Train model..
model = LinearRegression()
model.fit(X_train, y_train)

#Predict..
y_pred = model.predict(X_test)

#Plot..
plt.figure(figsize=(10,5))
plt.plot(data['Date'], data['Close'], label='Actual Price')
plt.plot(X_test.index.map(lambda i: data['Date'][i]), y_pred, label='Predicted Price', color='red')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()