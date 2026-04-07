import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import numpy as np

# Step 1: Load Data
df = pd.read_csv("sales.csv")

# Step 2: Clean Data
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

print("Data Loaded:\n", df.head())

# Step 3: Plot Original Data
plt.plot(df['Date'], df['Sales'])
plt.title("Original Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.show()

# Step 4: Prepare for Prophet
df_prophet = df[['Date','Sales']]
df_prophet.columns = ['ds','y']

# Step 5: Train Model
model = Prophet()
model.fit(df_prophet)

# Step 6: Future Prediction
future = model.make_future_dataframe(periods=10)
forecast = model.predict(future)

# Step 7: Show Prediction
print("\nForecast:\n", forecast[['ds','yhat']].tail())

# Step 8: Plot Forecast
model.plot(forecast)
plt.show()

# Step 9: Evaluate
predicted = forecast['yhat'][:len(df)]
actual = df['Sales']

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(((actual - predicted)**2).mean())

print("\nMAE:", mae)
print("RMSE:", rmse)

forecast[['ds','yhat']].to_csv("forecast_output.csv", index=False)
print("Forecast file saved!")