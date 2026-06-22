
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Define the data directly in the script
data = pd.DataFrame({
    'Year': [1900, 1901, 1902, 1903, 1904, 2020, 2021],  # Example years
    'Temperature': [13.50, 13.55, 13.60, 13.62, 13.67, 15.30, 15.35]  # Corresponding temperatures (in °C)
})

# Check the first few rows of the data
print(data.head())

# Plotting the temperature trend over time
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Temperature', data=data, color='b', label='Global Temperature')
plt.title('Global Temperature Trend Over Time', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True)
plt.show()

# Statistical modeling to see the trend (Linear Regression)
# Adding constant for the intercept
X = sm.add_constant(data['Year'])
y = data['Temperature']

# Fit the model (ignoring the warning by suppressing tests)
model = sm.OLS(y, X).fit()

# Print out the results of the regression
print(model.summary())

# Plot the regression line along with the data
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Temperature', data=data, color='b', label='Observed Temperature')
plt.plot(data['Year'], model.predict(X), color='r', label='Fitted Line (Trend)', linestyle='--')
plt.title('Global Temperature Trend with Linear Regression', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()
