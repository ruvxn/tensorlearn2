import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the dataset
dataset = pd.read_csv('dataset/SalesData.csv')

# Display the first 5 rows of the dataset
print(dataset.head())

# Plot the dataset in a scatter plot
dataset.plot('Temperature', 'Revenue', kind='scatter')

plt.scatter(dataset['Temperature'], dataset['Revenue'])
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.title('Scatter Plot of Temperature vs Revenue')
plt.show()

# Extract the input features and labels
x_train = dataset['Temperature']
y_train = dataset['Revenue']

# Create a model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

#summary of the model
print(model.summary())

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),loss='mean_squared_error')

# Train the model
epochs_hist = model.fit(x_train, y_train, epochs=100)
epochs_hist.history.keys()

# Display the training results
plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])
plt.show()

# Get the slope of the line
weights = model.get_weights()
slope = weights[0][0]

# Get the intercept of the line
intercept = weights[1][0]

# Print the slope and intercept of the line
print('The slope of the line is: ' + str(slope))
print('The intercept of the line is: ' + str(intercept))

# Make predictions
temp = 40
revenue = model.predict(np.array([[temp]])) 
print('The revenue predicted by the model for a temperature of 40 degrees is: ' + str(revenue))

# Plot the data points and the best-fit line
plt.scatter(x_train, y_train, color='gray')
plt.plot(x_train, model.predict(x_train), color='red')
plt.ylabel('Revenue [dollars]')
plt.xlabel('Temperature [degrees]')
plt.title('Revenue Generated vs. Temperature')
plt.show()

# Save the model
model.save('models/sales_model.h5')
print('Model saved to disk.')


