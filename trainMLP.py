import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
import csv
import os
from sklearn.model_selection import train_test_split
import joblib

full_dataset=  np.loadtxt('dataset_normalized.csv', delimiter=',', skiprows=1)

model_settings = {
    'hidden_layers': 3,
    'hidden_units_per_layer': [128 , 64, 32],
    'output_units': 2,
    'activation': 'relu',
    'output_activation': 'linear',
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 256,
    'validation_split': 0.2
}

loss_per_epochs = []
val_loss_per_epochs = []
train_data, test_data = train_test_split(full_dataset, test_size=0.2, random_state=42)

target_columns = slice(0, 2)
input_columns = slice(2, None)

X_train = train_data[:, input_columns]
y_train = train_data[:, target_columns]

X_test = test_data[:, input_columns]
y_test = test_data[:, target_columns]

model = Sequential()
model.add(Dense(units=model_settings['hidden_units_per_layer'][0], input_dim=X_train.shape[1],
                    activation=model_settings['activation']))
for units in model_settings['hidden_units_per_layer'][1:]:
        model.add(Dense(units=units, activation=model_settings['activation']))
model.add(Dense(units=model_settings['output_units'], activation=model_settings['output_activation']))

custom_optimizer = Adam(learning_rate=model_settings['learning_rate'])
model.compile(optimizer=custom_optimizer, loss='mean_squared_error')
model.summary()

start_time = time.time()

history = model.fit(X_train, y_train, epochs=model_settings['epochs'],
                        batch_size=model_settings['batch_size'],
                        validation_split=model_settings['validation_split'], verbose=0)
total_time = time.time() - start_time

model_filename = 'mpl_model.pkl'
joblib.dump(model, model_filename)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

loss_per_epochs.append(history.history['loss'])
val_loss_per_epochs.append(history.history['val_loss'])

print(f"MSE: {mse} - R²: {r2} - Total Time: {total_time} seconds")
model_settings['MSE_loss'] = mse
model_settings['r2'] = r2
model_settings['time'] = total_time
model_settings['loss_history'] = loss_per_epochs
model_settings['val_loss'] = val_loss_per_epochs


# Save to CSV or JSON
csv_file_path = 'Model_results.csv'

# Check if the file exists
if os.path.exists(csv_file_path):
    # If it exists, append data
    with open(csv_file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=model_settings.keys())
        writer.writerow(model_settings)
else:
    # If it doesn't exist, create a new file
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=model_settings.keys())
        writer.writeheader()
        writer.writerow(model_settings)
