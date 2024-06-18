#kfold_nny
import numpy as np
import time
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import csv
import os

# Load your dataset here
data=  np.loadtxt('dataset_normalized.csv', delimiter=',', skiprows=1)

        
model_settings = {
    'fold_number': 10,
    'hidden_layers': 3,
    'hidden_units_per_layer': [128 , 128, 128],
    'output_units': 2,
    'activation': 'relu',
    'output_activation': 'linear',
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 256,
    'validation_split': 0.2
}



# K-fold cross-validation
kf = KFold(n_splits=model_settings['fold_number'], shuffle=True, random_state=42)

#metrics for each fold
mse_scores = []
r2_scores = []
average_times = []
fold_mse = []  
fold_r2 = []  
fold_time = []  
loss_per_fold = []
val_loss_per_fold = []

# run model for each fold
for fold, (train_index, test_index) in enumerate(kf.split(data)):
    print(f"Training on Fold {fold + 1}/{model_settings['fold_number']}")

    train_data, test_data = data[train_index], data[test_index]
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    target_columns = slice(0, 2)
    input_columns = slice(2, None)
    X_train = train_data[:, input_columns]
    y_train = train_data[:, target_columns]
    X_test = test_data[:, input_columns]
    y_test = test_data[:, target_columns]

    # Create the model
    model = Sequential()
    model.add(Dense(units=model_settings['hidden_units_per_layer'][0], input_dim=X_train.shape[1],
                    activation=model_settings['activation']))
    for units in model_settings['hidden_units_per_layer'][1:]:
        model.add(Dense(units=units, activation=model_settings['activation']))
    model.add(Dense(units=model_settings['output_units'], activation=model_settings['output_activation']))

    # Compile the model
    custom_optimizer = Adam(learning_rate=model_settings['learning_rate'])
    model.compile(optimizer=custom_optimizer, loss='mean_squared_error')
    model.summary()

    # Train the model
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=model_settings['epochs'],
                        batch_size=model_settings['batch_size'],
                        validation_split=model_settings['validation_split'], verbose=0)
    total_time = time.time() - start_time

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    y_pred = y_pred.astype(np.float32)

    # metrics 
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)
    average_times.append(total_time)
    fold_mse.append(mse)
    fold_r2.append(r2)
    fold_time.append(total_time)

    # Save per epoch losses
    loss_per_fold.append(history.history['loss'])
    val_loss_per_fold.append(history.history['val_loss'])

    print(f"Fold {fold + 1}/{model_settings['fold_number']} - MSE: {mse} - R²: {r2} - Total Time: {total_time} seconds")

# Print the average MSE and R² across folds
average_mse = np.mean(mse_scores)
average_r2 = np.mean(r2_scores)
average_time = np.mean(average_times)
print(f"Average MSE across {model_settings['fold_number']} folds: {average_mse}")
print(f"Average R² across {model_settings['fold_number']} folds: {average_r2}")
print(f"Average Time across {model_settings['fold_number']} folds: {average_time} seconds")

model_settings['average_mse'] = average_mse
model_settings['average_r2'] = average_r2
model_settings['average_time'] = average_time
std_mse = np.std(mse_scores)
std_r2 = np.std(r2_scores)

print(f"Standard Deviation of MSE across {model_settings['fold_number']} folds: {std_mse}")
print(f"Standard Deviation of R² across {model_settings['fold_number']} folds: {std_r2}")

model_settings['std_mse'] = std_mse
model_settings['std_r2'] = std_r2

model_settings['fold_mse'] = fold_mse
model_settings['fold_r2'] = fold_r2 
model_settings['fold_time'] = fold_time

model_settings['loss_per_fold'] = loss_per_fold
model_settings['val_loss_per_fold'] = val_loss_per_fold

# Save to CSV
csv_file_path = 'kf_results.csv'

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
