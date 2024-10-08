
---

# Alphabet Soup Charity Deep Learning Classifier

## Overview of the Analysis

The objective of this analysis was to build a binary classification model that predicts whether applicants will be successful if funded by the Alphabet Soup charity. Using a dataset of past applicants, we created a deep learning model to help the charity make informed funding decisions. The model uses various features from the dataset to predict the success of applicants based on historical data.

## Data Preprocessing

### Target Variable
- **IS_SUCCESSFUL**: This is the target variable that indicates whether an organization was successful in using the funds received.

### Features
- **APPLICATION_TYPE**: The type of application submitted to Alphabet Soup.
- **AFFILIATION**: Affiliated sector of industry for the organization.
- **CLASSIFICATION**: Government classification for the organization.
- **USE_CASE**: The purpose for which the funds will be used.
- **ORGANIZATION**: The type of organization applying for funds.
- **STATUS**: The active status of the organization.
- **INCOME_AMT**: The income classification of the organization.
- **SPECIAL_CONSIDERATIONS**: Whether the organization has special considerations.
- **ASK_AMT**: The amount of money requested by the organization.

### Data Cleaning
- We dropped the columns **EIN** and **NAME** as they were not relevant features for the model.
- Categorical variables with a large number of unique values (e.g., `APPLICATION_TYPE`, `CLASSIFICATION`) were grouped based on a threshold of occurrences to reduce the number of distinct categories by combining less frequent values into a new category, **Other**.
- We used `pd.get_dummies()` to one-hot encode categorical variables.

### Splitting and Scaling Data
- We split the dataset into features (`X`) and target (`y`) variables.
- The data was then divided into training and testing datasets using an 80-20 split.
- We used `StandardScaler()` to scale the feature data to standardize the range of values before training the model.

## Model Development

### Baseline Model
The first model served as the baseline for comparison:

```python
nn_baseline = tf.keras.models.Sequential()

# Layers
nn_baseline.add(tf.keras.layers.Dense(units=80, input_dim=X_train_scaled.shape[1], activation='relu'))
nn_baseline.add(tf.keras.layers.Dense(units=30, activation='relu'))
nn_baseline.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile
nn_baseline.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history_baseline = nn_baseline.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate
model_loss_baseline, model_accuracy_baseline = nn_baseline.evaluate(X_test_scaled, y_test, verbose=2)
```

- **Baseline Model Accuracy**: ~72.3%

---

## Model Optimizations

### Attempt 1: Add Another Hidden Layer
This attempt involved adding an additional hidden layer to improve the model's performance.

```python
nn_opt = tf.keras.models.Sequential()

nn_opt.add(tf.keras.layers.Dense(units=100, input_dim=X_train_scaled.shape[1], activation='relu'))
nn_opt.add(tf.keras.layers.Dense(units=50, activation='relu'))
nn_opt.add(tf.keras.layers.Dense(units=25, activation='relu'))
nn_opt.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile
nn_opt.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history_opt = nn_opt.fit(X_train_scaled, y_train, epochs=150, batch_size=32, validation_split=0.2)

# Evaluate
model_loss_opt, model_accuracy_opt = nn_opt.evaluate(X_test_scaled, y_test, verbose=2)
```

- **Optimized Loss (Attempt 1)**: 0.55
- **Optimized Accuracy (Attempt 1)**: 73.9%

### Attempt 2: Change Activation Functions and Increase Epochs
In this attempt, we replaced the **ReLU** activation functions with **tanh** and increased the number of epochs to 200.

```python
nn_opt_2 = tf.keras.models.Sequential()

nn_opt_2.add(tf.keras.layers.Dense(units=100, input_dim=X_train_scaled.shape[1], activation='tanh'))
nn_opt_2.add(tf.keras.layers.Dense(units=50, activation='tanh'))
nn_opt_2.add(tf.keras.layers.Dense(units=25, activation='tanh'))
nn_opt_2.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile
nn_opt_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history_opt_2 = nn_opt_2.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_split=0.2)

# Evaluate
model_loss_opt_2, model_accuracy_opt_2 = nn_opt_2.evaluate(X_test_scaled, y_test, verbose=2)
```

- **Optimized Loss (Attempt 2)**: 0.54
- **Optimized Accuracy (Attempt 2)**: 74.1%

### Attempt 3: Add Dropout Layers to Reduce Overfitting
This attempt introduced dropout layers to reduce overfitting by randomly dropping neurons during training.

```python
nn_opt_3 = tf.keras.models.Sequential()

nn_opt_3.add(tf.keras.layers.Dense(units=100, input_dim=X_train_scaled.shape[1], activation='relu'))
nn_opt_3.add(tf.keras.layers.Dropout(0.2))
nn_opt_3.add(tf.keras.layers.Dense(units=50, activation='relu'))
nn_opt_3.add(tf.keras.layers.Dropout(0.2))
nn_opt_3.add(tf.keras.layers.Dense(units=25, activation='relu'))
nn_opt_3.add(tf.keras.layers.Dropout(0.2))
nn_opt_3.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile
nn_opt_3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history_opt_3 = nn_opt_3.fit(X_train_scaled, y_train, epochs=150, batch_size=32, validation_split=0.2)

# Evaluate
model_loss_opt_3, model_accuracy_opt_3 = nn_opt_3.evaluate(X_test_scaled, y_test, verbose=2)
```

- **Optimized Loss (Attempt 3)**: 0.54
- **Optimized Accuracy (Attempt 3)**: 74.0%

---

## Model Comparison

| Model                         | Loss   | Accuracy  |
|-------------------------------|--------|-----------|
| **Baseline Model**             | 0.57   | 72.3%     |
| **Optimized Model (Attempt 1)**| 0.55   | 73.9%     |
| **Optimized Model (Attempt 2)**| 0.54   | 74.1%     |
| **Optimized Model (Attempt 3)**| 0.54   | 74.0%     |

### Summary
- The **baseline model** achieved an accuracy of 72.3%, which served as the starting point for further optimization.
- **Attempt 1** involved adding more neurons and an additional hidden layer, which improved the accuracy to 73.9%.
- **Attempt 2** utilized **tanh** activation functions and increased the number of epochs, which resulted in the best accuracy at **74.1%**.
- **Attempt 3** added dropout layers to reduce overfitting and achieved an accuracy of 74.0%.

The best-performing model used **tanh activation functions** with **200 epochs**, achieving **74.1% accuracy**. While the model did not exceed the 75% target accuracy, various improvements such as adjusting the number of neurons, adding layers, and modifying activation functions improved performance.

## Recommendations
To further improve the model, you may consider:
- **Tuning the learning rate** or experimenting with other optimizers like **RMSprop** or **Adamax**.
- **Ensembling** models to combine predictions from multiple models for better accuracy.
- **Feature engineering** to identify new patterns in the data or remove noise.

## Files
- `AlphabetSoupCharity.h5`: The baseline model.
- `AlphabetSoupCharity_Optimization.h5`: Optimized model from Attempt 1.
- `AlphabetSoupCharity_Optimization_2.h5`: Optimized model from Attempt 2.
- `AlphabetSoupCharity_Optimization_3.h5`: Optimized model from Attempt 3.

---

