import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
df = pd.read_csv("delhi_gurugram_real_estate_dataset.csv")

# Separate features and target
X = df.drop("Price_INR", axis=1)
y = df["Price_INR"]

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Save column names for future use (this is the important new line)
pd.DataFrame(X).to_csv("trained_columns.csv", index=False)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Build the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))  # Output layer

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, validation_data=(X_test_scaled, y_test))

# Save the model
model.save("delhi_gurugram_price_model.keras")
