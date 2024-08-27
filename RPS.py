import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Mapping of moves to numbers and vice versa
move_map = {'R': 0, 'P': 1, 'S': 2}
reverse_move_map = {0: 'R', 1: 'P', 2: 'S'}

# Function to create the LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
    return model

# Initialize the model
model = create_lstm_model((5, 1))  # Now input shape matches 5 timesteps and 1 feature per timestep

# Store the opponent's history in a format suitable for the LSTM
opponent_history_encoded = []

# The player function
def player(prev_play, opponent_history=[], model=model):
    # Default move if no previous play
    if prev_play == '':
        return 'R'
    
    # Encode the previous play
    opponent_history.append(prev_play)
    encoded_move = move_map[prev_play]
    opponent_history_encoded.append([encoded_move])

    # If we have enough data, make a prediction
    if len(opponent_history_encoded) >= 5:
        # Prepare the input data by taking the last 5 moves
        input_data = np.array(opponent_history_encoded[-5:])
        input_data = input_data.reshape((1, 5, 1))  # Reshape to match model input
        
        # Predict the next move
        prediction = model.predict(input_data, verbose=0)
        predicted_move = np.argmax(prediction)

        # Optionally, train the model on the new data after the prediction
        if len(opponent_history_encoded) > 5:
            X_train = np.array(opponent_history_encoded[-6:-1]).reshape((1, 5, 1))
            y_train = tf.keras.utils.to_categorical([encoded_move], num_classes=3)
            model.fit(X_train, y_train, epochs=1, verbose=0)

        # Map the prediction to the ideal response
        ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}
        return ideal_response[reverse_move_map[predicted_move]]
    
    # If not enough history, return a random move
    return 'R'
