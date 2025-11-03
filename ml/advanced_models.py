# ml/advanced_models.py
import tensorflow as tf
from transformers import Transformer

class LSTMPredictor:
    def __init__(self, sequence_length=60, features=15):
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, 
                               input_shape=(sequence_length, features)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
    
    def prepare_sequences(self, df: pd.DataFrame):
        sequences = []
        targets = []
        
        for i in range(len(df) - self.sequence_length):
            seq = df.iloc[i:i+self.sequence_length][self.feature_columns].values
            target = df.iloc[i+self.sequence_length]['close']
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    