from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Bidirectional

from constantes import *

class Encoder():
    def __init__(self, max_text_len, vocab_size):
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        self.build_encoder()
        
    def build_encoder(self):
        self.encoder_inputs = Input(shape=(self.max_text_len,))
        self.encoder_embedding = Embedding(self.vocab_size, N_UNITS)(self.encoder_inputs)
        self.encoder_lstm = LSTM(N_UNITS, return_state=True)
        self.encoder_outputs, self.state_h, self.state_c = self.encoder_lstm(self.encoder_embedding)
        self.encoder_states = [self.state_h, self.state_c]
    
    def inference_encoder(self):
        # To generate summaries, you need to create an inference model for both the encoder and decoder.
        # Encoder inference model
        self.encoder_model = Model(inputs=self.encoder_inputs, outputs=[self.encoder_outputs, self.state_h, self.state_c])