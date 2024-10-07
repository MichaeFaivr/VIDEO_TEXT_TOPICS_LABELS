from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Bidirectional, Concatenate, Attention, AdditiveAttention

from constantes import *

class Decoder():
    def __init__(self, max_text_len, max_summary_len, vocab_size, encoder_states, state_h, state_c, encoder_outputs):
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len
        self.vocab_size = vocab_size
        self.encoder_states = encoder_states
        self.state_h = state_h
        self.state_c = state_c
        self.encoder_outputs = encoder_outputs
        self.build_decoder()

    # Retrouver à quoi correspond 128 dans l'architecture !
    # voir le code de vivabot2.0 de Advanced RNN
    def build_decoder(self):
        # Input layer
        self.decoder_inputs = Input(shape=(self.max_summary_len,))

        # Embedding layer
        self.decoder_embedding = Embedding(self.vocab_size, N_UNITS)(self.decoder_inputs)

        # LSTM
        self.decoder_lstm = LSTM(N_UNITS, activation='relu', return_sequences=True, return_state=True)
        self.decoder_outputs, _, _ = self.decoder_lstm(self.decoder_embedding,
                                                       initial_state=self.encoder_states)

        # Attention layer
        ###self.attention = AdditiveAttention()
        ###self.context_vector, self.attention_weights = self.attention([self.decoder_outputs, self.encoder_outputs], return_attention_scores=True)
        ###self.context_vector, self.attention_weights = self.attention([self.encoder_outputs, self.decoder_outputs])

        # Concatenate context vector and decoder LSTM output
        ###self.decoder_concat_input = Concatenate(axis=-1)([self.decoder_outputs, self.context_vector])

        # Dense layer
        self.decoder_dense = TimeDistributed(Dense(self.vocab_size, activation='softmax'))

        # Output layer
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)

    """
    def build_decoder_01(self):
        # Decoder
        self.decoder_inputs = Input(shape=(None,))
        self.dec_emb_layer = Embedding(self.vocab_size, 100, trainable=True)
        self.dec_emb = self.dec_emb_layer(self.decoder_inputs)

        self.decoder_lstm = LSTM(100, return_sequences=True, return_state=True)
        self.decoder_outputs, _, _ = self.decoder_lstm(self.dec_emb, initial_state=[self.state_h, self.state_c])

        # Attention layer
        self.attention = Attention()
        self.context_vector, self.attention_weights = self.attention([self.decoder_outputs, self.encoder_outputs], return_attention_scores=True)

        # Concatenate context vector and decoder LSTM output
        self.decoder_concat_input = Concatenate(axis=-1)([self.decoder_outputs, self.context_vector])

        # Dense layer
        self.decoder_dense = TimeDistributed(Dense(self.vocab_size, activation='softmax'))
        self.decoder_outputs = self.decoder_dense(self.decoder_concat_input)
    """

    # building the inference setup, that will allow to build answers from questions.
    #### CORRIGER LE BUG DANS LE DECODER: 
    # ValueError: Input 0 of layer "lstm_1" is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: (None, 20) 
    def inference_decoder(self):
        # Decoder inference model
        decoder_state_input_h = Input(shape=(LATENT_DIM,))
        decoder_state_input_c = Input(shape=(LATENT_DIM,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        ## original: Bug
        #decoder_outputs, state_h, state_c = self.decoder_lstm(self.decoder_inputs,
        #                                                      initial_state=decoder_states_inputs)
        # Test: ça passe avec decoder_embedding: mais je dois voir cela en détails
        decoder_outputs, state_h, state_c = self.decoder_lstm(self.decoder_embedding,
                                                              initial_state=decoder_states_inputs)

        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [self.decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

    """
    def inference_decoder_01(self):
        # Decoder inference model
        self.decoder_state_input_h = Input(shape=(100,))
        self.decoder_state_input_c = Input(shape=(100,))
        self.decoder_hidden_state_input = Input(shape=(self.max_text_len, 100))

        self.dec_emb2 = self.decoder_embedding #self.dec_emb_layer(self.decoder_inputs)
        self.decoder_outputs2, state_h2, state_c2 = self.decoder_lstm(
            self.dec_emb2, initial_state=[self.decoder_state_input_h, self.decoder_state_input_c]
        )

        self.context_vector, self.attention_weights = self.attention([self.decoder_hidden_state_input, self.decoder_outputs2])
        self.decoder_concat_input2 = Concatenate(axis=-1)([self.decoder_outputs2, self.context_vector])
        self.decoder_outputs2 = self.decoder_dense(self.decoder_concat_input2)

        self.decoder_model = Model(
            [self.decoder_inputs] + [self.decoder_hidden_state_input, self.decoder_state_input_h, self.decoder_state_input_c],
            [self.decoder_outputs2] + [state_h2, state_c2]
        )
    """