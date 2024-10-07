"""
Objectif:
- Résumer les topics d'une liste de textes 

Datasets:
voir: https://paperswithcode.com/datasets?task=abstractive-text-summarization
- WikiHow: 230,000 articles and summary pairs
-> https://paperswithcode.com/dataset/wikihow
-> voir les datasets loaders

- CORNELL NEWSROOM: 1.3 papers and summaries: large dataset for training and evaluating summarization systems
-> https://lil.nlp.cornell.edu/newsroom/index.html
The summaries are obtained from search and social metadata between 1998 and 2017
and use a variety of summarization strategies combining extraction and abstraction.

- https://paperswithcode.com/dataset/arxiv-summarization-dataset

- SamSUM:
-> passer par Kaggle pour dowload le dataset
https://www.kaggle.com/datasets/akch2914/samsum-dataset-for-chat-summarization?select=train.json

- EmailSum

Modèles:
- tester plusieurs modèles
- le + performant est BertSum

"""

# classe embedding DL methods
# for topic analysis from text

"""
def decorate_message(fun):

    # Nested function
    def addWelcome(site_name):
        return "Welcome to " + fun(site_name)

    # Decorator returns a function
    return addWelcome

@decorate_message
def site(site_name):
    return site_name;

print site("StackOverflow")

Out[0]: "Welcome to StackOverflow"
"""


"""
Doc où j'ai copié le code ChatGPT: Projet_perso_DS_juil2024.docx
ChatGPT - python RNN model to summarize text based on multiple topics:
----------------------------------------------------------------------
Creating a Recurrent Neural Network (RNN) model to summarize text based on multiple topics is a complex task
that involves several steps, including data preprocessing, model building, and training.
While modern architectures like Transformer-based models (e.g., BERT, GPT-3) are more commonly used for such tasks
due to their superior performance,
I'll provide an example using an RNN-based approach.

VOIR AUSSI LE CODE DU CHALLENGE Advanced RNN - Vivabot 2.0 !!
"""

# Imports
import os
import pandas as pd
import numpy as np
import json

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Bidirectional

from class_encoder import Encoder
from class_decoder import Decoder

from constantes import *

"""
Example of simple input for model training
texts = ["This is a text about topic A.", "Another text about topic B."]
summaries = ["Summary of topic A.", "Summary of topic B."]
"""

# Build texts and summaries from train.json

""" intermediate functions and decorators """
def build_RNN_model():
    pass

def decorate_reader(func):
    def wrapper():
        pass

# pouvoir tester plusieurs modèles avec cette classe
# class topics_analyzer
class TopicsAnalyzer():
    # Example of topic list: ['positive','negative','question','exp_duration','event']
    def __init__(self, model_name:str, truncation:bool, dataset_name:str, topics:list[str], max_text_len:int, max_summary_len:int)->None:
        self.model_name = model_name
        self.truncation = truncation
        self.dataset_name = dataset_name
        self.topics = topics
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len

    ## utiliser des decorators sur les fncs de lecture du dataset !!
    # read csv dataset
    def read_data(self):
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        one_levels_up = os.path.abspath(os.path.join(current_file_directory, '..'))
        relative_path_dataset = one_levels_up + '/Input/SamSUM/train.json/' + self.dataset_name
        self.df = pd.read_csv(relative_path_dataset)
        print(f'self.df.columns: {self.df.columns}')

    # read csv dataset
    def read_data_json(self):
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        one_levels_up = os.path.abspath(os.path.join(current_file_directory, '..'))
        relative_path_dataset = one_levels_up + '/Input/SamSUM/train.json/' + self.dataset_name
        self.df = pd.read_json(relative_path_dataset)
        print(f'self.df.columns: {self.df.columns}')
        ##self.df.columns: Index(['id', 'summary', 'dialogue'], dtype='object')

        ## rename 'dialogue' into 'text'
        self.df.rename(columns={'dialogue': 'text'}, inplace=True)
        
    def apply_dl_model(self):
        pass

    # Tokenization 
    def tokenization(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.df.text + self.df.summary)
    
    # Convert texts to sequences
    def texts_to_sequences(self):
        self.text_sequences = self.tokenizer.texts_to_sequences(self.df.text)
        self.summary_sequences = self.tokenizer.texts_to_sequences(self.df.summary)

    # Pad sequences
    def padding(self):
        max_text_len = self.max_text_len
        max_summary_len = self.max_summary_len
        self.text_sequences = pad_sequences(self.text_sequences, maxlen=self.max_text_len, padding='post')
        self.summary_sequences = pad_sequences(self.summary_sequences, maxlen=self.max_summary_len, padding='post')

    def vocabulary_size(self):
        self.vocab_size = len(self.tokenizer.word_index) + 1

    def build_rnn_model(self):
        # Encoder
        self.encoder = Encoder(self.max_text_len, self.vocab_size)

        # Decoder
        self.decoder = Decoder(self.max_text_len,
                               self.max_summary_len,
                               self.vocab_size,
                               self.encoder.encoder_states,
                               self.encoder.state_h,
                               self.encoder.state_c,
                               self.encoder.encoder_outputs)

        # Model Compile
        self.model = Model([self.encoder.encoder_inputs, self.decoder.decoder_inputs], self.decoder.decoder_outputs)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        self.model.summary()

    def train_model(self):
        # a priori Model can be of any DL type
        # Prepare target data for decoder
        decoder_target_data = np.array(self.summary_sequences).reshape(-1, self.max_summary_len, 1)

        # Train the model
        self.model.fit([self.text_sequences,
                        self.summary_sequences], 
                        decoder_target_data, 
                        batch_size=BATCH_SIZE, 
                        epochs=N_EPOCHS, 
                        validation_split=0.2)
        
        # Inference encoder
        self.encoder.inference_encoder()

        # Inference decoder
        self.decoder.inference_decoder()


    def decode_sequence(self,input_seq: str)->str:
        # Encode the input as state vectors.
        print(f'sentence in decode_sequence: {input_seq}')
        ##self.encoder_outputs, state_h, state_c = self.encoder.encoder_model.predict(input_seq)
        self.encoder_outputs, state_h, state_c = self.encoder.encoder_model.predict(x='sequenceTest')
        ##self.encoder_outputs, state_h, state_c = self.encoder.encoder_model.predict(x=np.array(['sequenceTest']))
        ###self.encoder_outputs, state_h, state_c = self.encoder.encoder_model.predict(np.array([input_seq]))

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.summary_tokenizer.word_index['starttoken']  # Assuming you have a start token

        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder.predict(
                [target_seq] + [self.encoder_outputs, state_h, state_c]
            )

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.summary_tokenizer.index_word[sampled_token_index]

            if sampled_word == 'endtoken' or len(decoded_sentence) > self.max_summary_len:
                stop_condition = True
            else:
                decoded_sentence += ' ' + sampled_word

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            state_h, state_c = h, c

        return decoded_sentence.strip()


    # Apply the model: 
    # Input : Sentence
    # Output: Summary
    # decode to get the summary
    """
    def decode_sequence_01(self,input_seq):
        # Encode the input as state vectors.
        states_value = self.encoder.encoder_lstm.predict(input_seq)
        
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        
        # Populate the first token of target sequence with the start token.
        target_seq[0, 0] = self.tokenizer.word_index['starttoken']
        
        stop_condition = False
        decoded_sentence = ''
        
        while not stop_condition:
            output_tokens, h, c = self.decoder.decoder_lstm.predict([target_seq] + states_value)
            
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.tokenizer.index_word[sampled_token_index]
            
            decoded_sentence += ' ' + sampled_word
            
            # Test d'arrêt 
            if sampled_word == 'endtoken' or len(decoded_sentence.split()) > self.max_summary_len:
                stop_condition = True
            
            # Update the target sequence (length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            
            # Update states
            states_value = [h, c]
        
        return decoded_sentence
    """

    """
    # Example usage
    input_seq = text_sequences[0].reshape(1, max_text_len)
    decoded_sentence = decode_sequence(input_seq)
    """

    """ decorators """
    """ DL Models for topics analysis in a text """
    def decorate_rnn_model(func):
        def wrapper():
            # model RNN
            model = build_RNN_model()
            model.summary()

        
