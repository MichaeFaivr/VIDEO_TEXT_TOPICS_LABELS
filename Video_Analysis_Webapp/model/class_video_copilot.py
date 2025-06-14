# create a class to store video attributes and methods for object detection

# In the class VideoBaselineClass, we will define the attributes and methods for object detection from a predefined list of object types.
# The class will have the following attributes:
# - frame_size: a tuple
# - duration: the duration of the video, trimmed at 10 mins to avoid too heavy files
# - video_type: the type of video
# The class will have the following methods:
# - landmarking: method to perform landmarking
# - start_obj_detection: method to start object detection
# - end_obj_detection: method to end object detection
# - nb_objects: method to count the number of objects of a specific type in the sequence
# - main_color: method to get the main color of the object if only one of this type is present in the sequence

# The class will take a video as input and output a JSON file with the start/end sequence, list of object types, and list of box coordinates.

# Example of sequence with multiple object types:
# - Object type: car
# - Object type: person
# - Object type: bicycle
# - Object
# - Start: 00:00:10
# - End: 00:00:20
# - Box coordinates: [x1, y1, x2, y2]
# - Object type: person
# - Start: 00:00:30
# - End: 00:00:40

# voir le landmarking d'objets dans une vidéo ou dans une image pour identifier les objets.

# bien concevoir les classes:
# - classe de base: VideoBaselineClass
# - classe fille #1: VideoToSpeechClass
# - classe fille #2: VideoToObjectsClass

# ATTENTION: Responsibilité Dev / DS : portabilité du package avec les paths flexibles / best practices

"""
 ____________________________
| 1. Video to Text  |  DONE  |
|___________________|________|
| 2. Add punctuation|  TODO  |
|___________________|________|
| 3. Summary        |  DONE  |
|___________________|________|
| 4. NER            |  DONE  |
|___________________|________|
| 5. Sentiment      |  DONE  |
|___________________|________|
| 6. Topic Modeling |  DONE  |
|___________________|________|
"""

import numpy as np
import pandas as pd
import random

# Image analysis
import cv2
import torch
from torchvision import transforms
from PIL import Image
import easyocr
import pytesseract
from ultralytics import YOLO

# Text analysis
import speech_recognition as sr
import json
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from textblob import TextBlob
from datetime import datetime
import os

# Post Validation of video based on text comparison with generative AI text on the same product
from difflib import SequenceMatcher 

## ATTENTION aux paths !!!!
## NO USE OF GOOGLE SERVICES API FOR NOW
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "" # path to the Google API credentials to add
from google.cloud import speech
from pydub import AudioSegment
# Vosk Method
from vosk import Model, KaldiRecognizer 
import wave

###from distutils import msvccompiler
###from punctuator import Punctuator
###from deepmultilingualpunctuation import PunctuationModel ## implies transformers

import inspect

from commons.json_files_management import *
from commons.decorators import *
from commons.common_functions import *
from model.constants import *

# The reason gensim no longer has a summarization module is that it was 
# deprecated and eventually removed in more recent versions of the library.

# CTRL+I -> chat prompt:
# Prompt: write the code of the function extract_speech(video_path) that extracts the text of what is said in a video

# 04Nov. Punctuation is critical in the text retrieved from the audio of the video.
# Import of deepmultilingualpunctuation is not working.
# This point has to be fixed.

# APPROACH: implement several methods for speech recognition and punctuation:
# 1. Use the Google API for speech recognition and punctuation
# 2. Use the nlp library for speech recognition and punctuation
# 3. Find another library for speech recognition and punctuation

## ATTENTION: check with the senior Dev how to handle the audio file
###TEMP_AUDIO_FILE = "temp_audio.wav"
###TEMP_MONO_AUDIO_FILE = "temp_mono_audio.wav"

# ATTENTION:
# use a config file to store the paths
# and the credentials for the Google API : Json key

# in imported constants
#TEXT_DETECTION_CONFIDENCE_THRESHOLD = 0.5
#DEFAULT_SUMMARY_TEXT = "No text to summarize"

"""
def add_punctuation(text):
    p = Punctuator('Demo-Europarl-EN.pcl')
    punctuated_text = p.punctuate(text)
    return punctuated_text
"""
"""
def add_punctuation(text):
    # Load the punctuation model
    try:
        model = PunctuationModel()
    except OSError:
        from deepmultilingualpunctuation import download
        download('deepmultilingualpunctuation')
        model = PunctuationModel()

    # Add punctuation to the text
    punctuated_text = model.restore_punctuation(text)
    return punctuated_text
"""

"""
def add_punctuation(text):
    # NOT WORKING and lowercase the entities
    # Add punctuation using spaCy: under assumption of English language
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        from spacy.cli import download
        download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    punctuated_text = ''.join([token.text_with_ws for token in doc])
    return punctuated_text
"""

"""
def add_punctuation(text):
    # NOT WORKING and lowercase the entities    
    # Load the SpaCy model
    nlp = spacy.load("en_core_web_sm")

    # Process the text
    doc = nlp(text)

    # Add punctuation based on the model's predictions
    punctuated_text = []
    for sent in doc.sents:
        punctuated_text.append(sent.text.capitalize() + "... ")

    # Join the sentences with spaces
    return " ".join(punctuated_text)
"""

### TODO : usual tool funuctions like below to be declared in toolbox/ directory
"""
def read_json_file(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data
"""

# Use the commons/json_files_managements.py save_json_file

def convert_to_mono(input_audio_file: str, output_audio_file: str) -> str:
    """
    Convert stereo audio to mono using pydub.
    Args:
        input_audio_file (str): Path to the input audio file.
        output_audio_file (str): Path to save the converted mono audio file.
    Returns:
        str: Path to the converted mono audio file.
    """
    # Load the audio file
    audio = AudioSegment.from_file(input_audio_file)
    # Convert to mono
    mono_audio = audio.set_channels(1)
    # Save the mono audio to a temporary file
    mono_audio_path = output_audio_file
    mono_audio.export(mono_audio_path, format="wav")
    return mono_audio_path

def transcribe_file_with_auto_punctuation(audio_file: str) -> speech.RecognizeResponse:
    """
    transcribe audio file with automatic punctuation using Google Cloud Speech-to-Text API.
    Args:
        audio_file (str): Path to the audio file.
    Returns:    
        speech.RecognizeResponse: Transcription response from Google Cloud Speech-to-Text API.
    """
    # parameter audio_file is expected to be a mono audio file
    client = speech.SpeechClient()

    # Load the audio file
    with open(audio_file, "rb") as f:
        content = f.read()

    # Configure the audio file and recognition settings
    # address long audio files:
    # google.api_core.exceptions.InvalidArgument:
    # 400 Sync input too long. For audio longer than 1 min use LongRunningRecognize with a 'uri' parameter.
    # trouver un Youtube ou ask AI
    ###audio = speech.LongRunningRecognizeResult()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        ##sample_rate_hertz=6000,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )

    # Perform the transcription
    response = client.recognize(config=config, audio=audio)
    return response


def format_price_strings(text_video: str) -> str:
    """
    Get the currency of the speech of the video.
    1. Find all prices strings in the text
    2. Modify the price strings as follow: '200 dollars' -> '$200' ; '200 euros' -> '€200'
    """
    # ATTENTION: échec dans le cas de strings de prix avec des ponctuations : 1200 euros. -> word = 'euros.': échec
    # !!!! CODE ci-dessous à corriger !!!!!
    dict_currency = {'dollars': '$', 'euros': '€', 'pounds': '£', 'yen': '¥', 'francs': '₣'}
    if not text_video or text_video == "":
        return ""

    # "The price of the item is 200 euros" -> "The price of the item is €200"
    modified_text = ""
    word_count = 1
    list_words = text_video.split()
    print(f"list_words: {list_words}")
    # Add condition of the length of the list_words
    if len(list_words) < 3:
        print("format_price_strings - list_words is too short, returning original text")
        return text_video

    currency = 'unknown'
    for word in list_words:
        if word in dict_currency.keys() or word in dict_currency.values():
            print(f"word in dict_currency: {word}")
            if word in dict_currency.keys():
                currency = word
                symbol = dict_currency[word]
            else:
                currency = [key for key, val in dict_currency.items() if val == word][0]
                symbol = word
            print(f"format_price_strings - word: {word}")
            # remove the preceding string
            print(f"symbol: {symbol}, list_words[word_count-1]: {list_words[word_count-1]}, list_words[word_count-2]: {list_words[word_count-2]}")
            if word_count >= 2:
                # concatenate the symbol and the previous word and remove the preceding string
                modified_text = " ".join(modified_text.split()[:-1]) + " " + symbol + list_words[word_count-2] + " "
            else:
                modified_text += word + " "
        else:
            modified_text += word + " "
        word_count += 1
            ##currency = word # see to take the most frequent currency
    text_video = modified_text
    return text_video


# BASE CLASS: empty methods to fill in child classes
class VideoBaselineClass:
    def __init__(self, video_path, frame_size, duration, video_type, random_time):
        self.frame_size = frame_size
        self.duration = min(duration, 600) if duration else None  # Trimmed at 10 mins
        self.video_type = video_type
        self.random_time = random_time
        self.video_path = video_path

    def loadVideo(self):
        """
        Load the video file using moviepy.
        Returns:
            video: Loaded video file from the video file path.
        """
        import moviepy.editor as mp
        # Load the video file
        video = mp.VideoFileClip(self.video_path)
        return video

    def landmarking(self):
        pass

    def start_obj_detection(self):
        pass

    def end_obj_detection(self):
        pass

    def nb_objects(self, object_type):
        pass

    def main_color(self, object_type):
        pass

    def logger_process_times(self, time_log_file):
        # Log the process times
        pass


# CHILD CLASS N°1: VideoToSpeechClass
class VideoToSpeechClass(VideoBaselineClass):
    def __init__(self, video_path, frame_size=None, duration=None, video_type=None, random_time=None, output_text=None):
        super().__init__(video_path, frame_size, duration, video_type, random_time)
        self.output_text = output_text

    def extract_audio(self, output_audio_file:str)-> None:
        """
        Extract the audio from the video file and save it to a temporary file.
        Args:
            output_audio_file (str): Path to save the extracted audio file.
        Returns:
            None    
        """
        # simulate speech extraction
        import moviepy.editor as mp

        # Load the video file
        video = mp.VideoFileClip(self.video_path)

        # Extract the audio from the video
        audio = video.audio
        # output the audio to a temporary file
        audio_path = output_audio_file
        audio.write_audiofile(audio_path) # Write the audio track to the output file
        self.audio_path = audio_path

        # Initialize the recognizer
        recognizer = sr.Recognizer()

        # Load the audio file
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            self.audio_data = audio_data


    def extract_speech_google_recognizer(self, temp_audio_file:str)->str:
        """
        Extract the speech from the video using Google Speech Recognition API.
        Args:
            temp_audio_file (str): Path to the temporary audio file.
        Returns:
            Transcribed text from the audio.
        """
        # Extract the audio from the video
        with measure_time(inspect.currentframe().f_code.co_name):
            self.extract_audio(temp_audio_file)
            # Initialize the recognizer
            recognizer = sr.Recognizer()
            try:
                # Transcribe audio to text
                text_video = recognizer.recognize_google(self.audio_data)
                self.output_text = text_video
                ##punctuated_text = add_punctuation(text_video)
                ##text_video = punctuated_text
                ### TEST
                """
                self.output_text = "I am looking for a new laptop for a price range of €1000 to €1200. " \
                "The brands I like are : Asus, Microsoft, Lenovo." \
                "I need a screen size of at least 17 inches." \
                "The RAM should be at least 16 GB. " \
                "The processor should be at least i7. " \
                "The storage should be at least 512 GB SSD. " \
                "I will use the laptop mostly for work as a developer, so for software coding. " \
                "Yet, I will also use it for gaming and digital art creation. " \
                "The price should be between €1000 and €1200. "
                """
                return text_video
            except sr.UnknownValueError:
                return "Speech recognition could not understand the audio"
            except sr.RequestError as e:
                return f"Could not request results from Google Web Speech API; {e}"

       
    def extract_speech_with_vosk(self, temp_audio_file:str)->str:
        """
        Extract the speech from the video using Vosk speech recognition.
        Args:
            temp_audio_file (str): Path to the temporary audio file to store the audio track.
            NB: the audio track has to be mono PCM WAV format for Vosk
            NB2: Github: Failing to load vosk model #1739
                 You need like 10GB of memory to load RNN model. You can remove rnnlm folder from the model, it will take less memory.
        Returns:
            Transcribed text from the audio with a Vosk lib.
        """
        # Extract the audio from the video
        PATH_VOSK_MODEL = "language_models/vosk-model-en-us-0.22" # fix the issue with missing file error
        # Result is not good with vosk-model-en-us-0.22 without rnnlm
        ###PATH_VOSK_MODEL = os.path.join(PATH_VOSK_MODEL, 'final.mdl')
        ##PATH_VOSK_MODEL = "language_models/vosk-model-small-en-us-0.15" # bad
        with measure_time(inspect.currentframe().f_code.co_name):
            ###self.extract_audio(temp_audio_file)
            try:
                # Load the Vosk model
                model = Model(PATH_VOSK_MODEL)  # Path to the Vosk model directory

                # Check if other solution
                # Extract the audio from the video
                self.extract_audio(temp_audio_file)
                # Convert the audio to mono
                temp_audio_file = convert_to_mono(self.audio_path, temp_audio_file)

                # Open the audio file
                print(f"temp_audio_file: {temp_audio_file}")
                # CRITICAL: the audio file must be mono PCM WAV format !!
                with wave.open(temp_audio_file, "rb") as wf:
                    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                        raise ValueError("Audio file must be WAV format mono PCM.")
                    
                    recognizer = KaldiRecognizer(model, wf.getframerate())
                    recognizer.SetWords(True)

                    text_video = ""
                    print(f'extract_speech_with_vosk before while loop')
                    while True:
                        data = wf.readframes(4000)
                        print(f"len(data): {len(data)}")
                        if len(data) == 0:
                            break
                        if recognizer.AcceptWaveform(data):
                            result = recognizer.Result()
                            text_video += json.loads(result).get("text", "") + " "

                    # Finalize the transcription
                    final_result = recognizer.FinalResult()
                    text_video += json.loads(final_result).get("text", "")

                    print(f"Vosk transcription: {text_video}")

                    self.output_text = text_video
                    return text_video
            except Exception as e:
                return f"Error during speech recognition: {e}"
            

    def transcribe_audio_with_punctuation_google_speech_api(self, output_audio_mono_path:str)->str:
        """
        Add punctuation to the transcribed text using Google Cloud Speech-to-Text API.
        Args:
            output_audio_mono_path (str): Path to the mono audio file in output.
        Returns:
            Transcribed text with punctuation.
        """
        # convert to mono track
        mono_audio_file = convert_to_mono(self.audio_path, output_audio_mono_path)
        response = transcribe_file_with_auto_punctuation(mono_audio_file)
        print(f"response: {response}")
        return response
    
    def process_text(self)->str:
        """
        Apply various text processing steps to the transcribed text.
        1. Price formatting
        2. Replace specific words in the text 
        """
        # Price formatting
        textVideo = format_price_strings(self.output_text)
        # Replace words
        for word in REPLACE_WORDS.keys():
            textVideo = textVideo.replace(word, REPLACE_WORDS[word])
        # Update the output text attribute
        self.output_text = textVideo
        return textVideo
    
    def create_output_path_text(self, video_path_uploaded:str)->None:
        """
        Create the output path for the text file.
        Args:
            video_path_uploaded (str): Path to the uploaded video file.
        Returns:
            None
        """
        # Create output directory based on current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        output_dir = os.path.join('Output', current_date, 'extracted_text')
        os.makedirs(output_dir, exist_ok=True)
        # Create a text filename based on the name from video_path and current time
        current_time = datetime.now().strftime('%H-%M-%S')
        video_filename = os.path.basename(video_path_uploaded)
        text_filename = os.path.splitext(video_filename)[0] + '_' + current_time + '_text.txt'
        self.output_path = os.path.join(output_dir, text_filename)

    def save_speech_from_video(self, video_path_uploaded:str, text_video:str)->None:
        """
        Save the transcribed text to a file.
        Args:
            video_path_uploaded (str): Path to the uploaded video file.
            text_video (str): Transcribed text from the video.
        Returns:
            None
        """
        self.create_output_path_text(video_path_uploaded)
        max_size_bytes = 1024*1000  # 1 MB limit
        # Write the text to a file with a size limit
        if text_video and text_video != "":
            with open(self.output_path, 'w') as file:
                file.write(text_video[:max_size_bytes])   

"""
# CHILD CLASS N°2: analyze the text extracted from the video and split it into topics:
# for each topic of interest, get the list of sentences related to that topic
# and summarize the text, extract keywords, and determine the sentiment
# 1. Text from Video Summary   : DONE
# 2. Name Entity Recognition   : DONE
# 3. Key Infos Extraction      : DONE
# 4. Sentiment Analysis        : DONE         
# 5. Topic Detection/Modeling
# 6. Json sentences per topic  : DONE
"""
class VideoTopicsSummaryClass():
    def __init__(self, text, topics, output_json):
        self.text = text
        self.summary = ""
        self.sentences = []
        self.summary_sentences = []
        self.default_text = DEFAULT_SUMMARY_TEXT
        self.topics = topics
        self.output_json = output_json
        nltk.download('punkt')
        nltk.download('punkt_tab')


    # ===========================
    # TEXT SUMMARY
    # ===========================
    def from_text_to_sentences_and_summary(self):
        """
        Summarize the text using spaCy lib
          1. get most relevant sentences of the text
          2. summarize each selected sentence
          3. join the selected sentences to form the summary
        """
        if not self.text or self.text == "":
            return DEFAULT_SUMMARY_TEXT
        print(f"from_text_to_sentences_and_summary - self.text: {self.text}")

        with measure_time(inspect.currentframe().f_code.co_name + " - load spaCy model"):
            try:
                nlp = spacy.load('en_core_web_sm')
            except OSError:
                from spacy.cli import download
                download('en_core_web_sm')
                nlp = spacy.load('en_core_web_sm')
            doc = nlp(self.text)

        # Calculate word frequencies
        with measure_time(inspect.currentframe().f_code.co_name + " - calculate word frequencies"):
            word_frequencies = {}
            for word in doc:
                if word.text.lower() not in STOP_WORDS and word.text.lower() not in punctuation:
                    if word.text.lower() not in word_frequencies:
                        word_frequencies[word.text.lower()] = 1
                    else:
                        word_frequencies[word.text.lower()] += 1

        if not word_frequencies:
            print("from_text_to_sentences_and_summary - No relevant words found in the text.")
            # Proceed to reinitialize the attributes
            self.summary = ""
            self.sentences = []
            self.summary_sentences = []
            self.text = ""
            return DEFAULT_SUMMARY_TEXT

        # Normalize word frequencies
        with measure_time(inspect.currentframe().f_code.co_name + " - normalize word frequencies"):
            print(f"word_frequencies.values(): {word_frequencies.values()}")
            max_frequency = max(word_frequencies.values())
            for word in word_frequencies.keys():
                word_frequencies[word] = word_frequencies[word] / max_frequency

        # Calculate sentence scores
        with measure_time(inspect.currentframe().f_code.co_name + " - calculate sentence scores"):
            sentence_scores = {}
            for sent in doc.sents:
                print(f"from_text_to_sentences_and_summary - sent.text: {sent.text}")
                # If the sentence is not empty
                if sent.text.strip():
                    # Add the sentence
                    self.sentences.append(sent.text)
                for word in sent:
                    if word.text.lower() in word_frequencies:
                        if sent not in sentence_scores:
                            sentence_scores[sent] = word_frequencies[word.text.lower()]
                        else:
                            sentence_scores[sent] += word_frequencies[word.text.lower()]

        # Select top sentences
        select_length = int(len(sentence_scores) * SENTENCE_SCORE_THRESHOLD)
        summary_sentences = nlargest(select_length, sentence_scores, key=sentence_scores.get)

        # Join selected sentences to form the summary
        with measure_time(inspect.currentframe().f_code.co_name + " - join selected sentences"):
            if len(summary_sentences) > 0:
                self.summary_sentences = [sent.text for sent in summary_sentences]    
                summary = ' '.join(self.summary_sentences)
                self.summary = summary
            else:
                summary = self.text # In this case, no sentences were selected, so we return the original text
            return summary


    # ===========================
    # NAMED ENTITY RECOGNITION (NER)
    # ===========================
    """ 04/28/2025 reactivate the code below to get the list of entities"""
    def perform_ner_analysis_second(self):
        """
        Perform Named Entity Recognition (NER) on the text and return a dictionary of entities.
        The dictionary contains entity labels as keys and lists of entity texts as values.
        """
        if not self.text or self.text == "":
            return {} 
        
        with measure_time(inspect.currentframe().f_code.co_name):
            try:
                nlp = spacy.load('en_core_web_sm')
            except OSError:
                from spacy.cli import download
                download('en_core_web_sm')
                nlp = spacy.load('en_core_web_sm')

            doc = nlp(self.text)
            ner_analysis = defaultdict(list)

            # Extract named entities and group them by their labels
            for ent in doc.ents:
                ner_analysis[ent.label_].append(ent.text)

            # Convert defaultdict to a regular dictionary
            ner_analysis = dict(ner_analysis)
            # Store the entities in an attribute
            self.entities = ner_analysis
            return ner_analysis

    """ ok on 73MB Nov.24; not complete on other test vidos """
    """ Need a complementary method to get the list of entities """
    def perform_ner_analysis(self):
        """
        Perform Named Entity Recognition (NER) on the text or the summary
        """
        self.entities = []
        if not self.text or self.text == "":
            return self.entities 
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            from spacy.cli import download
            download('en_core_web_sm')
            nlp = spacy.load('en_core_web_sm')
        
        doc = nlp(self.text) # self.summary
            
        # Extract named entities -> str
        spaces_count = self.text.count(' ')
        entities = [(ent.text, 'PRODUCT' if ent.label_=='PERSON' and ent.text.count(' ') > 0 else 'ORG' if ent.label_=='PERSON' else ent.label_) for ent in doc.ents]
        print(f'entities: {entities}')
        
        # Store the entities in an attribute
        self.entities = entities
        return entities
    
    def _get_surrounding_text(self, entity, doc, window_size=2)->str:
        """
        Get the surrounding text of an entity in the document.
        Args:
            entity: The entity object from spaCy.
            doc: The spaCy document object.
            window_size: The number of words to include before and after the entity.
        Returns:
            str: The surrounding text of the entity.
        """
        start = max(0, entity.start - window_size)
        end = min(len(doc), entity.end + window_size)
        return doc[start:end].text

    def get_entities_surrounding_infos(self, window_size=2)->dict:
        """
        Get the surrounding information of the entities
        Prompt: get the surrounding information of the entities
        """
        # Initialize spaCy model
        if not self.text or self.text == "":
            return {}
        
        with measure_time(inspect.currentframe().f_code.co_name):
            try:
                nlp = spacy.load('en_core_web_sm')
            except OSError:
                from spacy.cli import download
                download('en_core_web_sm')
                nlp = spacy.load('en_core_web_sm')

            # Process the text
            doc = nlp(self.text)
            surrounding_infos = {}

            for ent in doc.ents:
                sub_sentence = self._get_surrounding_text(ent, doc, window_size)
                blob = TextBlob(sub_sentence)
                sentiment = blob.sentiment
                sentiment_label = "positive" if sentiment.polarity > 0 else "negative" if sentiment.polarity < 0 else "neutral"
                surrounding_infos[ent.text] = {
                    "label": ent.label_,
                    #"start": ent.start_char,
                    #"end": ent.end_char,
                    "sentence": sub_sentence,
                    "sentiment": {
                        "label": sentiment_label,
                        "polarity": round(blob.polarity, 3),
                        "subjectivity": round(blob.subjectivity, 3)
                    }, 
                    #ent.sent.text
                }
            
            return surrounding_infos
    
    # ===========================
    # KEY INFOS EXTRACTION
    # ===========================
    # complementary method to get the list of entities / key infos
    def extract_key_infos(self)->dict:
        """
        Extract key information from the text using a reference JSON file.
        The reference JSON file contains the key information to extract.
        The method reads the reference JSON file, extracts the key information from the text,
        and saves the key information to a JSON file.
        
        read Json file: verifications/content_analysis_reference.json
        ATTENTION: ne pas passer le nom de fichier en dur dans le code
        Use a serialization or other method to get the reference file
        read the path of the reference json file in the config file
        """
        if not self.text or self.text == "":
            return {}

        JSON_ANALYSIS_PATH = os.path.join('verifications', 'content_analysis_reference.json')

        reference_analysis_file = read_json_file(JSON_ANALYSIS_PATH)
        if not reference_analysis_file:
            print(f'Error: Unable to read the reference JSON file: verifications/content_analysis_reference.json')
            return None
        
        # text in lower case
        text_lower_case = self.text.lower()

        # Extract the key infos from the reference analysis file
        key_infos = {}
        for key, value in reference_analysis_file.items():
            if isinstance(value, list):
                # Fetch and collect in a lists the values from the list present in self.text
                list_infos = [item for item in value if item in self.text]
                if list_infos:
                    key_infos[key] = list_infos
            elif isinstance(value, str):
                # Fetch the value from the string present in self.text
                if value in self.text:
                    key_infos[key] = value
            elif isinstance(value, dict):
                # Fetch the values from the dictionary present in self.text
                # all lists
                print(f"extract_key_infos key: {key}, value: {value}")
                for sub_key, sub_value in value.items():
                    if not key in key_infos.keys():
                        key_infos[key] = {}
                    if isinstance(sub_value, list):
                        list_infos = [item for item in sub_value if item.lower() in text_lower_case]
                        if list_infos:
                            key_infos[key][sub_key] = list_infos
                    elif isinstance(sub_value, str):
                        if sub_value in self.text:
                            key_infos[key][sub_key] = sub_value
                    else:
                        print(f"Unsupported data type for key: {sub_key}")
            else:
                print(f"Unsupported data type for key: {key}")
        # Store the key infos in an attribute
        self.key_infos = key_infos
        # Save the key infos to a JSON file
        json_filename = 'content_analysis_key_infos.json'
        save_json_file("key_infos", key_infos, json_filename)
        return key_infos
    
    def extract_type_product_specifications(self, product_type:str)->dict:
        """
        Extract the product specifications from the text using a reference JSON file.
        The reference JSON file contains the product specifications to extract.
        The method reads the reference JSON file, extracts the product specifications from the text,
        and saves the product specifications to a JSON file.
        Args:
            product_type (str): The type of product to extract specifications for.
        Returns:
            dict: The product specifications extracted from the text.
        """
        # Will need to use the specific serialized for the product type
        # NB: 'price' will be a common key for all product types
        # Extract the product specifications from the text
        # Prompt: extract the product specifications from the text
        # Initialize spaCy model
        # ATTENTION: Need a solution to find the specifications of the product type extracted from the text
        # TEST
        product_type = 'laptop'
        if not self.text or self.text == "":
            return {}

        JSON_PRODUCT_SPECIFICATIONS_PATH = os.path.join('products_specifications', product_type+'_specifications.json')
        product_type_specifications_data = read_json_file(JSON_PRODUCT_SPECIFICATIONS_PATH)
        if not product_type_specifications_data:
            print(f'Error: Unable to read the reference JSON file:' + JSON_PRODUCT_SPECIFICATIONS_PATH)
            return None
        #if not self.key_infos.type_product:
        #    print(f'Error: Unable to find the product type in the text')
        #    return None
        
        #print(f'self.key_infos.type_product: {self.key_infos.type_product}')

        # fill the specifications with the product type
        product_specifications = {}
        # ATTENTION: how to fill the product specifications with a generic code?
        for key, value in product_type_specifications_data.items():
            # get the text around the keyword
            #product_specifications[key] = self.text # values from the text
            if key == "price":
                product_specifications["price"] = []
                # get the currency
                if "€" in self.text:
                    product_specifications["currency"] = "euro"
                    # get the indices of the currency in self.text
                    indices = [m.start() for m in re.finditer("€", self.text)]
                    # get the price infos from the text as a list
                    for index_price in indices:
                        # get the price before the currency
                        price = self.text[index_price+1:index_price+5] # need to make it robust: extend to next space
                        # remove the spaces
                        price = price.replace(" ", "")
                        # remove the currency
                        price = price.replace("€", "")
                        # add the price to the product specifications
                        product_specifications["price"].append(price)            

        print(f'product_specifications: {product_specifications}')
        self.product_specifications = product_specifications

        # save the product specifications to a JSON file
        json_filename = 'content_analysis_product_specifications.json'
        save_json_file("product_specifications", product_specifications, json_filename)

        return product_specifications

    # ===========================
    # TOPIC MODELING
    # ===========================
    """
    perform topic modeling using LDA in Python with the gensim library. 
    You can modify the documents list to analyze your own text data. 
    Once you run the code, you'll see the identified topics printed in the console and a visual representation of 
    the topics in a browser. This can provide insight into the themes present in your text data.
    """
    def _preprocess(self, doc):
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(doc.lower())
        return [word for word in tokens if word.isalnum() and word not in stop_words]

    
    def topic_modeling_LSA(self):
        """
        Perform topic modeling using Latent Semantic Analysis (LSA) on the text.
        - The method uses TF-IDF vectorization and Truncated SVD to extract topics.
        - The number of topics is set to 2 for demonstration purposes.
        - The method stores the topics in the list_topics_from_summary attribute.
        """
        # Perform topic modeling using LSA
        self.list_topics_from_summary = []
        print(f"self.sentences: {self.sentences}")

        # Vectorize the sentences using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(self.sentences)

        # Apply LSA (Truncated SVD)
        num_topics = 2
        lsa_model = TruncatedSVD(n_components=num_topics, random_state=42)
        lsa_topic_matrix = lsa_model.fit_transform(X)

        # Get the terms associated with each topic
        terms = vectorizer.get_feature_names_out()
        for idx, topic in enumerate(lsa_model.components_):
            topic_terms = [terms[i] for i in topic.argsort()[:-11:-1]]
            self.list_topics_from_summary.append(f"Topic {idx}: {' '.join(topic_terms)}")


    def one_word_topic_per_summary_sentence(self):
        """
        Extract one-word topics from each sentence in the summary.
        The method uses spaCy to process the sentences and extract the most relevant noun or proper noun as the topic.
        If no noun or proper noun is found, the first word of the sentence is used as the topic.
        The method stores the one-word topics in the one_word_topics attribute.
        """
        # for each sentence in the summary, get the one word topic
        # Prompt: get the one word topic for each sentence in the summary
        # Initialize spaCy model
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            from spacy.cli import download
            download('en_core_web_sm')
            nlp = spacy.load('en_core_web_sm')

        # Process each sentence in the summary
        self.one_word_topics = []
        for sentence in self.sentences:
            doc = nlp(sentence)
            # Extract the most relevant noun or proper noun as the topic
            topic = None
            for token in doc:
                if token.pos_ in ['NOUN', 'PROPN']:
                    topic = token.text
                    break
            # If no noun or proper noun is found, use the first word as the topic
            if not topic:
                topic = doc[0].text
            self.one_word_topics.append(topic)
           

    def sentences_per_topic(self):
        """
        Split the text into sentences related to each topic.
        The method uses the list of topics to categorize the sentences.
        """
        # split the full text into list of texts per topic
        # Prompt.1: split a text into main topics (get_list_topics) with a max of 8 topics
        # Prompt.2: create a Json file which split the text into given topics: for each topic, get the list of sentences related to this topic
        # Tokenize the text into sentences
        sentences = sent_tokenize(self.text)
        
        # Create a dictionary to store sentences per topic
        topic_sentences = defaultdict(list)
        
        # Iterate over each sentence and assign it to a topic which is the sentence  
        for sentence in sentences:
            for topic in self.topics:
                if topic.lower() in sentence.lower():
                    topic_sentences[topic].append(sentence)
                    break
        
        # Convert the defaultdict to a regular dictionary
        topic_sentences = dict(topic_sentences)
        
        # Save the dictionary as a JSON file
        with open(self.output_json, 'w', encoding='utf-8') as json_file:
            json.dump(topic_sentences, json_file, indent=4, ensure_ascii=False)    
        return topic_sentences
    

    # ===========================
    # SENTIMENT ANALYSIS
    # ===========================
    def sentiment_analysis_per_summary_sentence(self, json_sentiments_filename:str)->list:
        """
        Perform sentiment analysis on each sentence in the summary.
        The method uses TextBlob to analyze the sentiment of each sentence.
        The method stores the sentiment scores in the sentiment_scores attribute.
        Args:
            output_json_file (str): Path to save the sentiment analysis results as a JSON file.
        Returns:    
            list: A list of dictionaries containing the sentiment scores for each sentence.
        """
        self.sentiment_scores = []

        print(f"entiment_analysis_per_summary_sentence - self.sentences: {self.sentences}")
        print(f"entiment_analysis_per_summary_sentence - self.text: {self.text}")

        if not self.text or self.text == "":
            return self.sentiment_scores

        # POUR TESTER:
        #self.sentences = ["I love this movie", "I hate this movie", "This movie is okay", "I am neutral about this movie",
        #                  "I am very happy", "I am very sad", "I am feeling great", "I am feeling terrible"]

        # Perform sentiment analysis on each sentence (not the summary per se)
        for sentence in self.sentences:
            blob = TextBlob(sentence)
            sentiment = blob.sentiment
            sentiment_label = "positive" if sentiment.polarity > 0 else "negative" if sentiment.polarity < 0 else "neutral"
            self.sentiment_scores.append({
                "sentence": sentence,
                "sentiment_label": sentiment_label,
                "polarity": round(sentiment.polarity,3),
                "subjectivity": round(sentiment.subjectivity,3)
            })
        # Save the dictionary as a JSON file
        # use save_json_file function
        save_json_file("sentiment_analysis", self.sentiment_scores, json_sentiments_filename)
  
        return self.sentiment_scores

    
    def get_list_topics(self):
        pass

    def summarize_topics(self):
        # Placeholder for summarizing topics
        pass

    def get_topic_summary(self, topic):
        # Placeholder for getting the summary of a specific topic
        pass

    def get_topic_keywords(self, topic):
        # Placeholder for getting the keywords related to a specific topic
        pass

    def get_topic_sentiment(self, topic):
        # Placeholder for getting the sentiment of a specific topic
        pass



#CTRL+i -> chat prompt: build a class derived from VideoBaselineClass to detect objects in a video
"""
1. extraction de la taille des frames de la video
2. random image (frame) from the video: 
3. detection des objets dans l'image
4. dessiner des boîtes bleues autour des objets détectés & label & confidence
"""

# CHILD CLASS N°3: VideoToObjectsClass
class VideoToObjectsClass(VideoBaselineClass):
    def __init__(self, video_path, frame_size=None, duration=None, video_type=None, random_time=None, list_object_types=None):
        super().__init__(video_path, frame_size, duration, video_type, random_time)
        self.object_types = list_object_types
        self.detections = []

    def extract_video_frame_size(self):
        self.video = self.loadVideo()
        self.frame_size = (self.video.w, self.video.h)

    def get_video_frame_size(self):
        # Extraction de la taille des frames de la video
        self.extract_video_frame_size()

    def draw_boxes(self, frame):
        """
        Draw red boxes around detected objects on the frame.
        Args:
            frame (numpy.ndarray): The video frame on which to draw the boxes.
        Returns:
            None
        """
        for detection in self.detections:
            start_x, start_y, end_x, end_y = detection["box_coordinates"]
            print(f"start_x: {start_x}, start_y: {start_y}, end_x: {end_x}, end_y: {end_y}")
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            # Add text above the rectangle: confidence and class
            label = f"{detection['confidence']:.2f} {detection['classe']}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(start_y, label_size[1] + 10)
            cv2.putText(frame, label, (start_x, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # DO NOT PUT THE SAVE FRAME HERE! Save the frame as an image with current date and time
        #output_image_path = "Output/frame_with_detections.jpg"
        #cv2.imwrite(output_image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


    # ===========================
    # TEXT DETECTION IN FRAMES
    # ===========================
    def recognize_text_in_frame(self, method:str)->bool:
        """
        Recognize text in the video frame using the specified method.
        Args:
            method (str): The method to use for text recognition ('easyocr' or 'yolo').
        Returns:
            bool: True if text recognition is successful, False otherwise.
        """
        if method == 'easyocr':
            return self.recognize_text_in_frame_easyocr()
        elif method == 'yolo':
            return self.recognize_text_in_frame_yolo(yolo_version='yolov8n')
        elif method == 'tesseract':
            return self.recognize_text_in_frame_tesseract()
        else:
            print(f"Unknown method: {method}")
            return False


    def recognize_text_in_frame_yolo(self, yolo_version:str='yolov5s')->bool:
        """
        Detect brands and products in the video frame using YOLOv5.
        The method uses the YOLOv5 model to perform object detection on the frame.
        The detected objects are stored in the self.detections attribute.
        Returns:
            list: A list of dictionaries containing information about detected objects.
            Each dictionary contains the object type, start and end times, confidence score, class, and bounding box coordinates.
        """
        # Need to spend time on the design of the Brand detection: so many brands
        #LIST_BRANDS = ['Nescafe', 'Apple', 'Samsung', 'Honor', 'Oppo', 'Huawei', 'Xiaomi'] # in imported constants
        #LIST_PRODUCTS = ['phone', 'laptop', 'tablet', 'watch', 'headphones']
        # Load the YOLO model (ensure you have the correct model path)
        if yolo_version not in ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x', 'yolov8n']:
            print(f"Invalid YOLO version: {yolo_version}. Using default 'yolov5s'.")
            yolo_version = 'yolov5s'
        if yolo_version == 'yolov5s':
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        elif yolo_version == 'yolov8n':
            model = YOLO('yolov8n.pt')  # Load YOLOv8n model weights
        else:
            return False  # Unsupported YOLO version

        # Load the image
        image = self.frame

        # Perform object detection
        results = model(image)

        return True
    
    def detect_brands_logos(self)->bool:
        pass

    
    def recognize_text_in_frame_easyocr(self)->bool:
        """
        Detect brands and products in the video frame using EasyOCR.
        The method uses the EasyOCR model to perform text recognition on the frame.
        The recognized text is stored in the self.detections attribute.
        Returns:
            list: A list of dictionaries containing information about detected objects.
            Each dictionary contains the object type, start and end times, confidence score, class, and bounding box coordinates.
        ATTENTION: need to have the language of the video here!
        """
        # Load the EasyOCR model
        reader = easyocr.Reader(['en'], gpu=False)  # Use GPU if available

        # Perform text detection and recognition
        results = reader.readtext(self.frame)

        text_list = []

        frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        print(f"recognize_text_in_frame_easyocr frame.shape: {frame.shape}")

        # Add attribute: self.detected_user_id
        self.detected_user_id = ""

        # Process the results and filter out weak confidence results
        for (bbox, text, prob) in results:
            if prob >= TEXT_DETECTION_CONFIDENCE_THRESHOLD:
                # Add the recognized text to the list and its confidence
                text_list.append(text + " ; " + str(round(prob,3)))
                # check if the text matches User ID number on the platform
                text_match_user_id = compare_strings(text, USER_ID_TEST, USER_ID_MIN_MATCHING_CHARACTERS)
                if text_match_user_id:
                    self.detected_user_id = text
                    text_list.append("detected USER ID : " + text + " ; " + str(round(prob,3)))

                # Extract the bounding box coordinates
                (top_left, top_right, bottom_right, bottom_left) = bbox
                (top_left_x, top_left_y) = top_left
                (bottom_right_x, bottom_right_y) = bottom_right

                # Draw the bounding box on the image
                start_x = int(top_left_x)
                start_y = int(top_left_y)
                end_x = int(bottom_right_x)
                end_y = int(bottom_right_y)
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
                # Add text above the rectangle
                label = f"{text} ({prob:.2f})"
                cv2.putText(frame, label, (int(top_left_x), int(top_left_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # Save the recognized text and bounding box coordinates
                self.detections.append({
                    "object_type": text,
                    "start": random.randint(0, int(self.video.duration)),
                    "end": random.randint(0, int(self.video.duration)),
                    "confidence": prob,
                    "classe": text,
                    "box_coordinates": [int(top_left_x), int(top_left_y), int(bottom_right_x), int(bottom_right_y)]
                })

        if len(text_list) > 0:
            # Save text list in a file
            save_ascii_file("frame_text_reading", text_list, "frame_text_reading.txt")
            # Save the image with bounding boxes for the recognized text
            save_frame_to_jpeg(frame, 'frame_with_text_recognition.jpg', False)

        return len(text_list) > 0  # Return True if any text was detected
    

    def recognize_text_in_frame_tesseract(self):
        """
        Detect brands and products in the video frame using Tesseract OCR.
        The method uses the Tesseract OCR model to perform text recognition on the frame.
        The recognized text is stored in the self.detections attribute.
        Returns:
            list: A list of dictionaries containing information about detected objects.
            Each dictionary contains the object type, start and end times, confidence score, class, and bounding box coordinates.
        """
        # Load the Tesseract OCR model
        # Perform text detection and recognition
        #results = pytesseract.image_to_data(self.frame, output_type=pytesseract.Output.DICT)
        results = pytesseract.image_to_string(self.frame, config='--psm 6', lang='eng')
        # Print the detected text
        print(f"results text: {results}")
        # Print the detected text
        #print(f"results['text']: {results['text']}")

        """
        # Print the detected text
        for i in range(len(results['text'])):
            # Filter out weak confidence results
            if int(results['conf'][i]) > 20:
                print(f"Detected text: {results['text'][i]} (Confidence: {results['conf'][i]})")
                # Extract the bounding box coordinates
                (top_left_x, top_left_y) = (results['left'][i], results['top'][i])
                (bottom_right_x, bottom_right_y) = (results['left'][i] + results['width'][i], results['top'][i] + results['height'][i])
                # Draw the bounding box on the image
                cv2.rectangle(self.frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
                # Add text above the rectangle
                label = f"{results['text'][i]} ({results['conf'][i]})"
                cv2.putText(self.frame, label, (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # Save the recognized text and bounding box coordinates
                self.detections.append({
                    "object_type": results['text'][i],
                    "start": random.randint(0, int(self.video.duration)),
                    "end": random.randint(0, int(self.video.duration)),
                    "confidence": results['conf'][i],
                    "classe": results['text'][i],
                    "box_coordinates": [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
                })
        """
        # Save the image with bounding boxes

    # ===========================
    # OBJECT DETECTION IN FRAMES
    # ===========================
    def detect_objects(self)->bool:
        """
        Detect objects in the video frame using YOLOv5.
        The method uses the YOLOv5 model to perform object detection on the frame.
        The detected objects are stored in the self.detections attribute.
        Returns:
            list: A list of dictionaries containing information about detected objects.
            Each dictionary contains the object type, start and end times, confidence score, class, and bounding box coordinates.
        """
        # Load the pre-trained YOLO model
        ##model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        # Define the transformation to convert frames to the format expected by YOLO
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])

        # add a counter for testing
        counter = 0
        counter_max = 2

        detections = []
        ### Attention: le random frame (image) est fixé dans la méthode save_frame_with_detections !!
        frame = self.frame
        # Transform the frame
        frame_transformed = transform(frame).unsqueeze(0)  # Add batch dimension

        # Perform object detection
        #with torch.no_grad():
        #    ############# ATTENTION: need to check the model and the frame size
        #    results = model(frame_transformed)
        results = model(frame)  # Perform object detection
        # Display the results
        #results.show()  # Displays the image with bounding boxes
        # Save the results in the specified directory and the specified file name
        output_path = save_frame_to_jpeg(frame, 'frame_objects_detection', False, False)
        # Get the output directory from the output path
        output_dir = os.path.dirname(output_path)
        print(f"output_dir: {output_dir}")

        # Rename output filename to include the current date and time
        current_time = datetime.now().strftime('%H-%M-%S')
        output_filename = f"frame_objects_detection_{current_time}.jpg"
        # Modify results.pandas().files to include the output filename
        print(f"results.__dict__.keys(): {results.__dict__.keys()}")
        results.__dict__['files'] = [output_filename]  # Set the filename to the image name 
        print(f"results.pandas().files: {results.pandas().files}")
        results.save(save_dir=output_dir, exist_ok=True)  # Saves the image with bounding boxes
        #results.save()  # Saves the image with bounding boxes to 'runs/detect/exp'
        # Quite good : saved the image in the proper directory

        # Process the results: voir class_frame.py
        print(f"results.__dict__: {results.__dict__};  results.__dict__.keys(): {results.__dict__.keys()}")
        print(f"results.pandas(): {results.pandas()}")

        # Save results.pandas().xyxy to self.object_detections
        #detections = []
        self.object_detections = results.pandas().xyxy  # Save the pandas DataFrame of detections
        print(f"self.object_detections: {self.object_detections}")
        print(f"type(self.object_detections): {type(self.object_detections)}")
        """
        self.object_detections: [       xmin        ymin         xmax        ymax         confidence  class name
                                    0   689.487732  156.435349  1163.907471  434.881836    0.890000      0  person
                                    1   196.987122  143.403168   460.037537  435.118378    0.732596      0  person
                                    2  1110.970093  301.935913  1158.418091  338.731079    0.672179      0  person
                                    3   936.182434  299.395874   975.041199  423.912048    0.410017     27  tie]
        """
        return results.pandas().xyxy is not None
    

    def save_frame_with_detections(self):
        """
        Save a random frame (snapshot) from the video with detected objects.
        The frame is saved with the current date and time in the filename.
        The frame is processed to draw bounding boxes around detected objects.
        """
        print(f"self.video.duration: {self.video.duration}")
        # RANDOM IMAGE FROM THE VIDEO 
        # Extract a random frame(image) from the video
        frame_time = random.uniform(0, self.video.duration)
        print(f"frame_time: {frame_time}")
        frame = self.video.get_frame(frame_time)

        # Save frame as an attribute
        self.frame = frame

        # Only once the frame above is defined, call detect_objects to build detections object
        #self.detections = self.detect_objects()
        self.any_object_detected = self.detect_objects()

        # Assess wherther a bottle is detected and real
        if self.any_object_detected:
            target_object = 'person'  # Example target object
            is_targetobject_detected_and_real = self.assess_object_is_real(target_object, confidence_threshold=0.5)
            print(f"Is {target_object} detected and real: {is_targetobject_detected_and_real}")

        ## Save frame with current date and time in file name
        self.output_path = save_frame_to_jpeg(frame, 'frame_with_detections.jpg', save_file=False)


    def assess_object_is_real(self, target_object:str, confidence_threshold:float=0.5)->bool:
        """
        Assess if the detected object is real based on its type and confidence score.
        The object type is the one reviewed by the user in the video.
        Args:
            object_type (str): The type of object to assess.
            confidence_threshold (float): The minimum confidence score to consider the object as real.
        Returns:
            bool: True if the object is real, False otherwise.
        Algo:
        1. Iterate through the list of detections
        2. For each detection, check if the object type matches the specified type
        3. If the object box is inside a box of cell phone or tv then object is not real (need to store the boxes boundaries of each detected object)
        detection

        If the target object is labeled as False, then the overall value accounting for the whole video is False: object not real.
        Except if the target object is : cell phone 
        Need to store the values of the outputs of the function in a txt output file
        """
        # Print the column 'name' of the object detections DataFrame
        print(f"type(self.object_detections): {type(self.object_detections)}")
        # Convert the object_detections to a DataFrame if it is not already
    
        self.object_detections = convert_list_to_dataframe(self.object_detections, expected_columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'])
        print(f"self.object_detections.columns: {self.object_detections.columns}")

        # Get the list of detected object types
        # 'name' column contains the object types detected by YOLOv5
        # Extract the 'name' column from the DataFrame
        detected_object_types = list(self.object_detections['name'].unique())
        print(f"type(detected_object_types): {type(detected_object_types)}")
        print(f"detected_object_types: {detected_object_types}")
        # Check if the target object is in the detected object types
        print(f"target_object: {target_object}")
        # If the target object is not in the detected object types, return False
        if target_object not in detected_object_types:
            print(f"assess_object_is_real - {target_object} not found in object_detections")
            return False
        elif 'cell phone' not in detected_object_types and 'tv' not in detected_object_types:
            return True # Object detected and not in a box of cell phone or tv
        else:
            # Iterate through the object detections
            # return True if one box of the target_object is not inside a box of cell phone or tv
            for idx, row in self.object_detections.iterrows():
                print(f"row['name']: {row['name']}")
                print(f"row['confidence']: {row['confidence']}")
                if row['name'] == target_object and row['confidence'] >= confidence_threshold:
                    target_box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                    inside_cell_phone_or_tv = False
                    for _, other_row in self.object_detections.iterrows():
                        if other_row['name'] in ['cell phone', 'tv']:
                            other_box = [other_row['xmin'], other_row['ymin'], other_row['xmax'], other_row['ymax']]
                            # Check if target_box is inside other_box +/- 10 pixels
                            # This is a tolerance to account for small variations in bounding box sizes
                            # If the target box is inside a box of cell phone or tv, then the object is not real
                            if (target_box[0] >= other_box[0]-BOX_DIM_TOLERANCE and target_box[1] >= other_box[1]-BOX_DIM_TOLERANCE and
                                target_box[2] <= other_box[2]+BOX_DIM_TOLERANCE and target_box[3] <= other_box[3]+BOX_DIM_TOLERANCE):
                                inside_cell_phone_or_tv = True
                                break
                    if not inside_cell_phone_or_tv:
                        return True
                    else:
                        print(f"assess_object_is_real - {target_object} is inside a box of cell phone or tv")
                        # If the target object is inside a box of cell phone or tv, return False
                        return False
        # If no valid object is found, return False
        return False


    def subimages_detection_from_frame(self, frame):
        """
        Detect objects in the video frame using YOLOv5.
        The method uses the YOLOv5 model to perform object detection on the frame.
        The detected objects are stored in the self.detections attribute.
        Args:
            frame (numpy.ndarray): The video frame on which to perform object detection.
        Returns:
            list: A list of dictionaries containing information about detected objects.
            Each dictionary contains the object type, start and end times, confidence score, class, and bounding box coordinates.
        """
        # Placeholder for subimage detection logic
        pass


    def end_obj_detection(self):
        # Placeholder for any cleanup after detection
        pass

    def nb_objects(self, object_type):
        return sum(1 for detection in self.detections if detection["object_type"] == object_type)

    def main_color(self, object_type):
        """
        TO COMPLETE
        Identify the main color of the detected object.
        Args:
            object_type (str): The type of object to identify the color for.
        Returns:
            str: The main color of the object.
        1. Get the bounding box coordinates of the object
        2. Extract the region of interest (ROI) from the frame
        3. Convert the ROI to HSV color space
        4. Calculate the histogram of the ROI
        5. Find the most common color in the histogram
        6. Map the HSV color to a color name
        7. Return the color name
        8. Save the color name in the detection dictionary
        9. Return the color name
        """
        # Placeholder for main color detection logic
        # This should be replaced with actual color detection code
        for detection in self.detections:
            if detection["object_type"] == object_type:
                return "red"  # Example color
        return None
    

    def estimate_gender_age_from_faces(self):
        """
        First proceed to the face detection
        If a face is detected, proceed to the estimation of
        - gender of speaker
        - age range of speaker

        Ref: Youtube: Python Age & Gender Detection Tutorial | Python OpenCV and CNN guide
        """

        # Constant to put in the constants.py file
        #MODEL_MEAN_VALUES = (104.0, 177.0, 123.0)
        #BLOB_MEAN_VALUES = (78.426, 87.768, 114.895)
        # Load the pre-trained models for face detection, age estimation, and
    
        # Define the pre-trained models paths to be used by opencv-python
        # These files can often be found in OpenCV's GitHub repository or other online resources that host pre-trained models.
        # OpenCV DNN module
        # caffe models downloaded from smahesh 29 Age-and-Gender-Detection GitHub repository
        face_detection_model = "face_models/opencv_face_detector.pbtxt" # prototxt file that defines the model architecture.
        face_detection_weights = "face_models/opencv_face_detector_uint8.pb" # weights file that contains the pre-trained model parameters.
        age_estimation_model = "face_models/age_net.caffemodel"
        age_estimation_weights = "face_models/age_deploy.prototxt"
        gender_estimation_model = "face_models/gender_net.caffemodel"
        gender_estimation_weights = "face_models/gender_deploy.prototxt"

        # Load the face estimation model
        face_net = cv2.dnn.readNet(face_detection_model, face_detection_weights)
        # Load the age estimation model
        age_net = cv2.dnn.readNet(age_estimation_model, age_estimation_weights)
        # Load the gender estimation model
        gen_net = cv2.dnn.readNet(gender_estimation_model, gender_estimation_weights)

        # Setup Classifications
        age_classifications = ["(0-8)", "(8-15)", "(15-25)", "(25-35)", "(35-45)", "(45-60)", "(60,100)"]
        gender_classifications = ["Male", "Female"]

        # Load the image
        ###image_cp = cv2.imread(self.frame)
        #image_cp = self.frame

        # Resize the image to the input size of the model
        #image_cp = cv2.resize(image_cp, (1000, 1500)) # see if use instead of self.frame
        #frame = cv2.cvtColor(image_cp, cv2.COLOR_RGB2BGR)

        frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)

        # Convert the image to a blob
        #blob = cv2.dnn.blobFromImage(self.frame, BLOB_FRAME_SCALE, BLOB_SIZE, MODEL_MEAN_VALUES, swapRB=True, crop=False)
        blob = cv2.dnn.blobFromImage(frame, BLOB_FRAME_SCALE, BLOB_SIZE, MODEL_MEAN_VALUES, swapRB=True, crop=False)
        
        # Set the input to the face detection model
        face_net.setInput(blob)
        
        # Perform face detection
        detected_faces = face_net.forward()

        # Counter on nb valid faces detected
        nb_valid_faces = 0
        
        # Loop over the detected faces
        for i in range(detected_faces.shape[2]):
            # Get the confidence of the detection
            confidence = detected_faces[0, 0, i, 2]
            # Filter out weak detections
            if confidence > FACE_CONFIDENCE_THRESHOLD:
                print(f"confidence: {confidence}")
                # Increment the counter of valid faces detected
                nb_valid_faces += 1
                # Get the bounding box coordinates
                box = detected_faces[0, 0, i, 3:7] * np.array([self.frame.shape[1], self.frame.shape[0], self.frame.shape[1], self.frame.shape[0]])
                (start_x, start_y, end_x, end_y) = box.astype("int")
                # Increase the bounding box coordinates to avoid cropping the face
                start_x = max(0, start_x - 25)
                start_y = max(0, start_y - 25)
                end_x = min(self.frame.shape[1], end_x + 25)
                end_y = min(self.frame.shape[0], end_y + 25)
                # Draw the bounding box on the image: Bug Fix TODO
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                # Extract the face region from the image
                #face = self.frame[start_y:end_y, start_x:end_x]
                print(f"start_x: {start_x}, start_y: {start_y}, end_x: {end_x}, end_y: {end_y}")
                face = frame[start_y:end_y, start_x:end_x]
                print(f"frame.shape: {frame.shape}")
                # Convert the face region to a blob
                print(f"face.shape: {face.shape}")
                face_blob = cv2.dnn.blobFromImage(face, 1.0, BLOB_SIZE, BLOB_MEAN_VALUES, swapRB=False, crop=True)
                # Set booleans for age and gender estimation
                age_estimation = False
                gender_estimation = False 

                # Set the input to the age estimation model
                age_net.setInput(face_blob)
                # Perform age estimation
                # Add an exception here to avoiding blocking the continuation of the program
                try:
                    age_predictions = age_net.forward()
                    # Get the index of the maximum prediction
                    age_index = age_predictions[0].argmax()
                    # Get the predicted age range
                    age_range = age_classifications[age_index]
                    print(f"Predicted age range: {age_range}")
                    age_estimation = True
                except cv2.error as e:
                    print(f"Error during age estimation: {e}")
                    continue
                
                # Set the input to the gender estimation model
                gen_net.setInput(face_blob)
                try:
                    # Perform gender estimation
                    gen_predictions = gen_net.forward()
                    # Get the index of the maximum prediction
                    gen_index = gen_predictions[0].argmax()
                    # Get the predicted gender
                    gender = gender_classifications[gen_index]
                    print(f"Predicted gender: {gender}")
                    gender_estimation = True
                except cv2.error as e:
                    print(f"Error during gender estimation: {e}")
                    continue

                if age_estimation and gender_estimation:
                    # Save the detected face information
                    # Age range
                    label_size, _ = cv2.getTextSize(age_range, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 1)
                    label_y = max(0, start_y-5, label_size[1] + 15)
                    cv2.putText(frame, "Age range: " + age_range, (start_x, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    # Gender range
                    label_y = max(0, label_y - 30, label_size[1] - 40)
                    cv2.putText(frame, "Gender: " + gender, (start_x, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Save the image with bounding boxes for the recognized text
        # Do not save frame when no face is detected
        print(f"nb_valid_faces: {nb_valid_faces}")
        if nb_valid_faces > 0:
            # Save the image with bounding boxes for the recognized text
            save_frame_to_jpeg(frame, 'frame_faces_age_gender.jpg', False)

        return True


# The class VideoBaselineClass is defined with the required attributes and methods for object detection from a predefined list of object types. The class can be used
# to process videos and extract information about objects present in the video sequences. The methods in the class can be implemented to perform object detection,
# count the number of objects of a specific type, and identify the main color of the object. The class provides a structured approach to handle video data and
# perform object detection tasks efficiently.

# CHILD CLASS N°4: VideoPostValidationClass Child class of VideoToSpeechClass
class VideoPostValidationClass(VideoToSpeechClass):
    def __init__(self, video_path, video_text:str=None):
        super().__init__(video_path, video_text) 
        self.video_text = video_text  # The text extracted from the video       
        self.object_types = []
        self.similarity_ratios_to_genAI = []  # computed similarity ratios to consider a speech as similar to the ones generated by AI

    def get_ai_generated_speech(self):
        """
        FUNCTION TO WRITE DURING THE MVP DEVELOPMENT
        Get the AI-generated speech from the video.
        This method should be implemented to extract the AI-generated speech from the video.
        Returns:
            str: The AI-generated speech.
        """
        # Placeholder for AI-generated speech extraction logic
        # For now, return a dummy string
        return "This is a dummy AI-generated speech for testing purposes."
    

    def compute_similarity_ratios(self, generated_speech:str, user_speech:str):
        """
        Args:
            generated_speech (str): The speech generated by AI.
            user_speech (str): The speech provided by the user.
        Returns:
            float: The similarity ratio between the generated speech and the user speech.
        Computes the similarity ratio between the generated speech and the user speech using SequenceMatcher.
        """
        try:
            print(f"compute_similarity_ratios - generated_speech: {generated_speech}")
            print(f"compute_similarity_ratios - user_speech: {user_speech}")
            matcher = SequenceMatcher(None, generated_speech, user_speech)
            similarity_ratio = matcher.ratio()
            print(f"Similarity ratio: {similarity_ratio}")
            self.similarity_ratios_to_genAI.append(similarity_ratio)
            return similarity_ratio
        except Exception as e:
            print(f"compute_similarity_ratios - Error computing similarity ratios: {e}")
            return None
        
        
    def is_ai_generated_speech(self, generated_speech:str)->bool:
        """
        Check if the user speech is similar to the generated speech.
        Args:
            generated_speech (str): The speech generated by AI.
        Returns:
            bool: True if the user speech is similar to the generated speech, False otherwise.
        """
        user_speech = self.video_text  # The text extracted from the video
        if not user_speech or not generated_speech:
            print("is_ai_generated_speech - No user speech or generated speech provided.")
            return False
        similarity_ratio = self.compute_similarity_ratios(generated_speech, user_speech)
        return similarity_ratio, similarity_ratio >= THRESHOLD_SIMILARITY_RATIO if similarity_ratio is not None else False


    def save_ai_validation_result(self, generated_speech:str, user_speech:str, similarity_ratio:float, is_ai_generated:bool, video_path:str):
        """
        Save the AI validation result to a file.
        Args:
            generated_speech (str): The speech generated by AI.
            user_speech (str): The speech provided by the user.
            similarity_ratio (float): The similarity ratio between the generated speech and the user speech.
            is_ai_generated (bool): True if the user speech is similar to the generated speech, False otherwise.
            video_path (str): The path to the video file.
        """
        # Create a dictionary with the validation result
        validation_result = {
            "video_path": video_path,
            "generated_speech": generated_speech,
            "user_speech": user_speech,
            "similarity_ratio": similarity_ratio,
            "is_ai_generated": is_ai_generated
        }
        # Save the validation result to a file
        save_json_file("ai_validation_result", validation_result, "ai_validation_result.json")