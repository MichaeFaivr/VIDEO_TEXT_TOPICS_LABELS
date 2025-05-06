# create a class to store video attributes and methods for object detection

# In the class VideoCopilot, we will define the attributes and methods for object detection from a predefined list of object types.
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
# - classe de base: VideoCopilot
# - classe fille #1: VideoToSpeechCopilot
# - classe fille #2: VideoToObjectsCopilot

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

# Use the save_json_file
"""
def save_json_file(json_data, json_filename):
    # ATTENTION: path hard coded in the code for the output folder !
    current_date = datetime.now().strftime('%Y-%m-%d')
    output_dir = os.path.join('../Output', current_date)
    os.makedirs(output_dir, exist_ok=True)
    # Create a text filename based on the name from video_path and current time
    current_time = datetime.now().strftime('%H-%M-%S')
    json_path = os.path.basename(json_filename)
    json_path = os.path.splitext(json_path)[0] + '_' + current_time + '_' + json_filename
    output_path = os.path.join(output_dir, json_path)
    # Save file
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)
    return True
"""

def convert_to_mono(input_audio_file: str, output_audio_file: str) -> str:
    # Load the audio file
    audio = AudioSegment.from_file(input_audio_file)
    # Convert to mono
    mono_audio = audio.set_channels(1)
    # Save the mono audio to a temporary file
    mono_audio_path = output_audio_file
    mono_audio.export(mono_audio_path, format="wav")
    return mono_audio_path

def transcribe_file_with_auto_punctuation(audio_file: str) -> speech.RecognizeResponse:
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


def format_price_strings(text_video):
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
            # remove the preceding string
            print(f"symbol: {symbol}, list_words[word_count-1]: {list_words[word_count-1]}, modified_text[-3:-2]: {modified_text[-3:-2]}")
            modified_text = " ".join(list_words[:word_count-2])
            modified_text += " " + symbol + list_words[word_count-2] + " " # TODO: remove preceding string
        else:
            modified_text += word + " "
        word_count += 1
            ##currency = word # see to take the most frequent currency
    text_video = modified_text
    return text_video

# BASE CLASS: empty methods to fill in child classes
class VideoCopilot:
    def __init__(self, video_path, frame_size, duration, video_type, random_time):
        self.frame_size = frame_size
        self.duration = min(duration, 600) if duration else None  # Trimmed at 10 mins
        self.video_type = video_type
        self.random_time = random_time
        self.video_path = video_path

    def loadVideo(self):
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


# CHILD CLASS N°1: VideoToSpeechCopilot
class VideoToSpeechCopilot(VideoCopilot):
    def __init__(self, video_path, frame_size=None, duration=None, video_type=None, random_time=None, output_text=None):
        super().__init__(video_path, frame_size, duration, video_type, random_time)
        self.output_text = output_text

    def extract_audio(self, input_audio_file):
        # simulate speech extraction
        import moviepy.editor as mp

        # Load the video file
        video = mp.VideoFileClip(self.video_path)

        # Extract the audio from the video
        audio = video.audio
        # output the audio to a temporary file
        audio_path = input_audio_file
        audio.write_audiofile(audio_path)
        self.audio_path = audio_path

        # Initialize the recognizer
        recognizer = sr.Recognizer()

        # Load the audio file
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            self.audio_data = audio_data


    def extract_speech_google_recognizer(self, temp_audio_file):
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

    """"        
    def extract_speech_with_vosk(self, temp_audio_file):
        # Extract the audio from the video
        PATH_VOSK_MODEL = "../language_models/vosk-model-en-us-0.22"
        with measure_time(inspect.currentframe().f_code.co_name):
            ###self.extract_audio(temp_audio_file)
            try:
                # Load the Vosk model
                model = Model(PATH_VOSK_MODEL)  # Path to the Vosk model directory

                # Open the audio file
                with wave.open(temp_audio_file, "rb") as wf:
                    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                        raise ValueError("Audio file must be WAV format mono PCM.")
                    
                    recognizer = KaldiRecognizer(model, wf.getframerate())
                    recognizer.SetWords(True)

                    text_video = ""
                    while True:
                        data = wf.readframes(4000)
                        if len(data) == 0:
                            break
                        if recognizer.AcceptWaveform(data):
                            result = recognizer.Result()
                            text_video += json.loads(result).get("text", "") + " "

                    # Finalize the transcription
                    final_result = recognizer.FinalResult()
                    text_video += json.loads(final_result).get("text", "")

                    self.output_text = text_video
                    return text_video
            except Exception as e:
                return f"Error during speech recognition: {e}"
    """

    def transcribe_audio_with_punctuation_google_speech_api(self, output_audio_mono_path):
        # Input = audio file path
        # convert to mono track
        mono_audio_file = convert_to_mono(self.audio_path, output_audio_mono_path)
        response = transcribe_file_with_auto_punctuation(mono_audio_file)
        print(f"response: {response}")
        return response
    
    def process_text(self):
        # Price formatting
        textVideo = format_price_strings(self.output_text)
        self.output_text = textVideo
        return textVideo
    
    def create_output_path_text(self, video_path_uploaded, text_video):
        # Create the output path for the text file
        # Create output directory based on current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        output_dir = os.path.join('../Output', current_date)
        os.makedirs(output_dir, exist_ok=True)
        # Create a text filename based on the name from video_path and current time
        current_time = datetime.now().strftime('%H-%M-%S')
        video_filename = os.path.basename(video_path_uploaded)
        text_filename = os.path.splitext(video_filename)[0] + '_' + current_time + '_text.txt'
        self.output_path = os.path.join(output_dir, text_filename)

    def save_speech_from_video(self, video_path_uploaded, text_video):
        self.create_output_path_text(video_path_uploaded, text_video)
        max_size_bytes = 1024*1000  # 1 MB limit
        # Write the text to a file with a size limit
        if text_video and text_video != "":
            with open(self.output_path, 'w') as file:
                file.write(text_video[:max_size_bytes])   



# CHILD CLASS N°2: analyze the text extracted from the video and split it into topics:
# for each topic of interest, get the list of sentences related to that topic
# and summarize the text, extract keywords, and determine the sentiment
# 1. Text from Video Summary   : DONE
# 2. Name Entity Recognition   : DONE
# 3. Key Infos Extraction      : DONE
# 4. Sentiment Analysis        : DONE         
# 5. Topic Detection/Modeling
# 6. Json sentences per topic  : DONE
#
class VideoTopicsSummaryCopilot():
    def __init__(self, text, topics, output_json):
        self.text   = text
        self.topics = topics
        self.output_json = output_json
        nltk.download('punkt')
        nltk.download('punkt_tab')


    # ===========================
    # TEXT SUMMARY
    # ===========================
    def text_summary(self):
        # Summarize the text using spaCy lib
        # 1. get most relevant sentences of the text
        # 2. summarize each selected sentence
        # 3. join the selected sentences to form the summary
        if not self.text or self.text == "":
            return "No text to summarize" 

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

        # Normalize word frequencies
        with measure_time(inspect.currentframe().f_code.co_name + " - normalize word frequencies"):
            max_frequency = max(word_frequencies.values())
            for word in word_frequencies.keys():
                word_frequencies[word] = word_frequencies[word] / max_frequency

        # Calculate sentence scores
        with measure_time(inspect.currentframe().f_code.co_name + " - calculate sentence scores"):
            sentence_scores = {}
            for sent in doc.sents:
                for word in sent:
                    if word.text.lower() in word_frequencies:
                        if sent not in sentence_scores:
                            sentence_scores[sent] = word_frequencies[word.text.lower()]
                        else:
                            sentence_scores[sent] += word_frequencies[word.text.lower()]

        # Select top sentences: factor 0.3
        select_length = int(len(sentence_scores) * 0.7)
        summary_sentences = nlargest(select_length, sentence_scores, key=sentence_scores.get)

        # Join selected sentences to form the summary
        with measure_time(inspect.currentframe().f_code.co_name + " - join selected sentences"):
            self.sentences = [sent.text for sent in summary_sentences]    
            summary = ' '.join(self.sentences)
            self.summary = summary
            return summary


    # ===========================
    # NAMED ENTITY RECOGNITION (NER)
    # ===========================
    """ 04/28/2025 reactivate the code below to get the list of entities"""
    def perform_ner_analysis_second(self):
        # Perform Named Entity Recognition (NER) on the text
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
        # Perform Named Entity Recognition (NER) on the text or the summary
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
    
    def _get_surrounding_text(self, entity, doc, window_size=2):
        start = max(0, entity.start - window_size)
        end = min(len(doc), entity.end + window_size)
        return doc[start:end].text

    def get_entities_surrounding_infos(self, window_size=2):
        # Get the surrounding information of the entities
        # Prompt: get the surrounding information of the entities
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
    def extract_key_infos(self):
        # read Json file: verifications/content_analysis_reference.json
        # ATTENTION: ne pas passer le nom de fichier en dur dans le code
        # Use a serialization or other method to get the reference file
        # read the path of the reference json file in the config file
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
        save_json_file(key_infos, json_filename)

        return key_infos
    
    def extract_type_product_specifications(self, product_type: str) -> dict:
        # Wiil need to use the specific serialized for the product type
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
        save_json_file(product_specifications, json_filename)

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
    def sentiment_analysis_per_summary_sentence(self, output_json_file):
        self.sentiment_scores = []

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
                "polarity": sentiment.polarity,
                "subjectivity": sentiment.subjectivity
            })
        # Save the dictionary as a JSON file
        with open(output_json_file, 'w', encoding='utf-8') as json_file:
            json.dump(self.sentiment_scores, json_file, indent=4, ensure_ascii=False)    
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



#CTRL+i -> chat prompt: build a class derived from VideoCopilot to detect objects in a video
"""
1. extraction de la taille des frames de la video
2. random image (frame) from the video: 
3. detection des objets dans l'image
4. dessiner des boîtes bleues autour des objets détectés & label & confidence
"""

# CHILD CLASS N°3: VideoToObjectsCopilot
class VideoToObjectsCopilot(VideoCopilot):
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
        for detection in self.detections:
            start_x, start_y, end_x, end_y = detection["box_coordinates"]
            print(f"start_x: {start_x}, start_y: {start_y}, end_x: {end_x}, end_y: {end_y}")
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            # Add text above the rectangle: confidence and class
            label = f"{detection['confidence']:.2f} {detection['classe']}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(start_y, label_size[1] + 10)
            cv2.putText(frame, label, (start_x, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # DO NOT PUT THE SAVE FRAME HERE! Save the frame as an image wirth current date and time
        #output_image_path = "Output/frame_with_detections.jpg"
        #cv2.imwrite(output_image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def save_frame_with_detections(self):
        print(f"self.video.duration: {self.video.duration}")
        # RANDOM IMAGE FROM THE VIDEO 
        # Extract a random frame(image) from the video
        frame_time = random.uniform(0, self.video.duration)
        print(f"frame_time: {frame_time}")
        frame = self.video.get_frame(frame_time)
        self.frame = frame

        # Only once the frame above is defined, call detect_objects to build detections object
        self.detections = self.detect_objects()

        # If the image is marked as read-only, you can use this to make it writable
        frame = np.array(frame)
        frame.setflags(write=1)

        # Draw red boxes around detected objects: boxes are computed in the ** detect_objects ** method called above
        # error : cv2.error: OpenCV(4.10.0) :-1: error: (-5:Bad argument) in function 'rectangle'
        print(f"len(self.detections): {len(self.detections)}")

        self.draw_boxes(frame)

        ## Save frame with current date and time in file name
        current_date = datetime.now().strftime('%Y-%m-%d')
        output_dir = os.path.join('../Output', current_date)
        os.makedirs(output_dir, exist_ok=True)
        # Create a text filename based on the name from video_path and current time
        current_time = datetime.now().strftime('%H-%M-%S')
        video_filename = os.path.basename('frame_with_detections.jpg')
        video_filename = os.path.splitext(video_filename)[0] + '_' + current_time + '.jpg'
        self.output_path = os.path.join(output_dir, video_filename)
        # Save file
        cv2.imwrite(self.output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


    def detect_objects(self):
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
        ##results = model(frame_transformed)
        results = model(frame)  # Perform object detection
        # Display the results
        results.show()  # Displays the image with bounding boxes
        # Save the results
        results.save()  # Saves the image with bounding boxes to 'runs/detect/exp'

        # Process the results: voir class_frame.py
        print(f"results.__dict__: {results.__dict__}")
        print(f"results.pandas(): {results.pandas()}")

        for result in results.pandas().xyxy:
            print(f"result: {result}, columns: {result.columns}")
            print(f"result.xmin type: {type(result.xmin)}") # result.xmin type: <class 'pandas.core.series.Series'>

            for _, row in result.iterrows():
                x1 = row.xmin
                y1 = row.ymin
                x2 = row.xmax
                y2 = row.ymax
                cnf= row.confidence
                cls= row.name
                print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
                object_type = row.iloc[-1] ##model.names[int(cls)] # voir pourquoi NOK avec model.names...
                print(f"object_type: {object_type}")
                print(f"self.object_types: {self.object_types}")
                if object_type in self.object_types:
                    detections.append({
                        "object_type": object_type,
                        "start": random.randint(0, int(self.video.duration)),
                        "end": random.randint(0, int(self.video.duration)),
                        "confidence": cnf,
                        "classe": object_type,
                        "box_coordinates": [int(x1), int(y1), int(x2), int(y2)]
                        })
        return detections


    def end_obj_detection(self):
        # Placeholder for any cleanup after detection
        pass

    def nb_objects(self, object_type):
        return sum(1 for detection in self.detections if detection["object_type"] == object_type)

    def main_color(self, object_type):
        # Placeholder for main color detection logic
        # This should be replaced with actual color detection code
        for detection in self.detections:
            if detection["object_type"] == object_type:
                return "red"  # Example color
        return None


# The class VideoCopilot is defined with the required attributes and methods for object detection from a predefined list of object types. The class can be used
# to process videos and extract information about objects present in the video sequences. The methods in the class can be implemented to perform object detection,
# count the number of objects of a specific type, and identify the main color of the object. The class provides a structured approach to handle video data and
# perform object detection tasks efficiently.



















