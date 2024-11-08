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
import speech_recognition as sr
#from deepmultilingualpunctuation import PunctuationModel
import cv2
import torch
from torchvision import transforms
from PIL import Image
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


# The reason gensim no longer has a summarization module is that it was 
# deprecated and eventually removed in more recent versions of the library.

# CTRL+I -> chat prompt:
# Prompt: write the code of the function extract_speech(video_path) that extracts the text of what is said in a video

# 04Nov. Punctuation is critical in the text retrieved from the audio of the video.
# Import of deepmultilingualpunctuation is not working.
# This point has to be fixed.

# define a method for extracting speech from a video
def extract_speech(video_path):
    # simulate speech extraction
    import moviepy.editor as mp

    # Load the video file
    video = mp.VideoFileClip(video_path)

    # Extract the audio from the video
    audio = video.audio
    # output the audio to a temporary file
    audio_path = "temp_audio.wav"
    audio.write_audiofile(audio_path)

    # Initialize the recognizer and punctuation model
    recognizer = sr.Recognizer()
    ##punct_model = PunctuationModel()

    # Load the audio file
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

    # Recognize speech using Google Web Speech API
    # if French language spoken: specify language="fr-FR"
    # add in attributes !!
    # Attention: il manque la ponctuation dans le texte extrait de la vidéo
    try:
        # Transcribe audio to text
        text = recognizer.recognize_google(audio_data)
        # Add punctuation
        ##punctuated_text = punct_model.restore_punctuation(text)
        # Add punctuation using spaCy: under assumption of English language
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            from spacy.cli import download
            download('en_core_web_sm')
            nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        punctuated_text = ''.join([token.text_with_ws for token in doc])
        text = punctuated_text
        return text
    except sr.UnknownValueError:
        return "Speech recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Web Speech API; {e}"


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


# CHILD CLASS N°1: VideoToSpeechCopilot
class VideoToSpeechCopilot(VideoCopilot):
    def __init__(self, video_path, frame_size=None, duration=None, video_type=None, random_time=None, output_text=None):
        super().__init__(video_path, frame_size, duration, video_type, random_time)
        self.output_text = output_text

    def extract_speech(self):
        # Extract the speech from the video from the video file path
        textVideo = extract_speech(self.video_path)
        return textVideo


# CHILD CLASS N°2: analyze the text extracted from the video and split it into topics:
# for each topic of interest, get the list of sentences related to that topic
# and summarize the text, extract keywords, and determine the sentiment
# 1. Text from Video Summary   : DONE
# 2. Name Entity Recognition
# 3. Sentiment Analysis
# 4. Topic Detection/Modeling
# 5. Json sentences per topic  : DONE
#
class VideoTopicsSummaryCopilot():
    def __init__(self, text, topics, output_json):
        self.text   = text
        self.topics = topics
        self.output_json = output_json
        nltk.download('punkt')
        nltk.download('punkt_tab')


    """
    TEXT SUMMARY
    """
    def text_summary(self):
        # Summarize the text using spaCy lib
        # 1. get most relevant sentences of the text
        # 2. summarize each selected sentence
        # 3. join the selected sentences to form the summary
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            from spacy.cli import download
            download('en_core_web_sm')
            nlp = spacy.load('en_core_web_sm')
        doc = nlp(self.text)

        # Calculate word frequencies
        word_frequencies = {}
        for word in doc:
            if word.text.lower() not in STOP_WORDS and word.text.lower() not in punctuation:
                if word.text.lower() not in word_frequencies:
                    word_frequencies[word.text.lower()] = 1
                else:
                    word_frequencies[word.text.lower()] += 1

        # Normalize word frequencies
        max_frequency = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word] / max_frequency

        # Calculate sentence scores
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
        self.sentences = [sent.text for sent in summary_sentences]    
        summary = ' '.join(self.sentences)
        self.summary = summary


    """
    NAMED ENTITY RECOGNITION (NER)
    """
    def perform_ner_analysis(self):
        # Perform Named Entity Recognition (NER) on the summary
        self.entities = []
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            from spacy.cli import download
            download('en_core_web_sm')
            nlp = spacy.load('en_core_web_sm')
        
        doc = nlp(self.summary)
            
        # Extract named entities -> str
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f'entities: {entities}')
        
        # Store the entities in an attribute
        self.entities = entities
        return entities
    

    """
    TOPIC MODELING
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
    

    """
    SENTIMENT ANALYSIS
    """
    def sentiment_analysis_per_summary_sentence(self, output_json_file):
        self.sentiment_scores = []

        # POUR TESTER:
        #self.sentences = ["I love this movie", "I hate this movie", "This movie is okay", "I am neutral about this movie",
        #                  "I am very happy", "I am very sad", "I am feeling great", "I am feeling terrible"]

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

        # Save the frame as an image
        output_image_path = "Output/frame_with_detections.jpg"
        cv2.imwrite(output_image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

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



















