
"""
Algo:
1. Read the Json file template for data analysis
2. Get the data from the video analysis
3. Create a new Json file with the data from the video analysis from the infos and the template
"""

import json
from moviepy.editor import VideoFileClip
from langdetect import detect
##import ffmpeg
##from pprint import pprint
##from tinytag import TinyTag
import os
from datetime import datetime
import exifread

from commons.json_files_management import *

# Voir si utile de faire une classe ici
"""
        "crm_video_type": "consumer review",
        "nb_speakers": 1,
        "language": "english",
        "adult_speaker": 0,
        "type_environment": "kitchen"
"sentiment": {
        "positive_points": [
            "great camera",
            "long battery life",
            "reliable"
        ],
        "negative_points": [
            "small screen"
        ],
        "main_sentiment": "positive",
        "upcoming_purchase_new_item": 1,
        "budget_for_new_item": 500,
        "currency": "euro"
    },
"content": {
        "product": "",
        "brand_tag": 0,
        "brand": "",
        "serial_number": "",
        "list_positive_points": [],
        "list_negative_points": [],
        "main_sentiment": "",
        "upcoming_purchase_new_item": 0,
        "budget_for_new_item": "",
        "currency": "euro",
        "auth_AI_training_usage": 1
    }
"""

# TODO: Serialize the data to JSON
balises_analyse_video = {"metadata": ["format", "duration_secs", "date", "number_of_words"], 
                         "context":  ["crm_video_type", "nb_speakers", "language", "adult_speaker", "type_environment"],
                         "sentiment":["positive_points", "negative_points", "main_sentiment"],
                         "content":  ["product", "brand_tag", "brand", "serial_number", "list_positive_points", 
                                      "list_negative_points", "main_sentiment", "upcoming_purchase_new_item",
                                      "budget_for_new_item", "currency", "auth_AI_training_usage"]}
"""
def read_json_file(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data
"""

def get_format_video(video_path:str)->str:
    """
    Get the format of the video file.
    """
    format_video = video_path.split('.')[-1]
    return format_video


def get_duration_video(video_path:str)->float:
    clip = VideoFileClip(video_path)
    return clip.duration


def get_date_video(video_path:str):
    # TOFIX: the date of the video file is not always the same as the creation date of the file
    # Get the date of the video file
    # ATTENTION: the processing modifies the creation date of the file to current date !
    creation_time = os.path.getctime(video_path)
    creation_date = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
    return creation_date


def get_geolocation_video(video_path:str)->dict:
    """
    Get the geolocation of the video file.
    code by copilot
    NB: MP4 files can contain geolocation information, but it is not as commonly embedded as it is in image files like JPEGs. 
    The presence of geolocation data in an MP4 file depends on how the file was created and the device or software used to record the video.
    """
    try:
        with open(video_path, 'rb') as f:
            tags = exifread.process_file(f)
            gps_latitude = tags.get('GPS GPSLatitude')
            gps_latitude_ref = tags.get('GPS GPSLatitudeRef')
            gps_longitude = tags.get('GPS GPSLongitude')
            gps_longitude_ref = tags.get('GPS GPSLongitudeRef')

            print(f'GPS Latitude: {gps_latitude}')
            print(f'GPS Latitude Ref: {gps_latitude_ref}')

            if gps_latitude and gps_longitude and gps_latitude_ref and gps_longitude_ref:
                lat = [float(x.num) / float(x.den) for x in gps_latitude.values]
                lon = [float(x.num) / float(x.den) for x in gps_longitude.values]

                latitude = lat[0] + lat[1] / 60. + lat[2] / 3600.
                longitude = lon[0] + lon[1] / 60. + lon[2] / 3600.

                if gps_latitude_ref.values[0] != 'N':
                    latitude = -latitude
                if gps_longitude_ref.values[0] != 'E':
                    longitude = -longitude

                return {"latitude": latitude, "longitude": longitude}
            else:
                print("No GPS data found in the video file.")
                return {"latitude": None, "longitude": None}
    except Exception as e:
        print(f"Error extracting geolocation: {e}")
        return {"latitude": None, "longitude": None}


def get_number_of_words_speech(text_video:str)->int:
    """
    Get the number of words in the speech of the video.
    """
    text_video = text_video.split()
    return len(text_video)


def get_number_of_speakers(text_video):
    """
    TODO: from snapshot of the video, get the number of speakers
    Get the number of speakers in the speech of the video.
    """
    nb_of_persons = 0 # utiliser la fcn de labellisation des objets dans une vidéo
    return nb_of_persons


def get_language(text_video:str)->str:
    """
    Get the language of the speech of the video.
    """
    language = detect(text_video)
    if language == 'en':
        language = 'english'
    elif language == 'fr':
        language = 'french'
    elif language == 'es':
        language = 'spanish'
    elif language == 'de':
        language = 'german'
    elif language == 'it':
        language = 'italian'
    elif language == 'pt':
        language = 'portuguese'
    return language


def get_prices(dict_NER:dict)->str:
    """
    Get the price from the NER dictionary.
    """
    #dict_NER = {'ORG': ['ASUS VivoBook', 'ASUS', 'Intel', 'Office 365', 'IDE', 'ASUS', 'AI'], 'DATE': ['April 2024', '8 months', '8-month'], 'GPE': ['Italy', 'Paris'], 'PRODUCT': ['VivoBook 17', 'VivoBook 17', 'Google Cloud'], 'CARDINAL': ['680', '60', 'four'], 'QUANTITY': ['17 inches'], 'PERSON': ['Iris']}
    prices = []
    if "MONEY" in dict_NER.keys():
        prices = list(set(dict_NER["MONEY"]))
    return prices


def get_currency(text_video:str)->str:
    """
    Get the currency of the speech of the video.
    1. Find all prices strings in the text
    2. Modify the price strings as follow: '200 dollars' -> '$200' ; '200 euros' -> '€200'
    """
    dict_currency = {'dollars': '$', 'euros': '€', 'pounds': '£', 'yen': '¥', 'francs': '₣'}
    """
    modified_text = ""
    word_count = 1
    list_words = text_video.split()
    currency = 'unknown'
    for word in list_words:
        if word in dict_currency.keys() or word in dict_currency.values():
            if word in dict_currency.keys():
                currency = word
                symbol = dict_currency[word]
            else:
                currency = [key for key, val in dict_currency.items() if val == word][0]
                symbol = word
            modified_text += symbol + list_words[word_count-1] + " "
        else:
            modified_text += word + " "
        word_count += 1
            ##currency = word # see to take the most frequent currency
    text_video = modified_text
    """

    # Check if the currency is in the text
    for currency, symbol in dict_currency.items():
        index_currency = text_video.find(currency)
        index_symbol = text_video.find(symbol)
        if index_currency != -1 or index_symbol != -1:
            return currency
    # If no currency is found 
    return 'unknown'


def get_positive_sentiments(text_video):
    """
    Get the positive sentiments of the speech of the video.
    """
    pass


def get_brand_product(dict_NER:dict)->tuple:
    """
    Get the brand and product of the speech of the video.
    """
    #dict_NER = {'ORG': ['ASUS VivoBook', 'ASUS', 'Intel', 'Office 365', 'IDE', 'ASUS', 'AI'], 'DATE': ['April 2024', '8 months', '8-month'], 'GPE': ['Italy', 'Paris'], 'PRODUCT': ['VivoBook 17', 'VivoBook 17', 'Google Cloud'], 'CARDINAL': ['680', '60', 'four'], 'QUANTITY': ['17 inches'], 'PERSON': ['Iris']}
    brand = []
    product = []
    if "ORG" in dict_NER.keys():
        brand = list(set(dict_NER["ORG"]))
    if "PRODUCT" in dict_NER.keys():
        # get unique elements from entity["PRODUCT"]
        product = list(set(dict_NER["PRODUCT"]))

    # complete the list of brand and product with content analysis result

    brand = list(set(brand))
    product = list(set(product))
    return brand, product

def sentiment_analysis(sentiment_scores:list)->dict:
    """
    Get the sentiment analysis of the speech of the video.
    Example: sentiment_scores = [{'sentence': 'The camera is great', 'sentiment_label': 'positive'},
                        {'sentence': 'The battery life is long', 'sentiment_label': 'positive'},
                        {'sentence': 'The screen is small', 'sentiment_label': 'negative'}]
    """
    dict_sentiments = {}
    positive_points = [sent['sentence'] for sent in sentiment_scores if sent['sentiment_label'] == 'positive']
    negative_points = [sent['sentence'] for sent in sentiment_scores if sent['sentiment_label'] == 'negative']
    nb_positive_points = len([sent for sent in sentiment_scores if sent['sentiment_label'] == 'positive'])
    nb_negative_points = len([sent for sent in sentiment_scores if sent['sentiment_label'] == 'negative'])
    if nb_positive_points != nb_negative_points:
        main_sentiment = 'positive' if nb_positive_points > nb_negative_points else 'negative'
    else:
        main_sentiment = 'neutral'
    dict_sentiments['positive_points'] = positive_points
    dict_sentiments['negative_points'] = negative_points
    dict_sentiments['main_sentiment'] = main_sentiment
    dict_sentiments['nb_positive_points'] = nb_positive_points
    dict_sentiments['nb_negative_points'] = nb_negative_points
    return dict_sentiments


def read_fill_save_json_file(json_filename, video_path, text_video, dict_NER, key_infos, sentiment_scores):
    """
    Read the Json file template for data analysis, fill it with the data from the video analysis and save it.
    Args:
        json_file (str): path to the Json file to save
        video_path (str): path to the video file
        text_video (str): text of the video
        dict_NER (dict): dictionary of named entities recognition
        key_infos (dict): dictionary of key information
        sentiment_scores (list): list of sentiment scores
    Returns:
        json_file (str): path to the Json file saved
    """
    # read the template analysis Json file
    # ATTENTION: pas de path en dur dans le code, sinon pas portable
    video_analysis_template = 'verifications/video_analysis_template.json'
    json_data = read_json_file(video_analysis_template)
    if not json_data:
        print(f'Error: Unable to read the template JSON file: {video_analysis_template}')
        return None

    # fill metadata format of json_data
    json_data['metadata']['format'] = get_format_video(video_path)
    json_data['metadata']['duration_secs'] = get_duration_video(video_path)
    json_data['metadata']['date'] = get_date_video(video_path)
    json_data['metadata']['number_of_words'] = get_number_of_words_speech(text_video)
    json_data['metadata']['nb_words_per_minute'] = round(get_number_of_words_speech(text_video) / (json_data['metadata']['duration_secs'] / 60)) if json_data['metadata']['duration_secs'] > 0 else 0
    json_data['metadata']['geoloc'] = get_geolocation_video(video_path)
    # fill context format of json_data
    json_data['context']['crm_video_type'] = "consumer review" # To get the type of video from the choice made by the user on the website
    json_data['context']['nb_speakers'] = get_number_of_speakers(text_video)
    json_data['context']['language'] = get_language(text_video)
    json_data['context']['adult_speaker'] = 0
    json_data['context']['type_environment'] = "kitchen" # To get the type of environment from the choice made by the user on the website
    # fill speech
    json_data['speech'] = text_video
    # fill summary
    json_data['summary'] = "" # Add summary in the args
    # fill content infos of json_data
    brand, product = get_brand_product(dict_NER)
    if key_infos and isinstance(key_infos, dict) and 'type_product' in key_infos.keys():
        json_data['content']['type_products'] = key_infos['type_product']
    json_data['content']['products'] = product
    json_data['content']['brands'] = brand
    json_data['content']['serial_numbers'] = ""
    json_data['content']['NER_tags'] = str(dict_NER)
    json_data['content']['upcoming_purchase_new_item'] = 0
    json_data['content']['budget_for_new_item'] = ""
    json_data['content']['price'] = get_prices(dict_NER)
    json_data['content']['currency'] = get_currency(text_video)
    json_data['content']['auth_AI_training_usage'] = 1
    # fill sentiment format of json_data
    dict_sentiments = sentiment_analysis(sentiment_scores)
    json_data['sentiment']['nb_positive_points'] = dict_sentiments['nb_positive_points']
    json_data['sentiment']['nb_negative_points'] = dict_sentiments['nb_negative_points']
    json_data['sentiment']['positive_points'] = dict_sentiments['positive_points']
    json_data['sentiment']['negative_points'] = dict_sentiments['negative_points']
    json_data['sentiment']['main_sentiment'] = dict_sentiments['main_sentiment']

    # save the Json file
    json_video_analysis_file = save_json_file("json_video_analysis", json_data, json_filename)
    return json_video_analysis_file
    
