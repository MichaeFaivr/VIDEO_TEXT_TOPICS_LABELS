import sys
import os
import argparse
import numpy as np
from src.class_video_copilot import VideoToSpeechCopilot, VideoToObjectsCopilot, VideoTopicsSummaryCopilot
from datetime import datetime

#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger('wqp.main')

def write_limited_text(filename, text, max_size_bytes):
    # Check if file exists and its size
    if os.path.exists(filename):
        file_size = os.path.getsize(filename)
    else:
        file_size = 0

    # If the file is too large, stop writing
    if file_size >= max_size_bytes:
        print(f"File exceeds the size limit of {max_size_bytes} bytes.")
        return

    # Otherwise, write to the file
    with open(filename, 'a') as f:
        # Check remaining space
        remaining_space = max_size_bytes - file_size
        # Write only the allowed amount of data
        f.write(text[:remaining_space])


# faire une version avec une seule vidéo en argument
def main(video_path, video_path2):
    #-------------------------------------------------------
    # OPERATION 1 : Extract the speech from the video - SPEECH
    videoToSpeech = VideoToSpeechCopilot(video_path)
    textVideo = videoToSpeech.extract_speech()
    # Create output directory based on current date
    current_date = datetime.now().strftime('%Y-%m-%d')
    output_dir = os.path.join('Output', current_date)
    os.makedirs(output_dir, exist_ok=True)

    # Create a text filename based on the name from video_path and current time
    current_time = datetime.now().strftime('%H-%M-%S')
    video_filename = os.path.basename(video_path)
    text_filename = os.path.splitext(video_filename)[0] + '_' + current_time + '_text.txt'
    output_path = os.path.join(output_dir, text_filename)

    # Enregistrer le texte extrait dans un fichier texte
    max_size_bytes = 1024*1000  # 1 MB limit
    write_limited_text(output_path, textVideo, max_size_bytes)

    #-------------------------------------------------------
    # OPERATION 2 : TOPICS TEXTS & SUMMARIES & NER & SENTIMENT
    text_filename = os.path.splitext(video_filename)[0] + '_' + current_time + '_text_per_topic.json'
    output_json_topics = os.path.join(output_dir, text_filename)
    file_test_text = output_path
    with open(file_test_text, 'r') as file:
        test_text = file.read()

    # il faudra adapter les topics en fonction du thème/produit de la vidéo !!
    # enregistrement du fichier Json des textes par topic
    videoTextTopics = VideoTopicsSummaryCopilot(test_text, ['RAM', 'customer', 'price', 'programming', 'politics', 'business'], output_json_topics)

    # Json des sentences per topic
    videoTextTopics.sentences_per_topic()

    # SUMMARY du texte de la video
    videoTextTopics.text_summary()

    # save summary in a file
    # save Json file in same directory as the text file from video
    text_video_summary_filename = os.path.splitext(video_filename)[0] + '_' + current_time + '_textVideoSummary.txt'
    file_text_summary = os.path.join(output_dir, text_video_summary_filename)
    max_size_bytes = 1024*1000  # 1 MB limit
    write_limited_text(file_text_summary, videoTextTopics.summary, max_size_bytes)

    # NER from summary
    videoTextTopics.perform_ner_analysis()
    text_video_ner_filename = os.path.splitext(video_filename)[0] + '_' + current_time + '_textVideoNER.txt'
    file_text_ner = os.path.join(output_dir, text_video_ner_filename)
    write_limited_text(file_text_ner, str(videoTextTopics.entities), max_size_bytes)

    # TOPICS from summary
    videoTextTopics.topic_modeling_LSA()
    print(f'videoTextTopics.topics: {videoTextTopics.list_topics_from_summary}')

    # TOPIC PER SENTENCE in the summary
    videoTextTopics.one_word_topic_per_summary_sentence()
    print(f'videoTextTopics.one_word_topics: {videoTextTopics.one_word_topics}')

    # SENTIMENT ANALYSIS from summary
    # save Json file in same directory as the text file from video
    json_sentiments_filename = os.path.splitext(video_filename)[0] + '_' + current_time + '_sentiments_per_sentence.json'
    output_json_sentiments = os.path.join(output_dir, json_sentiments_filename)
    videoTextTopics.sentiment_analysis_per_summary_sentence(output_json_sentiments)
    print(f'videoTextTopics.sentiment_scores: {videoTextTopics.sentiment_scores}')   
    # json : {sentence: '...', sentiment: 'positive', score: 0.8}


    #-------------------------------------------------------
    # OPERATION 3 : Extract objects from the video - OBJECTS
    liste_objets = ['person', 'cup', 'dish', 'knife', 'bottle', 'scissor', 'cake', 'plate', 'punnet', 'basket', 'eye', 'carrot',
                    'bowl', 'fork', 'spoon', 'bag', 'glove', 'book', 'board', 'strawberry', 'hand', 'socket', 'sink', 'handle',
                    'cabinet']
    videoToObjects = VideoToObjectsCopilot(video_path2, list_object_types=liste_objets)

    videoToObjects.get_video_frame_size()

    # Save an image from the video with the detected objects framed in red boxes
    videoToObjects.save_frame_with_detections()

if __name__ == "__main__":
    # Operation 1: Extract the speech from the video: Video.1
    ##video_path = 'Inputs/VID20241018125303.mp4'
    video_path  = 'Inputs/VID20241104151449.mp4'
    # Operation 1: Extract the speech from the video: Video.2
    video_path2  = 'Inputs/VID20240512155311.mp4'
    ##video_path2  = 'Inputs/VID20241104151449.mp4'
    # video_path2 est juste utilisé pour l'opération 3 de détection d'objets
    main(video_path, video_path2)
