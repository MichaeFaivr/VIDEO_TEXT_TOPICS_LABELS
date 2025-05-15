import pickle
import os
from datetime import datetime
from flask import Flask, render_template, request

from model.class_video_copilot import VideoCopilot, VideoToSpeechCopilot, VideoToObjectsCopilot, VideoTopicsSummaryCopilot

from verifications.check_policy_compliance import read_json_file, check_policy_compliance
from verifications.build_analysis_json import *

app = Flask(__name__)

DIRECTORY_VIDEOS = '../Uploads/'

LISTE_OBJETS = ['person', 'cup', 'dish', 'knife', 'bottle', 'scissor', 'cake', 'plate', 'punnet', 'basket', 'eye', 'carrot',
                'bowl', 'fork', 'spoon', 'bag', 'glove', 'book', 'board', 'strawberry', 'hand', 'socket', 'sink', 'handle',
                'cabinet', 'switch', 'lamp', 'banana', 'tree', 'canvas', 'frame', 'chair','glasses','smartphone','laptop',
                'monitor','keyboard','mouse','headphone','earphone','speaker','tablet','camera','projector','printer', 'radio',
                'television','remote','clock','watch','calculator','scale','tape measure','ruler','pencil','pen',
                'lion','box']


TEMP_AUDIO_FILE = "temp_audio.wav" # better to read from config file
# 06-mai TEST
TEMP_AUDIO_FILE = "temp_mono_audio.wav"


@app.route('/', methods=['GET'])
def upload_video(): 
    return render_template('video_upload.html')


@app.route('/analysis/', methods=['POST'])
def result():
    if request.method == 'POST':
        video_file = request.files['video']
        print('filename:', video_file.filename) # ok: file: <FileStorage: 'VID20241018125303.mp4' ('video/mp4')>
        # Save the video file in Uploads folder
        video_file.save(DIRECTORY_VIDEOS + video_file.filename)

        # ==========================
        # OPERATION.1: Extract the text from the video
        # ==========================
        # Attention: Need the punctuation marks in the text
        video_path = DIRECTORY_VIDEOS + video_file.filename
        videoToSpeech = VideoToSpeechCopilot(video_path)
        text_video = videoToSpeech.extract_speech_google_recognizer(TEMP_AUDIO_FILE)
        #print('app.py extract_speech_google_recognizer - text_video:', text_video)   
        #_ = videoToSpeech.extract_speech_with_vosk(TEMP_AUDIO_FILE) # find why not the model in the unzipped folder
        text_video = videoToSpeech.process_text()
        print('app.py process_text - text_video:', text_video) 

        # Transcript of the audio extracted from the video into a text 
        # NOK when called onto video longer than 1 minute
        #_ = videoToSpeech.transcribe_audio_with_punctuation_google_speech_api()

        # Save the text in a file in Outputs avec date et time
        # TO FIX: marks are missing in the text
        videoToSpeech.save_speech_from_video(video_path, text_video)
        # ===========================
        # OPERATION.2: Summarize the text extracted from the video
        # ===========================
        videoTopicsSummary = VideoTopicsSummaryCopilot(text_video, ['test'], 'test.json')
        # Cancel Summarization
        summary_text = videoTopicsSummary.text_summary()
        ##summary_text = ""
        # save summary in a file

        # ===========================
        # OPERATION.3: Key informations of the text from the video
        # ===========================

        # ---------------------------
        # ** Analysis.1: Key infos from the text with NER
        # ---------------------------
        text_NER = videoTopicsSummary.perform_ner_analysis_second()
        text_NER = str(text_NER)
        # save NER in a file

        # ---------------------------
        # ** Analysis.2: NER entities with surrounding text
        # ---------------------------
        WINDOW_SIZE = 5 # need to take the sentences breaks into account
        text_NER_surrounding = videoTopicsSummary.get_entities_surrounding_infos(window_size=WINDOW_SIZE) # needs a window size sufficiently large to get the context of the entity
        text_NER_surrounding = str(text_NER_surrounding)

        # ----------------------------
        # ** Analysis.3: Key topics from the text with Complementary Content Analysis
        # ----------------------------
        _ = videoTopicsSummary.extract_key_infos()

        # ----------------------------        
        # ** Analysis.4: Key topics from the text with product type specific analysis
        # ----------------------------
        _ = videoTopicsSummary.extract_type_product_specifications('laptop')

        # ==========================
        # OPERATION.4: TOPICS from summary
        # ==========================
        #videoTopicsSummary.topic_modeling_LSA()
        #list_topics = videoTopicsSummary.list_topics_from_summary
        #print(f'list_topics: {list_topics}')

        # ===========================
        # OPERATION.5: SENTIMENT ANALYSIS from summary
        # ===========================
        # save Json file in same directory as the text file from video
        #current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')
        json_sentiments_filename = os.path.splitext(video_file.filename)[0] + '_' + current_time + '_json_sentiments_per_sentence.json'
        videoTopicsSummary.sentiment_analysis_per_summary_sentence(json_sentiments_filename)
        print(f'videoTextTopics.sentiment_scores: {videoTopicsSummary.sentiment_scores}')   

        # ===========================
        # OPERATION.6: Save the analysis data in a Json file
        # ===========================
        analysis_json_filename = os.path.splitext(video_file.filename)[0] + '_' + current_time + '_json_video_analysis.json'
        # Save the analysis data in a Json file with the function from build_analysis_json.py
        print(f'app.py text_video: {text_video}')
        if text_video and text_video != '':
            json_video_analysis_file = read_fill_save_json_file(analysis_json_filename, video_path, text_video, videoTopicsSummary.entities, videoTopicsSummary.key_infos, videoTopicsSummary.sentiment_scores)
            if not json_video_analysis_file:
                print(f'Error: Unable to save the JSON file: {analysis_json_filename}')
        # Save the analysis data in a pickle file

        # ===========================
        # OPERATION.7: COMPLIANCE of the text with the policy
        # ===========================
        # Attention: need priorly to have stored the analysis data in a Json file !
        if text_video and text_video != '':
            policy_data_file = 'verifications/cahier_des_charges.json'
            video_analysis_file = json_video_analysis_file

            # Read the policy JSON file
            policy_data = read_json_file(policy_data_file)
            if policy_data:
                # Read the data JSON file
                analysis_data = read_json_file(video_analysis_file)

                # computation of the compliance metrics
                compliance_dict, compliance_metrics = check_policy_compliance(policy_data, analysis_data)
                print('compliance_dict filled:', compliance_dict)
                print('compliance_metrics computed:', compliance_metrics)

        # DISPLAY RESULTS FROM THE VIDEO ANALYSIS
        return render_template('video_analysis.html', 
                            video_path=video_path, 
                            extracted_text=text_video, 
                            summary_text=summary_text, 
                            ner_text=text_NER,
                            ner_surrounding_text=text_NER_surrounding)


@app.route('/objectdetection/', methods=['POST'])
def display_image():
    if request.method == 'POST':
        video_file = request.form['video_path'] ##voir comment récupérer video file
        # ===========================
        # OPERATION.8: Extract objects from the video
        # ===========================
        liste_objets = LISTE_OBJETS
        print('video_file:', video_file) # empty

        videoToObjects = VideoToObjectsCopilot(video_file, list_object_types=liste_objets)
        videoToObjects.get_video_frame_size()
        # Save an image from the video with the detected objects framed in red boxes
        videoToObjects.save_frame_with_detections() # with current date and time in file name
        # Recognize text in the frame
        videoToObjects.recognize_text_in_frame("easyocr")

        return render_template('video_objects_detection.html', frame=videoToObjects.output_path)


if __name__ == '__main__':
    app.debug = True
    app.run(
        host='127.0.0.1',
        port=8000,
        debug=True)
