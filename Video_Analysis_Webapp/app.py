import pickle
from flask import Flask, render_template, request

from model.class_video_copilot import VideoCopilot, VideoToSpeechCopilot, VideoToObjectsCopilot, VideoTopicsSummaryCopilot


app = Flask(__name__)

DIRECTORY_VIDEOS = '../Uploads/'

LISTE_OBJETS = ['person', 'cup', 'dish', 'knife', 'bottle', 'scissor', 'cake', 'plate', 'punnet', 'basket', 'eye', 'carrot',
                'bowl', 'fork', 'spoon', 'bag', 'glove', 'book', 'board', 'strawberry', 'hand', 'socket', 'sink', 'handle',
                'cabinet']


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

        # OPERATION.1: Extract the text from the video
        video_path = DIRECTORY_VIDEOS + video_file.filename
        videoToSpeech = VideoToSpeechCopilot(video_path)
        text_video = videoToSpeech.extract_speech()

        # Save the text in a file in Outputs avec date et time
        videoToSpeech.save_speech_from_video(video_path,text_video)

        # OPERATION.2: Summarize the text extracted from the video
        videoTopicsSummary = VideoTopicsSummaryCopilot(text_video, ['test'], 'test.json')
        summary_text = videoTopicsSummary.text_summary()
        # save summary in a file

        # OPERATION.3: NER of the text from the video
        text_NER = videoTopicsSummary.perform_ner_analysis()
        text_NER = str(text_NER)
        # save NER in a file

        # DISPLAY RESULTS FROM THE VIDEO ANALYSIS
        return render_template('video_analysis.html', video_path=video_path, extracted_text=text_video, summary_text=summary_text, ner_text=text_NER)


@app.route('/objectdetection/', methods=['POST'])
def display_image():
    if request.method == 'POST':
        video_file = request.form['video_path'] ##voir comment récupérer video file
        # OPERATION.4: Extract objects from the video
        liste_objets = LISTE_OBJETS
        print('video_file:', video_file) # empty

        videoToObjects = VideoToObjectsCopilot(video_file, list_object_types=liste_objets)
        videoToObjects.get_video_frame_size()
        # Save an image from the video with the detected objects framed in red boxes
        videoToObjects.save_frame_with_detections() # with current date and time in file name
        print('output_path:', videoToObjects.output_path)

        return render_template('video_objects_detection.html', frame=videoToObjects.output_path)


if __name__ == '__main__':
    app.debug = True
    app.run(
        host='127.0.0.1',
        port=8000,
        debug=True)
