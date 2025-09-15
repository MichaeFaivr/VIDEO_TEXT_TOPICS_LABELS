from locale import currency
import pickle
import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
import sqlite3

from model.class_video_copilot import VideoToSpeechClass, VideoToObjectsClass, VideoTopicsSummaryClass, VideoPostValidationClass

from verifications.check_policy_compliance import *
from verifications.build_analysis_json import *
from model.constants import *
from commons.advanced_functions import *

app = Flask(__name__)

TEMP_AUDIO_FILE = "temp_audio.wav" # better to read from config file
# 06-mai TEST
TEMP_AUDIO_FILE = "temp_mono_audio.wav"

PATH_DATABASE_USERS = 'databases/user_accounts/users5.db'

# Initialize the database
def init_db():
    # connect to the SQLite database (it will be created if it doesn't exist)
    # in databases/user_accounts/users.db
    # CREATE USERS TABLE
    conn = sqlite3.connect(PATH_DATABASE_USERS)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL,
            password TEXT NOT NULL,
            interests TEXT
        )           
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS history_ctbs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            file TEXT NOT NULL,
            date DATE,
            credit INTEGER,
            currency TEXT
        )
    ''')
    conn.commit()
    conn.close()


# save the database
def save_video_analysis_to_db(username, video_filename, credit=0, currency='USD'):
    if video_filename:
        # Save the video analysis result in the database
        if username:
            conn = sqlite3.connect(PATH_DATABASE_USERS)
            c = conn.cursor()
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'username: {username}, video filename: {video_filename}, current_date: {current_date}')
            c.execute('INSERT INTO history_ctbs (username, file, date, credit, currency) VALUES (?, ?, ?, ?, ?)',
                        (username, video_filename, current_date, credit, currency))
            conn.commit()
            conn.close()

# Route for handling the upload video page post successful Login
# Prompt to Claude4: Write the route for upload_data. Strong condition: I need the username value from the login page.
@app.route('/upload_data/', methods=['GET', 'POST'])
def upload_data():
    # POST method in login_page.html when clicking on Login button
    if request.method == 'POST':
        # Get username from login form
        username = request.form.get('username')
        print(f'username in upload_data: {username}') # ok: username value correctly retrieved
        if username:
            return render_template('video_upload.html', username=username) # Pass username to the upload page
        else:
            # Redirect back to login if no username provided
            return redirect(url_for('login_page'))
    else:
        # For GET requests, redirect to login
        return redirect(url_for('login_page'))


@app.route('/', methods=['GET'])
def login_page():
    return render_template('login_page.html') # check if the user exists in the database


# Register page route
@app.route('/register/', methods=['GET'])
def register_page():
    return render_template('simple_registration_form.html')
    #return render_template('register_page.html') # voir bug lié à username


# Route to handle form submission for a new user registration
# Fill the user infos in the users table 
@app.route('/register', methods=['POST'])
def register_user():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    interests = request.form.getlist('interests')

    conn = sqlite3.connect(PATH_DATABASE_USERS)
    c = conn.cursor()
    c.execute('INSERT INTO users (username, email, password, interests) VALUES (?, ?, ?, ?)',
              (username, email, password, ', '.join(interests)))
    conn.commit()
    conn.close()

    return redirect(url_for('success'))


# Route to display success message and redirect to login page
@app.route('/success')
def success():
    ##return "User registered successfully!"
    return render_template('login_page.html')


# Route for the user account information page display  
@app.route('/account_info/', methods=['GET'])
def account_info():
    username = request.args.get('username')
    print(f'username in account_info: {username}')
    # Get email from database based on username
    email = None
    interests = None

    # Display the first record of users in the database for debugging purposes
    conn = sqlite3.connect(PATH_DATABASE_USERS)
    c = conn.cursor()
    # First user record
    c.execute('SELECT * FROM users LIMIT 1')
    first_record = c.fetchone()
    conn.close()
    if first_record:
        print(f'First record: ID={first_record[0]}, Username={first_record[1]}, Email={first_record[2]}, Password={first_record[3]}, Interests={first_record[4]}')
    else:
        print('No records found in the database')

    # Fetch user info if username is provided
    if username:
        conn = sqlite3.connect(PATH_DATABASE_USERS)
        c = conn.cursor()
        c.execute('SELECT email, interests FROM users WHERE username = ?', (username,))
        result = c.fetchone()
        conn.close()
        
        if result:
            email = result[0]
            interests = result[1].split(', ') if result[1] else []
        else:
            email = "Email not found"
            interests = []
    else:
        email = "No username provided"
    print(f'email in account_info: {email}')
    # Open on click the user contributions history page
    # add a link to user_history page with username as parameter
    return render_template('user_account_infos.html', username=username, email=email, interests=interests)
                           # history_link=url_for('user_history', username=username))

# Route for the history of contributions page display  
@app.route('/user_history/', methods=['GET'])
def user_history():
    username = request.args.get('username')
    print(f'username in user_history: {username}')
    # Get user contributions from database based on username
    contributions = []

    if username:
        conn = sqlite3.connect(PATH_DATABASE_USERS)
        c = conn.cursor()
        c.execute('SELECT * FROM history_ctbs WHERE username = ?', (username,))
        contributions = c.fetchall()
        print(f'contributions in user_history: {contributions}')
        conn.close()

    return render_template('user_history.html', username=username, contributions=contributions)
    

""" VIDEO VERBATIM ANALYSIS ROUTE """
""" Keep the video analysis route, but do not display the AI analysis results. Display the result of the validation only. """
@app.route('/analysis/', methods=['POST'])
def result():
    if request.method != 'POST':
        return redirect(url_for('login_page'))
    
    # Get video file and username
    video_file = request.files.get('video')
    username = request.form.get('username')
    
    if not video_file or not video_file.filename:
        return redirect(url_for('upload_data'))
    
    print(f'Processing video: {video_file.filename} for user: {username}')
    
    # Save video file
    video_path = DIRECTORY_VIDEOS + video_file.filename
    video_file.save(video_path)
    
    try:
        # Extract and process speech
        analysis_results = process_video_speech(video_path)
        
        # Perform text analysis if speech extraction was successful
        if analysis_results['text_video']:
            perform_text_analysis(analysis_results, video_file.filename)
            check_compliance(analysis_results, video_file.filename)
            validate_speech(analysis_results, video_path)
        
        # Save to database
        compliance_metrics = analysis_results.get('compliance_dict', {}).get('compliance_metrics', {})
        compliance_result = analysis_results.get('compliance_dict', {}).get('result', 0)
        payment = analysis_results.get('compliance_dict', {}).get('payment', 0)
        currency = analysis_results.get('compliance_dict', {}).get('currency', 'USD')
        save_video_analysis_to_db(username, video_file.filename, payment, currency)
        
        # Render results page
        """
        return render_template('video_analysis.html', 
                             video_path=video_path,
                             extracted_text=analysis_results['text_video'],
                             summary_text=analysis_results['summary_text'],
                             ner_text=analysis_results['text_NER'],
                             ner_surrounding_text=analysis_results['text_NER_surrounding'])
        """

        """ Display only the validation results page """
        return render_template('display_video_validation.html',
                            username=username,
                            video_path=video_path,
                            compliance_metrics=compliance_metrics,
                            compliance_result=compliance_result,
                            payment=payment,
                            currency=currency,
                            compliance_dict=analysis_results.get('compliance_dict', {}))
    
    except Exception as e:
        print(f'Error processing video: {e}')
        return redirect(url_for('upload_data'))



@app.route('/objectdetection/', methods=['POST'])
def display_image():
    if request.method == 'POST':
        video_file = request.form['video_path'] ##voir comment récupérer video file
        # Need to apply the image operations from the video onto one or multiple frames randomly extracted from the video
        liste_objets = LISTE_OBJETS
        print('video_file:', video_file) # empty

        videoToObjects = VideoToObjectsClass(video_file, list_object_types=liste_objets)
        videoToObjects.get_video_frame_size()

        # call constantes.py NB_SNAPSHOTS_VIDEO_ANALYSIS

        for _ in range(0, NB_SNAPSHOTS_VIDEO_ANALYSIS):
            # ===========================
            # OPERATION.1: Extract objects from the video
            # ===========================
            # Save an image from the video with the detected objects framed in red boxes
            videoToObjects.save_frame_with_detections() # with current date and time in file name

            # ===========================
            # OPERATION.2: Extract texts from the video
            # ===========================
            # Recognize text in the frame
            object_labeling_method = OBJECT_LABELLING_METHOD # "easyocr"
            videoToObjects.recognize_text_in_frame(object_labeling_method)

            # ===========================
            # OPERATION.3: Detect faces in the frame and assess the age and the gender
            # ===========================
            videoToObjects.estimate_gender_age_from_faces()

        return render_template('video_objects_detection.html', frame=videoToObjects.output_path)


if __name__ == '__main__':
    app.debug = True
    # Initialize the database
    init_db()
    app.run(
        host='127.0.0.1',
        port=8000,
        debug=True)
