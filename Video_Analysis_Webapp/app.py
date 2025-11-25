import email
from locale import currency
import pickle
import os
from datetime import datetime
from flask import Flask, app, flash, render_template, request, redirect, url_for
import sqlite3

from model.class_video_copilot import VideoToSpeechClass, VideoToObjectsClass, VideoTopicsSummaryClass, VideoPostValidationClass

from verifications.check_policy_compliance import *
from verifications.build_analysis_json import *
from model.constants import *
from commons.advanced_functions import *
from commons.common_functions import *
import random
import string
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from model.classes_tables_sqlalchemy import Company, CompanyMessages, User, HistoryCtbs, UserBrandCredits, UserMessages, UserDeals, Brands, UserGiftCards, TOPIC_CHOICES  # Import the db instance and User model
from model.classes_ai_agents import SpecificAIAgentResponse, create_research_ai_agent_openai, create_research_ai_agent_anthropic, create_csv_ai_agent


# Import the config
from config import app, db
####app.config.from_object(Config) : see later if create a Config class in config.py

TEMP_AUDIO_FILE = "temp_audio.wav" # better to read from config file
# 06-mai TEST
TEMP_AUDIO_FILE = "temp_mono_audio.wav"

# Initialize the database with the app : done in config.py
# Create the database and tables: displaced to the main section

def get_country_from_ipaddress(ip_address):
    # Placeholder function to get country from IP address
    # In a real implementation, use a geolocation service or database
    # Voir les aspects de sécurité et de vie privée liés à l'utilisation des adresses IP
    return "France" # For testing purposes


def generate_random_key(length=15):
    characters = string.ascii_letters + string.digits + string.punctuation
    generated_password = ''.join(random.choice(characters) for _ in range(length))
    return generated_password

""" save the video analysis in the database - SQLAlchemy version """
# Check if feasible to move this function to the User model as a static method
def save_video_analysis_to_db_sqlalchemy(username, video_filename, credit=0, currency='USD', brand_names=[]):
    if video_filename:
        # Save the video analysis result in the database using SQLAlchemy: HistoryCtbs and update User credits
        if username:
            user = User.query.filter_by(username=username).first()
            if user:
                current_date = datetime.now()
                new_entry = HistoryCtbs(user_id=user.id, username=username, contribution_file=video_filename, contribution_date=current_date, credit=credit, currency=currency)

                # Compute total credit for the user after this contribution
                total_credit = db.session.query(db.func.sum(HistoryCtbs.credit)).filter_by(username=username).scalar() or 0
                print(f'Total credit for user {username} before this contribution: {total_credit}')
                total_credit += credit
                print(f'Total credit for user {username} after this contribution: {total_credit}')

                # Update user's total credits and total valid contributions
                user.overall_earned_credits = total_credit
                user.current_credits += credit
                user.total_valid_contributions += 1
                user.currency_credits = currency
                # get current month in format 'YYYY-MM'
                # ATTENTION: this logic assumes that the function is called only once per contribution
                current_month_str = current_date.strftime('%Y-%m')
                if user.current_month == current_month_str:
                    user.current_month_earned_credits += credit
                    user.nb_valid_contributions_this_month += 1
                else:
                    user.current_month = current_month_str
                    user.current_month_earned_credits = credit
                    user.nb_valid_contributions_this_month = 1
                print(f'User {username} current month: {user.current_month}, current month earned credits: {user.current_month_earned_credits}, number of valid contributions this month: {user.nb_valid_contributions_this_month}')

                # Normalize credit w/ brand names number
                if brand_names:
                    credit = int(credit / len(brand_names))

                # Update user's brand credits
                for brand_name in brand_names:
                    brand_credits = UserBrandCredits.query.filter_by(user_id=user.id, brand_name=brand_name).first()
                    if brand_credits:
                        brand_credits.credit += credit
                    else:
                        brand_credits = UserBrandCredits(user_id=user.id, brand_id=1, brand_name=brand_name, credit=credit, currency=currency)
                        db.session.add(brand_credits)

                db.session.add(new_entry)
                db.session.commit()
            else:
                print(f'User {username} not found in the database.')

""" Route for the login page """
@app.route('/user_login', methods=['GET'])
def login_page():
    return render_template('login_page.html') # check if the user exists in the database


""" main url: Home page """
@app.route('/', methods=['GET'])
def home_page():
    return render_template('home_page.html')


""" Route for handling the upload video page post successful Login
Prompt to Claude Sonnet 4: Write the route for upload_data. Strong condition: I need the username value from the login page.
Use decorators to route the specific use cases : verbatims, ads, others
rename upload_data to verbatim_upload_video for clarity 
If the nb of contributions per user per month is exceeded, redirect to a page informing the user about it."""
@app.route('/verbatim_video_upload/', methods=['GET', 'POST'])
def verbatim_video_upload():
    # POST method in login_page.html when clicking on Login button
    print(f'Request method: {request.method}')
    if request.method == 'POST':
        # Get username from login form data
        # Also check for username in URL parameters for GET requests
        username = request.form.get('username') or request.args.get('username')
        print(f'username retrieved from form or args: {username}')
        if username:
            if MAX_CONTRIBUTIONS_PER_USER_PER_MONTH > 0:
                # Check the number of contributions for this user in the current month
                user = User.query.filter_by(username=username).first()
                if user:
                    current_month_str = datetime.now().strftime('%Y-%m')
                    if user.current_month == current_month_str:
                        if user.nb_valid_contributions_this_month >= MAX_CONTRIBUTIONS_PER_USER_PER_MONTH:
                            # Redirect to a page informing the user about exceeding the limit
                            return render_template('annexes/contribution_limit_reached.html', username=username, max_contributions=MAX_CONTRIBUTIONS_PER_USER_PER_MONTH)
            return render_template('verbatims/verbatim_video_upload.html', username=username) # Pass username to the upload page
        else:
            # Redirect back to login if no username provided
            return redirect(url_for('login_page'))
    else:
        # For GET requests, redirect to login
        return redirect(url_for('login_page'))
    

@app.route('/place_video_upload/', methods=['GET', 'POST'])
def place_video_upload():
    # POST method in login_page.html when clicking on Login button
    print(f'Request method: {request.method}')
    if request.method == 'POST':
        # Get username from login form data
        # Also check for username in URL parameters for GET requests
        username = request.form.get('username') or request.args.get('username')
        print(f'username retrieved from form or args: {username}')
        if username:
            return render_template('home_kitchen_outdoor/home_kitchen_outdoor_vid_upload.html', username=username) # Pass username to the upload page
        else:
            # Redirect back to login if no username provided
            return redirect(url_for('login_page'))
    else:
        # For GET requests, redirect to login
        return redirect(url_for('login_page'))
    

""" Select contribution type page route """
@app.route('/select_contribution_type/', methods=['GET', 'POST'])
def select_contribution_type():
    if request.method == 'POST':
        # Get username from login form
        username = request.form.get('username')
        print(f'username in select_contribution_type: {username}') # ok: username value correctly retrieved
        if username:
            return render_template('select_contribution_type.html', username=username) # Pass username to the upload page
        else:
            # Redirect back to login if no username provided
            return redirect(url_for('login_page')) 
        

""" Route to display information about no personal data collection """
@app.route('/no_personal_data', methods=['GET'])
def no_personal_data():
    return render_template('annexes/no_personal_data.html')

""" Route to display information about contributors becoming shareholders """
@app.route('/contributors_shareholders', methods=['GET'])
def contributors_shareholders():
    return render_template('annexes/contributors_shareholders.html')


""" Route to handle form submission for user login - SQLAlchemy version """
@app.route('/login_checking', methods=['POST'])
def login_user():
    username = request.form['username']
    password = request.form['password']

    user = User.query.filter_by(username=username, password=password).first()

    if user:
        # Redirect to the upload video page with username as a parameter
        # Login successful, redirect to select_contribution_type route
        return redirect(url_for('select_contribution_type'), code=307)
    else:
        error_message = "Invalid credentials. Please try again."
        return render_template('login_page.html', error=error_message)


# Register page route
@app.route('/register/', methods=['GET'])
def register_page():
    return render_template('register_page.html')


""" Route to handle form submission for a new user registration - SQLAlchemy version """
@app.route('/register', methods=['POST'])
def register_user():
    # mandatory fields: username, password
    username = request.form['username']
    password = request.form['password']
    # optional fields: email, interests (multiple choice), age_group
    email = request.form.get('email', '')
    interests = request.form.getlist('interests')
    age_group = request.form.get('age_group', '')

    print(f'register_user_alchemy username: {username}')

    # Check if username already exists!
    existing_user = User.query.filter_by(username=username).first()

    if existing_user:
        error_message = "Username already exists. Please choose a different username."
        return render_template('register_page.html', error=error_message)

    # Generate a random 20-character password
    generated_password = generate_random_key(20)
    long_account_recovery_token = generated_password

    print(f'username: {username}, email: {email}, password: {password}, long_account_recovery_token: {long_account_recovery_token}, interests: {interests}, age_group: {age_group}')

    # Get the country from the request IP address
    country = get_country_from_ipaddress(request.remote_addr)
    print(f'Country from IP address: {country}')

    new_user = User(username=username, email=email, password=password, long_account_recovery_token=long_account_recovery_token, interests=', '.join(interests), age_group=age_group, country=country)
    db.session.add(new_user)
    db.session.commit()

    return redirect(url_for('success'))


""" Fill or update the user profile information page route - SQLAlchemy version """
@app.route('/fill_update_your_profile/', methods=['GET'])
def fill_update_your_profile_sqlalchemy():
    # GET method to display the form with pre-filled values if they exist
    username = request.args.get('username')
    print(f'username in fill_update_your_profile_sqlalchemy: {username}')
    # Get existing profile data from the database based on username
    monthly_hours_available = five_favorite_brands = desired_extra_income = why_favorite_brands = five_favorite_product_categories = ''
    if username:
        user = User.query.filter_by(username=username).first()
        if user:
            monthly_hours_available = user.monthly_hours_available if user.monthly_hours_available else ''
            five_favorite_brands = user.five_favorite_brands if user.five_favorite_brands else ''
            desired_extra_income = user.desired_extra_income if user.desired_extra_income else ''
            why_favorite_brands = user.why_favorite_brands if user.why_favorite_brands else ''
            five_favorite_product_categories = user.five_favorite_product_categories if user.five_favorite_product_categories else ''

    print(f'five_favorite_brands: {five_favorite_brands}, desired_extra_income: {desired_extra_income}, why_favorite_brands: {why_favorite_brands}, five_favorite_product_categories: {five_favorite_product_categories}')

    return render_template('annexes/profiling_page.html', username=username, 
                         monthly_hours_available=monthly_hours_available,
                         five_favorite_brands=five_favorite_brands, 
                         desired_extra_income=desired_extra_income,
                         why_favorite_brands=why_favorite_brands,
                         five_favorite_product_categories=five_favorite_product_categories)


""" POST method to handle form submission and update the database - SQLAlchemy version """
@app.route('/submit_profile', methods=['POST'])
def submit_profile_sqlalchemy():
    username = request.form['username']
    five_favorite_brands = request.form['five_favorite_brands']
    monthly_hours_available = request.form['monthly_hours_available']
    desired_extra_income = request.form['desired_extra_income']
    why_favorite_brands = request.form['why_favorite_brands']

    user = User.query.filter_by(username=username).first()
    if user:
        user.five_favorite_brands = five_favorite_brands
        user.monthly_hours_available = monthly_hours_available if monthly_hours_available else 0.0
        user.desired_extra_income = desired_extra_income
        user.why_favorite_brands = why_favorite_brands
        db.session.commit()

    return render_template('select_contribution_type.html', username=username)



# Route to display success message and redirect to login page
@app.route('/success')
def success():
    ##flash('Profile updated successfully!', 'success') # requires a secret key in config.py
    return render_template('login_page.html')

    
""" Route for the user account information page display - SQLAlchemy version """
@app.route('/account_info/', methods=['GET'])
def account_info_sqlalchemy():
    username = request.args.get('username')
    print(f'username in account_info_sqlalchemy: {username}')
    # Get email from database based on username
    email = interests = age_group = None

    # Display the first record of users in the database for debugging purposes
    first_user = User.query.first()
    if first_user:
        print(f'First record: ID={first_user.id}, Username={first_user.username}, Email={first_user.email}, Password={first_user.password}, Interests={first_user.interests}, Age Group={first_user.age_group} ')
    else:
        print('No records found in the database')

    # Fetch user info if username is provided
    if username:
        user = User.query.filter_by(username=username).first()
        if user:
            email = user.email
            interests = user.interests.split(', ') if user.interests else []
            age_group = user.age_group if user.age_group else "Not specified"
        else:
            email = "Email not found"
            interests = []
            age_group = "Not specified"
    else:
        email = "No username provided"
    print(f'email in account_info_sqlalchemy: {email}')
    # Open on click the user contributions history page
    # add a link to user_history page with username as parameter
    return render_template('user_account_infos.html', username=username, email=email, interests=interests, age_group=age_group)


""" Route for the history of contributions page display - SQLAlchemy version """
@app.route('/user_history/', methods=['GET'])
def user_history():
    username = request.args.get('username')
    print(f'username in user_history: {username}')
    # Get user contributions from database based on username
    contributions = []

    if username:
        contributions = HistoryCtbs.query.filter_by(username=username).filter(HistoryCtbs.credit > 0).all()
        print(f'contributions in user_history: {contributions}')

        # Get total credits, total_valid_contributions and currency for the current user
        overall_earned_credits = contributions[0].compute_total_credits() if contributions else 0
        total_valid_contributions = contributions[0].compute_total_valid_contributions() if contributions else 0
        nb_valid_contributions_this_month = contributions[0].compte_nb_valid_contributions_this_month() if contributions else 0
        currency = contributions[0].get_currency() if contributions else 'USD'

    return render_template('user_history.html', username=username, contributions=contributions, total_credits=overall_earned_credits, total_valid_contributions=total_valid_contributions, currency=currency, nb_valid_contributions_this_month=nb_valid_contributions_this_month)

""" Route to display the deals by brands for the user - SQLAlchemy version """
@app.route('/user_deals/', methods=['GET'])
def user_deals():
    username = request.args.get('username')
    print(f'username in user_deals_sqlalchemy: {username}')
    # Get user deals from database based on username
    deals = []

    # The deals provided by the brands will be filtered based on the user's credit balance
    # and which represent a deal of at least 30% discount on the regular price of the product
    if username:
        user = User.query.filter_by(username=username).first()
        if not user:
            return render_template('brands_deals/display_brands_deals.html', username=username, deals=deals)
        deals = UserDeals.query.filter_by(username=username).all()

        # Add dummy data if no deals found for the user
        if not deals:
            deal1 = UserDeals(
                user_id=user.id,
                username=username,
                deal_name='Deal on TVs CYSUIBSQ0125',
                description='Deal of Samsung TVs',
                brand='Samsung',
                category='TV',
                product='TV2025_JK0002154',
                specs='Diag. Size 240 inches, 8K, dolbysourround',
                discount='30%',
                price=850.0,
                currency='USD',
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31)
            )

            deal2 = UserDeals(
                user_id=user.id,
                username=username,
                deal_name='Deal on LG TVs MLSMDSS0312254',
                description='Deal of LG TVs',
                brand='LG',
                category='TV',
                product='LG2024_MP211646',
                specs='Diag. Size 250 inches, 8K, dolbysourround3D',
                discount='40%',
                price=880.0,
                currency='USD',
                start_date=datetime(2024, 2, 1),
                end_date=datetime(2024, 11, 30)
            )
            
            db.session.add(deal1)
            db.session.add(deal2)
            db.session.commit()
            deals = [deal1, deal2]

        for deal in deals:
            print(f'Deal ID: {deal.id}, Username: {deal.username}, Description: {deal.description}, End Date: {deal.end_date}, Brand: {deal.brand}, Category: {deal.category}, Product: {deal.product}, Specs: {deal.specs}, Discount: {deal.discount}, Price: {deal.price}, Currency: {deal.currency}')
        print(f'deals in user_deals: {deals}')

    return render_template('brands_deals/display_brands_deals.html', username=username, deals=deals)


@app.route('/use_credits/', methods=['GET'])
def use_credits():
    username = request.args.get('username')

    # Get user credits from database based on username
    user = User.query.filter_by(username=username).first()
    if user:
        current_credits = user.current_credits or 0
        currency = user.currency_credits or 'USD'
        total_valid_contributions = user.nb_valid_contributions_this_month or 0
        # get user contributions with positive credits
        contributions = HistoryCtbs.query.filter_by(username=username).filter(HistoryCtbs.credit > 0).all()
        total_valid_contributions = len(contributions)
    else:
        current_credits = 0
        total_valid_contributions = 0
        currency = 'USD'

    # Get user brand credits
    brand_credits = UserBrandCredits.query.filter_by(user_id=user.id).filter(UserBrandCredits.credit > 0).all()
    brand_credits_dict = {}
    for credit in brand_credits:
        brand_credits_dict[credit.brand_name] = credit.credit

    print(f'username in use_credits: {username}')
    return render_template('brands_deals/use_credits.html', username=username, contributions=contributions, total_credits=current_credits, total_valid_contributions=total_valid_contributions, currency=currency, brand_credits_dict=brand_credits_dict)


""" Route for accessing Pro/Expert contributions page """
@app.route('/select_expert_contributions/', methods=['GET'])
def select_expert_contributions():
    username = request.args.get('username')
    print(f'username in select_expert_contributions: {username}')
    return render_template('pro_expert/select_expert_contributions.html', username=username)


""" FUNCTIONS FOR MESSAGING SYSTEM """

""" Save a message - SQLAlchemy version """
""" Save in UserMessages the message submitted to SHAIRE from contact_us page or To a Brand from contact_brands page """
def save_message(username, content, recipient='SHAIRE', topic='Other'):
    user = User.query.filter_by(username=username).first()
    if not user:
        raise ValueError("User not found")
    print(f'save_message_sqlalchemy user_id: {user.id}, username: {user.username}, content: {content}, recipient: {recipient}, topic: {topic}')
    message = UserMessages(user_id=user.id, content=content, recipient=recipient, topic=topic)
    db.session.add(message)
    db.session.commit()
    return message.id


""" Route for the messaging system - SQLAlchemy version """
""" save in UserMessages the message submitted to SHAIRE from contact_us page """
@app.route('/messages_to_shaire/', methods=['GET', 'POST'])
def messages_to_shaire():
    if request.method == 'POST':
        username = request.form.get('username') or request.args.get('username')
        content = request.form.get('content')
        print(f'username in messages POST: username {username}, content: {content}')
        if username and content:
            try:
                message = UserMessages.save_message(username, content)
                message.__repr__()
            except Exception as e:
                print(f'Error saving message: {e}')
                return "Error saving message", 500
        return redirect(url_for('messages_to_shaire', username=username))
    else:
        username = request.form.get('username') or request.args.get('username')
        print(f'username in messages GET: {username}')
        user_id = None
        messages = []
        if username:
            user = User.query.filter_by(username=username).first()
            if user:
                user_id = user.id
                username = user.username
                messages = UserMessages.get_user_inbox(username=username, limit=20)
                print(f'Fetched messages for user_id {user_id}, username {username}: {messages}')
        return render_template('select_contribution_type.html', username=username)
    
""" Route to handle sending messages from the contact brands page """
@app.route('/messages/', methods=['POST'])
def send_message_to_brand():
    username = request.form.get('username')
    recipient = request.form.get('brand') # Brand name
    topic = request.form.get('topic')
    content = request.form.get('content')
    print(f'Sending message to brand {recipient} from {username} with topic "{topic}"')

    if username and recipient and content:
        try:
            message = UserMessages.save_message(username, content, recipient=recipient, topic=topic)
            message.__repr__()
        except Exception as e:
            print(f'Error saving message to brand: {e}')
            return "Error saving message", 500

    return redirect(url_for('contact_brands', username=username))


""" ANNEXE PAGES RELATED ROUTES """
@app.route('/about_shaire', methods=['GET'])
def about_shaire():
    """ Need a presentation video of Shaire on this page """
    return render_template('annexes/about_shaire.html')

@app.route('/private_policy', methods=['GET'])
def private_policy():
    return render_template('annexes/private_policy.html')

@app.route('/contact_us', methods=['GET'])
def contact_us():
    username = request.form.get('username') or request.args.get('username')
    # Get the list of messages for the user
    messages = UserMessages.get_user_inbox(username=username, limit=20, recipient='SHAIRE')
    print(f'username in contact_us: {username}')
    return render_template('annexes/contact_us.html', username=username, messages=messages)

@app.route('/faq_and_bot', methods=['GET', 'POST'])
def faq_and_bot():
    if request.method == 'POST':
        question = request.form.get('question')
        print(f'Question submitted: {question}')
        # Here you would add the logic to process the question
        username = request.form.get('username')
        #csv_ai_agent = create_research_ai_agent_openai()
        #csv_ai_agent = create_research_ai_agent_anthropic()
        #csv_ai_agent = create_csv_ai_agent(FAQ_CSV_FILE_PATH)
        csv_ai_agent = SpecificAIAgentResponse(
            specificity="csv-rows-columns",
            ai_model="chatgpt-4o-mini",
            additional_context=COLS_ROWS_CSV_PATH_FILE,
            question=question,
            topic="Cost of industrial gadgets",
            summary="The cost of industrial gadgets varies between $100 and $1000 depending on features.",
            sources=["https://example.com/source1", "https://example.com/source2"],
            tools_used=["web_search", "calculator"]
            )
        answer = csv_ai_agent.ask_csv_research_bot()
        """
        SOLVE THE 429 RATE LIMIT ERROR FROM OPENAI HERE
        openai.RateLimitError: Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.', 'type': 'insufficient_quota', 'param': None, 'code': 'insufficient_quota'}}
        SOLUTION: CREDIT OPENAI ACCOUNT WITH 6 USD AT LEAST
        """
        print(f'Answer generated: {answer}')
    return render_template('annexes/faq_and_bot.html', answer=answer if request.method == 'POST' else None)

@app.route('/specifications', methods=['GET'])
def specifications():
    return render_template('annexes/specifications.html')

@app.route('/contact_brands', methods=['GET'])
def contact_brands():
    username = request.args.get('username')
    messages = UserMessages.get_user_inbox(username=username, limit=100)
    # Check if Brands table is empty and create initial records if needed
    try:
        brands_count = Brands.query.count()
        if brands_count == 0:
            # Create initial brand records if table is empty - FOR TESTING PURPOSES ONLY
            Brands.create_initial_brands()
            initial_brands = Brands.query.all()
            print(f'Created {len(initial_brands)} initial brand records')
    except Exception as e:
        print(f'Error creating initial brands: {e}')
        db.session.rollback()
    print(f'username in contact_brands: {username}')
    return render_template('annexes/contact_brands.html', username=username, brands_list=INITIAL_BRANDS, topics=TOPIC_CHOICES, messages=messages)

@app.route('/back_to_selection_contribution', methods=['GET'])
def back_to_selection_contribution():
    username = request.args.get('username')
    print(f'username in back_to_selection_contribution: {username}')
    return render_template('select_contribution_type.html', username=username)

@app.route('/how_to_earn', methods=['GET'])
def how_to_earn():
    username = request.args.get('username')
    print(f'username in how_to_earn: {username}')
    return render_template('annexes/table_reference_incomes.html', username=username)

@app.route('/gift_card_brand', methods=['GET'])
def gift_card_brand():
    username = request.args.get('username')
    print(f'username in gift_card_brand: {username}')
    # get the User records to find the currency
    currency = User.get_user_currency(username)

    # get user record
    user = User.query.filter_by(username=username).first()
    if not user:
        return "User not found", 404

    # In Generate a Gift Card for a Brand, the brand can be selected from the user's Brands having credits
    brand = request.args.get('brand') # Brand name selected by the user

    # Get the credits for the brand
    brand_credits = UserBrandCredits.query.filter_by(user_id=user.id, brand_name=brand).first()
    if brand_credits:
        print(f'Brand credits for {brand}: {brand_credits.credit} {brand_credits.currency}')
        credits = brand_credits.credit
    else:
        print(f'No brand credits found for {brand}')
        credits = 0

    # Generate QR code for the gift card
    random_key = generate_random_key(15)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = f'Gift Card for {username}\nBrand: {brand}\nValue: 50 {currency}\nCode: {random_key}\nIssued on: {current_time}'
    qr_code_image_path = f'static/qr_codes/{username}_{brand}_{current_time}_gift_card_qr.png'
    qr_code_path = UserGiftCards.generate_qr_code(data, qr_code_image_path)

    # Save the gift card as a PDF
    gift_card_pdf_path = UserGiftCards.save_gift_card_pdf(username, brand, credits, currency, random_key, qr_code_path)

    # Save the gift card record in the database (instantiate UserGiftCards)
    gift_card = UserGiftCards(user_id=user.id, brand_name=brand, total_credits=credits, currency=currency, gift_card_code=random_key, qr_code_path=qr_code_path, pdf_path=gift_card_pdf_path)
    db.session.add(gift_card)
    db.session.commit()
    print(f'Gift card record saved in database for {username}, brand: {brand}, credits: {credits} {currency}')

    # Subtract the gift card value from the user's total credits
    User.subtract_credits_for_gift_card(username, credits, brand)

    # Subtract the gift card value from the user's brand credits
    gift_card_value = credits
    if brand_credits and brand_credits.credit >= gift_card_value:
        brand_credits.credit -= gift_card_value
        db.session.commit()
        print(f'Subtracted {gift_card_value} {currency} from {username} brand credits for {brand}. New balance: {brand_credits.credit} {brand_credits.currency}')
    else:
        print(f'Insufficient brand credits for {username} to generate gift card for {brand}')

    # Display the gift card page
    return render_template('brands_deals/gift_card_brand.html', username=username, brand=brand, credits=credits, currency=currency, random_key=random_key, qr_code=qr_code_path)

@app.route('/gift_cards_history/', methods=['GET'])
def gift_cards_history():
    username = request.args.get('username')
    print(f'username in gift_cards_history: {username}')
    # Get user gift cards from database based on username
    gift_cards = []

    if username:
        gift_cards = UserGiftCards.query.filter_by(username=username).all()
        print(f'gift_cards in gift_cards_history: {gift_cards}')

    return render_template('brands_deals/gift_cards_history.html', username=username, gift_cards=gift_cards)

@app.route('/light_contributions_tuto', methods=['GET'])
def light_contributions_tuto():
    username = request.args.get('username')
    return render_template('annexes/light_contributions_tuto.html', username=username)

@app.route('/company_contact', methods=['GET'])
def company_contact():
    return render_template('annexes/company_contact.html')

@app.route('/company_meeting_schedule', methods=['GET'])
def company_meeting_schedule():
    return render_template('annexes/company_meeting_schedule.html')

@app.route('/company_email_us', methods=['GET'])
def company_email_us():
    return render_template('annexes/company_email_us.html')

@app.route('/update_company_contact_email', methods=['POST'])
def update_company_contact_email():
    company_name = request.form.get('company_name')
    contact_email = request.form.get('contact_email')
    print(f'New company contact email submitted for {company_name}: {contact_email}')
    # Update the AppointmentsWithCompanies table with the new contact email
    Company.update_company_contact_email(company_name, contact_email)

    # Here you would add the logic to update the company contact email
    return render_template('annexes/company_email_us.html', success_message="Contact email updated successfully.")

@app.route('/company_messages_to_shaire', methods=['POST'])
def company_messages_to_shaire():
    company_name = request.form.get('company_name')
    content = request.form.get('content')
    print(f'New message submitted from {company_name}: {content}')
    # Save the message to the database or perform other actions
    CompanyMessages.save_company_message(company_name, content)
    return render_template('annexes/company_email_us.html', success_message="Message sent successfully.")


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
        return redirect(url_for('verbatim_video_upload'))

    print(f'Processing video: {video_file.filename} for user: {username}')
    
    # Save video file
    video_path = DIRECTORY_VIDEOS + video_file.filename
    video_file.save(video_path)
    
    try:
        # Extract and process speech
        analysis_results = process_video_speech(video_path)

        # Extract Brand Names from video text
        brand_names = extract_brand_names_from_video_text(analysis_results['text_video'])
        print(f'Extracted brand names from video text: {brand_names}')

        # Perform text analysis if speech extraction was successful
        if analysis_results['text_video']:
            perform_text_analysis(analysis_results, video_file.filename)
            check_compliance(analysis_results, video_file.filename)
            validate_speech(analysis_results, video_path)
        
        # Save to database
        compliance_metrics = analysis_results.get('compliance_dict', {}).get('compliance_metrics', COMPLIANCE_METRIC_DEFAULT)
        compliance_result = analysis_results.get('compliance_dict', {}).get('result', COMPLIANCE_RESULT_DEFAULT)
        payment = analysis_results.get('compliance_dict', {}).get('payment', VALIDATION_PAYMENT_DEFAULT)
        currency = analysis_results.get('compliance_dict', {}).get('currency', VALIDATION_CURRENCY_DEFAULT)
        save_video_analysis_to_db_sqlalchemy(username, video_file.filename, payment, currency, brand_names)

        # Calculate equity share based on payment
        equity_share = User.compute_equity_share(username)
        
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
        # NB: at this stage, compliance_dict contains compliance_metrics, result, payment, currency (3 int/float values and 1 string value)
        return render_template('display_video_validation.html',
                            username=username,
                            video_path=video_path,
                            compliance_metrics=compliance_metrics,
                            compliance_result=compliance_result,
                            payment=payment,
                            equity_share=equity_share,
                            compliance_dict=analysis_results.get('compliance_dict', {}))
    
    except Exception as e:
        print(f'Error processing video: {e}')
        return redirect(url_for('verbatim_video_upload'))



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
    # Initialize the database : done in config.py
    with app.app_context():
        db.create_all()  # Create database tables for all models
    # Run the Flask app
    app.run(
        host='127.0.0.1',
        port=8000,
        debug=True)
