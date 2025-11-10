from flask_sqlalchemy import SQLAlchemy

# Initialize SQLAlchemy without binding it to the app yet
from config import db
#db = SQLAlchemy()

TOPIC_CHOICES = ['Ask for a deal on a product', 'Question on a product', 'Proposal of contribution', 'Other']

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), nullable=True)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    long_account_recovery_token = db.Column(db.String(200), nullable=True)
    long_account_recovery_token_expiration = db.Column(db.DateTime, nullable=True)
    short_account_recovery_token = db.Column(db.String(10), nullable=True)
    interests = db.Column(db.String(500), nullable=True)  # Comma-separated list of interests
    is_admin = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    is_premium = db.Column(db.Boolean, default=False)
    premium_expiration = db.Column(db.DateTime, nullable=True)
    api_key = db.Column(db.String(200), unique=True, nullable=True)
    api_key_expiration = db.Column(db.DateTime, nullable=True)
    monthly_hours_available = db.Column(db.Float, default=0.0)
    desired_extra_income = db.Column(db.String(50), nullable=True)
    five_favorite_product_categories = db.Column(db.String(500), nullable=True)  # Comma-separated list
    why_favorite_brands = db.Column(db.String(1000), nullable=True)
    five_favorite_brands = db.Column(db.String(500), nullable=True)  # Comma-separated list
    five_favorite_products = db.Column(db.String(500), nullable=True)  # Comma-separated list
    country = db.Column(db.String(100), nullable=True)
    age_group = db.Column(db.String(100), nullable=True)
    gender = db.Column(db.String(50), nullable=True)
    total_credits = db.Column(db.Integer, default=0)
    currency_credits = db.Column(db.String(10), default='USD')
    total_valid_contributions = db.Column(db.Integer, default=0)
    total_shares = db.Column(db.Integer, default=0)

    def __repr__(self):
        return f'<User {self.username}>'
    
    def to_json(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at,
            'is_admin': self.is_admin,
            'is_active': self.is_active,
            'is_premium': self.is_premium,
            'premium_expiration': self.premium_expiration,
            'monthly_hours_available': self.monthly_hours_available,
            'country': self.country,
            'age_group': self.age_group
        }

""" Store video analysis results including labels, text, topics, sentiment, and additional info """
class VideoAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    video_url = db.Column(db.String(500), nullable=False)
    analysis_date = db.Column(db.DateTime, server_default=db.func.now())
    labels = db.Column(db.Text, nullable=True)  # JSON string of labels
    text_content = db.Column(db.Text, nullable=True)  # Extracted text content
    topics = db.Column(db.Text, nullable=True)  # JSON string of topics
    sentiment = db.Column(db.String(50), nullable=True)  # e.g., Positive, Negative, Neutral
    additional_info = db.Column(db.Text, nullable=True)  # Any additional info as JSON string

    user = db.relationship('User', backref=db.backref('video_analyses', lazy=True))

    def __repr__(self):
        return f'<Video_analysis {self.video_url} for User ID {self.user_id}>'
    

""" History of contributions, credits, and actions performed by users """
class HistoryCtbs(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    username = db.Column(db.String(80), nullable=False)
    contribution_file = db.Column(db.String(200), nullable=False)
    contribution_date = db.Column(db.DateTime, server_default=db.func.now())
    contribution_type = db.Column(db.String(100), nullable=True)  # e.g., 'upload', 'edit'
    credit = db.Column(db.Integer, nullable=False)
    currency = db.Column(db.String(10), nullable=False, default='USD')
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    details = db.Column(db.Text, nullable=True)  # Additional details as JSON string

    user = db.relationship('User', backref=db.backref('history_ctbs', lazy=True))

    def __repr__(self):
        return f'<History_ctbs {self.contribution_file} for User ID {self.user_id}>'    


""" User deals and multiple brands deals and promotions information """
class UserDeals(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    username = db.Column(db.String(80), nullable=False)
    deal_name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    product = db.Column(db.String(200), nullable=True)
    brand = db.Column(db.String(100), nullable=True)
    category = db.Column(db.String(100), nullable=True)
    specs = db.Column(db.String(200), nullable=True)
    discount = db.Column(db.String(50), nullable=True)  # e.g., "20% off"
    price = db.Column(db.Float, nullable=True)
    currency = db.Column(db.String(10), nullable=False, default='USD')
    start_date = db.Column(db.DateTime, nullable=False)
    end_date = db.Column(db.DateTime, nullable=False)
    is_active = db.Column(db.Boolean, default=True)

    user = db.relationship('User', backref=db.backref('deals', lazy=True))

    def __repr__(self):
        return f'<User_deals {self.deal_name} for User ID {self.user_id}>'


""" User messages and notifications information to/from Shaire - not emails to brands """
""" Need to manage read/unread status, timestamps, and message content """
class UserMessages(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    recipient = db.Column(db.String(100), default='SHAIRE')
    topic = db.Column(db.Enum(*TOPIC_CHOICES, name='message_topics'), nullable=False, default='Other')
    is_read = db.Column(db.Boolean, default=False)
    message_type = db.Column(db.String(50), nullable=True)  # e.g., 'notification', 'alert'
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    answer = db.Column(db.Text, nullable=True)  # Response to the message, if any

    user = db.relationship('User', backref=db.backref('messages', lazy=True))

    def __repr__(self):
        return f'<User_messages to User ID {self.user_id} at {self.timestamp}>'
    

class Brands(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    brand_name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)
    website = db.Column(db.String(200), nullable=True)
    contact_email = db.Column(db.String(120), nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

    def __repr__(self):
        return f'<Brands {self.brand_name}>'
    

class UserGiftCards(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    RSA_Key = db.Column(db.String(500), nullable=False)
    QR_code = db.Column(db.String(500), nullable=False)
    total_credits = db.Column(db.Integer, nullable=False)
    currency = db.Column(db.String(10), nullable=False, default='USD')
    brand = db.Column(db.String(500), nullable=True) # Associated brand for the gift card
    issued_date = db.Column(db.DateTime, server_default=db.func.now())
    expiration_date = db.Column(db.DateTime, nullable=True)

    user = db.relationship('User', backref=db.backref('gift_cards', lazy=True))

    def __repr__(self):
        return f'<GiftCards for User ID {self.user_id} with {self.total_credits} {self.currency}>'

class UserApiUsageLogs(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    endpoint = db.Column(db.String(200), nullable=False)
    request_date = db.Column(db.DateTime, server_default=db.func.now())
    response_status = db.Column(db.Integer, nullable=False)
    usage_details = db.Column(db.Text, nullable=True)  # Additional details as JSON string

    user = db.relationship('User', backref=db.backref('api_usage_logs', lazy=True))

    def __repr__(self):
        return f'<ApiUsageLogs {self.endpoint} for User ID {self.user_id}>'
    
class UserCreditsInvestmentPlans(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    plan_name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)
    price = db.Column(db.Float, nullable=False)
    currency = db.Column(db.String(10), nullable=False, default='USD')
    duration_days = db.Column(db.Integer, nullable=False)  # Duration of the plan in days
    features = db.Column(db.Text, nullable=True)  # JSON string of features

    def __repr__(self):
        return f'<SubscriptionPlans {self.plan_name}>'
    

class UserCoupons(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    code = db.Column(db.String(50), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)
    discount_percentage = db.Column(db.Float, nullable=False)
    valid_from = db.Column(db.DateTime, nullable=False)
    valid_to = db.Column(db.DateTime, nullable=False)
    is_active = db.Column(db.Boolean, default=True)

    def __repr__(self):
        return f'<Coupons {self.code} - {self.discount_percentage}%>'
    

class ContributionCredits(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    contribution_type = db.Column(db.String(100), nullable=False)  # e.g., 'video_analysis', 'data_upload'
    country = db.Column(db.String(100), nullable=True)
    currency = db.Column(db.String(10), nullable=False, default='USD')
    verbatim_credit_range = db.Column(db.String(50), nullable=True)  # e.g., "1-10"
    place_credit_range = db.Column(db.String(50), nullable=True)  # e.g., "5-20"
    wardrobe_credit_range = db.Column(db.String(50), nullable=True)  # e.g., "10-30"
    dance_credit_range = db.Column(db.String(50), nullable=True)  # e.g., "15-40"
    comparison_credit_range = db.Column(db.String(50), nullable=True)  # e.g., "20-50"
    credits_earned = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<ContributionCredits {self.contribution_type} - {self.credits_earned} {self.currency}>'
