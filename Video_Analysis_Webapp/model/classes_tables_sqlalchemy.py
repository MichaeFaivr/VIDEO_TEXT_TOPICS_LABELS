from flask_sqlalchemy import SQLAlchemy

# Initialize SQLAlchemy without binding it to the app yet
db = SQLAlchemy()

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

    def __repr__(self):
        return f'<User {self.username}>'
    

""" Store video analysis results including labels, text, topics, sentiment, and additional info """
class Video_analysis(db.Model):
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
class History_ctbs(db.Model):
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
class User_deals(db.Model):
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


""" User messages and notifications information """
""" Need to managge read/unread status, timestamps, and message content """
class User_messages(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    sent_at = db.Column(db.DateTime, server_default=db.func.now())
    is_read = db.Column(db.Boolean, default=False)

    user = db.relationship('User', backref=db.backref('messages', lazy=True))

    def __repr__(self):
        return f'<User_messages to User ID {self.user_id} at {self.sent_at}>'