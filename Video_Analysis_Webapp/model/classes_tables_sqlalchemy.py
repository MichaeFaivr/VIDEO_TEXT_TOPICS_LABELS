from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
import qrcode
from io import BytesIO
from sqlalchemy.orm import validates
from model.constants import INITIAL_BRANDS

# Initialize SQLAlchemy without binding it to the app yet
from config import db
#db = SQLAlchemy()

TOPIC_CHOICES = ['Ask for a deal on a product', 'Question on a product', 'Proposal of contribution', 'Other']

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), nullable=True)
    password = db.Column(db.String(200), nullable=False)
    temporary_password = db.Column(db.String(200), nullable=True)
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
    current_credits = db.Column(db.Integer, default=0)
    currency_credits = db.Column(db.String(10), default='USD')
    current_month_earned_credits = db.Column(db.Integer, default=0)
    current_month = db.Column(db.String(20), nullable=True) # e.g., '2024-06'
    overall_earned_credits = db.Column(db.Integer, default=0)
    total_valid_contributions = db.Column(db.Integer, default=0)
    nb_valid_contributions_this_month = db.Column(db.Integer, default=0)
    total_shares = db.Column(db.Integer, default=0)
    equity_share = db.Column(db.Float, default=0.0)  # Equity share as percentage

    def __repr__(self):
        return f'<User {self.username}>'
    
    def print_equity_share(self):
        return f'User {self.username} has an equity share of {self.equity_share:.2f}%'
    
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
    
    @classmethod
    def register(cls, username, email, password, long_account_recovery_token=None, interests=None, age_group=None, country=None):
        new_user = cls(
            username=username,
            email=email,
            password=password,
            long_account_recovery_token=long_account_recovery_token,
            interests=interests,
            age_group=age_group,
            country=country
        )
        db.session.add(new_user)
        db.session.commit()
        return new_user

    @classmethod
    def get_user_id(cls, username):
        user = cls.query.filter_by(username=username).first()
        if user:
            return user.id
        return None
    
    @classmethod
    def get_user_current_credits(cls, username):
        user = cls.query.filter_by(username=username).first()
        if user:
            return user.current_credits
        return None
    
    @classmethod
    def get_user_currency(cls, username):
        user = cls.query.filter_by(username=username).first()
        if user:
            return user.currency_credits or 'USD'
        return 'USD'
    
    @classmethod
    def subtract_credits_for_gift_card(cls, username, credits, brand):
        user = cls.query.filter_by(username=username).first()
        if user:
            if user.current_credits < credits:
                print(f'Insufficient total credits for {username} to generate gift card for {brand}')
                return False
            user.current_credits -= credits
            db.session.commit()
            print(f'Subtracted {credits} {user.currency_credits} from {username} total credits. New balance: {user.current_credits} {user.currency_credits}')
            return True
        return False
    
    @classmethod
    def add_credits(cls, username, credits):
        user = cls.query.filter_by(username=username).first()
        if user:
            user.current_credits += credits
            db.session.commit()
            print(f'Added {credits} {user.currency_credits} to {username} total credits. New balance: {user.current_credits} {user.currency_credits}')
            return True
        return False
    
    # Compute equity share based on total shares and overall company shares
    # as the ratio of user's total_credits to overall total users' total_credits
    # multiplied by 10% 
    # e.g., if user has 1000 credits and total credits of all users is 1,000,000
    # then equity share = (1000 / 1000000) * 10% = 0.01%
    # ATTENTION: Anytime a user's current_credits change, this method should be called to update equity_share !
    @classmethod
    def compute_equity_share(cls, username):
        user = cls.query.filter_by(username=username).first()
        if not user:
            return 0.0
        total_credits_all_users = db.session.query(db.func.sum(cls.current_credits)).scalar() or 0
        print(f'Total credits of all users: {total_credits_all_users} and user {username} has {user.current_credits} credits.')
        if total_credits_all_users == 0:
            return 0.0
        user.equity_share = (user.current_credits / total_credits_all_users) * 10.0  # 10% of total equity
        db.session.commit()
        return user.equity_share
    

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

    def compute_total_credits(self):
        # Example method to compute total credits from history
        total_credits = sum(entry.credit for entry in self.user.history_ctbs)
        return total_credits 

    def get_user_valid_contributions(self):
        # Example method to get valid contributions
        valid_contributions = [entry for entry in self.user.history_ctbs if entry.credit > 0]
        return valid_contributions

    def compute_total_user_credits(self):
        total_credits = sum(entry.credit for entry in self.user.history_ctbs)
        return total_credits

    def compute_total_valid_contributions(self):
        total_valid = sum(1 for entry in self.user.history_ctbs if entry.credit > 0)
        return total_valid

    def compte_nb_valid_contributions_this_month(self):
        from datetime import datetime
        current_month_str = datetime.now().strftime('%Y-%m')
        nb_valid = sum(1 for entry in self.user.history_ctbs if entry.credit > 0 and entry.contribution_date.strftime('%Y-%m') == current_month_str)
        return nb_valid

    def get_currency(self):
        if self.user.history_ctbs:
            return self.user.history_ctbs[0].currency
        return 'USD' 
    

""" Credits by User by Brands for contributions related to specific brands """
class UserBrandCredits(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    brand_id = db.Column(db.Integer, db.ForeignKey('brands.id'), nullable=False)
    brand_name = db.Column(db.String(100), nullable=False)
    contribution_type = db.Column(db.String(100), nullable=True)  # e.g., 'upload', 'edit'
    credit = db.Column(db.Integer, nullable=False)
    currency = db.Column(db.String(10), nullable=False, default='USD')
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    details = db.Column(db.Text, nullable=True)  # Additional details as JSON string

    user = db.relationship('User', backref=db.backref('brand_credits', lazy=True))
    brand = db.relationship('Brands', backref=db.backref('user_credits', lazy=True))

    def __repr__(self):
        return f'<UserBrandCredits User ID {self.user_id} Brand ID {self.brand_id} - {self.credit} {self.currency}>'
    

    def get_user_brand_credits(self, user_id, brand_name):
        user = User.query.filter_by(id=user_id).first()
        if not user:
            return 0
        user_brand_credits = db.session.query(db.func.sum(self.credit)).filter_by(user_id=user.id, brand_name=brand_name).scalar()
        return user_brand_credits or 0

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
    
    def is_currently_active(self):
        from datetime import datetime
        now = datetime.utcnow()
        return self.is_active and self.start_date <= now <= self.end_date
    
    def duration_days(self):
        return (self.end_date - self.start_date).days
    
    def discount_value(self):
        if self.discount and '%' in self.discount:
            try:
                return float(self.discount.replace('% off', '').strip())
            except ValueError:
                return None
        return None
    
    def final_price(self):
        discount_val = self.discount_value()
        if discount_val is not None and self.price is not None:
            return self.price * (1 - discount_val / 100)
        return self.price
    



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
    
    @classmethod
    def get_user_inbox(cls, username, limit, recipient=None):
        user = User.query.filter_by(username=username).first()
        if not user:
            return []
        user_id = user.id
        query = cls.query.filter_by(user_id=user_id)
        if query is None:
            return []
        if recipient:
            query = query.filter_by(recipient=recipient)
        return query.order_by(cls.timestamp.desc()).limit(limit).all()
    
    @classmethod
    def save_message(cls, username, content, recipient='SHAIRE', topic='Other', message_type=None):
        user = User.query.filter_by(username=username).first()
        if not user:
            return None
        new_message = cls(
            user_id=user.id,
            content=content,
            topic=topic,
            recipient=recipient,
            message_type=message_type
        )
        db.session.add(new_message)
        db.session.commit()
        return new_message
    
    @classmethod
    def mark_as_read(cls, message_id):
        message = cls.query.get(message_id)
        if message:
            message.is_read = True
            db.session.commit()
            return True
        return False
    

class Brands(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    brand_name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)
    website = db.Column(db.String(200), nullable=True)
    contact_email = db.Column(db.String(120), nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

    def __repr__(self):
        return f'<Brands {self.brand_name}>'
    
    @classmethod
    def create_initial_brands(cls):
        initial_brands = INITIAL_BRANDS
        for brand_data in initial_brands:
            existing_brand = cls.query.filter_by(brand_name=brand_data['brand_name']).first()
            if not existing_brand:
                new_brand = cls(
                    brand_name=brand_data['brand_name'],
                    description=brand_data.get('description'),
                    website=brand_data.get('website'),
                    contact_email=brand_data.get('contact_email')
                )
                db.session.add(new_brand)
        db.session.commit()
    

class UserGiftCards(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    gift_card_code = db.Column(db.String(500), nullable=False)
    total_credits = db.Column(db.Integer, nullable=False, default=0)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.total_credits > 100:
            self.total_credits = 100
    
    @db.validates('total_credits')
    def validate_total_credits(self, key, value):
        if value > 100:
            return 100
        return value
    currency = db.Column(db.String(10), nullable=False, default='USD')
    brand_name = db.Column(db.String(500), nullable=True) # Associated brand for the gift card
    issue_date = db.Column(db.DateTime, server_default=db.func.now())
    #expiration_date = db.Column(db.DateTime, nullable=False, default=lambda: db.func.now() + db.text("INTERVAL 1 YEAR"))
    qr_code_path = db.Column(db.String(500), nullable=True)  # Path to stored QR code image
    pdf_path = db.Column(db.String(500), nullable=True)  # Path to stored PDF file

    user = db.relationship('User', backref=db.backref('gift_cards', lazy=True))

    def __repr__(self):
        return f'<GiftCards for User ID {self.user_id} with {self.total_credits} {self.currency}>'
    
    @classmethod
    def get_user_gift_cards(cls, username):
        user = User.query.filter_by(username=username).first()
        if not user:
            return []
        return cls.query.filter_by(user_id=user.id).all()
    
    @classmethod
    def get_gift_card_by_code(cls, gift_card_code):
        return cls.query.filter_by(gift_card_code=gift_card_code).first()
    
    @classmethod
    def generate_unique_gift_card_code(cls):
        import uuid
        while True:
            code = str(uuid.uuid4()).replace('-', '').upper()[:12]  # 12-character unique code
            existing_code = cls.query.filter_by(gift_card_code=code).first()
            if not existing_code:
                return code
            

    @classmethod
    def generate_qr_code(cls, data: str, qr_code_image_path: None):
        """
        Inputs:
        - data: str, the data to encode in the QR code
        - qr_code_path: str, the path to save the generated QR code image
        Outputs:
        - qr_code_path: str, the path to the generated QR code image
        """
        if qr_code_image_path is not None:
            qr_code_path = qr_code_image_path
        else:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            qr_code_path = f'static/qr_codes/{data}_{current_time}_gift_card_qr_code.png'

        # Create QR code instance
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        # Add data to the QR code
        qr.add_data(data)
        qr.make(fit=True)

        # Generate the QR code image
        img = qr.make_image(fill_color="black", back_color="white")

        # Save the image
        img.save(qr_code_path)
        return qr_code_path
    
    @classmethod
    def generate_qr_code_in_memory(cls, gift_card_code: str):
        """ Generate a QR code image in memory for the given gift card code """
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(gift_card_code)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        byte_io = BytesIO()
        img.save(byte_io, format='PNG')
        byte_io.seek(0)
        return byte_io  # Return BytesIO object containing the QR code image
    
    @classmethod
    def save_gift_card_pdf(cls, username: str, brand: str, credits: int, currency: str, gift_card_code: str, qr_code_path: str, pdf_path: str = None):
        """
        Inputs:
        - username: str, the username of the user
        - brand: str, the brand of the gift card
        - credits: int, the value of the gift card
        - currency: str, the currency of the gift card
        - gift_card_code: str, the unique code of the gift card
        - qr_code_path: str, the path to the QR code image
        - pdf_path: str, the path to save the generated PDF file
        Outputs:
        - pdf_path: str, the path to the generated PDF file
        """
        from fpdf import FPDF

        if pdf_path is None:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            pdf_path = f'static/pdfs/{username}_{brand}_{current_time}_gift_card.pdf'

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Gift Card", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Username: {username}", ln=True)
        pdf.cell(200, 10, txt=f"Brand: {brand}", ln=True)
        pdf.cell(200, 10, txt=f"Credits: {credits} {currency}", ln=True)
        pdf.cell(200, 10, txt=f"Gift Card Code: {gift_card_code}", ln=True)

        # Add QR code image
        pdf.image(qr_code_path, x=80, y=60, w=50, h=50)

        pdf.output(pdf_path)
        return pdf_path
    

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
    
""" Contribution credits configuration for different types of contributions """
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



class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_name = db.Column(db.String(200), unique=True, nullable=False)
    company_email = db.Column(db.String(120), nullable=True)
    company_phone = db.Column(db.String(50), nullable=True)
    country = db.Column(db.String(100), nullable=True)
    website = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    password_hash = db.Column(db.String(128), nullable=False)

    def __repr__(self):
        return f'<Company {self.company_name}>'
    
    @classmethod
    def register(cls, company_name, company_email, country, password):
        # Add logic to register a new company, e.g., hashing the password, saving to the database
        new_company = cls(company_name=company_name, company_email=company_email, country=country, password_hash=password)
        db.session.add(new_company)
        db.session.commit()
        return new_company
    
    @classmethod
    def update_company_contact_email(cls, company_name, new_email):
        company = cls.query.filter_by(company_name=company_name).first()
        if company:
            company.company_email = new_email
            db.session.commit()
            return True
        return False
    

class CompanyMessages(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_name = db.Column(db.String(200), db.ForeignKey('company.company_name'), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    sender = db.Column(db.String(100), nullable=False, default='SHAIRE')  # e.g., SHAIRE or company name
    recipient = db.Column(db.String(100), nullable=False)  # e.g., company email or SHAIRE
    message_type = db.Column(db.String(50), nullable=True)  # e.g., 'notification', 'inquiry', 'follow-up'
    is_read = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    reply_to_message_id = db.Column(db.Integer, db.ForeignKey('company_messages.id'), nullable=True)

    company = db.relationship('Company', backref=db.backref('messages', lazy=True))

    def __repr__(self):
        return f'<CompanyMessages to/from Company ID {self.company_id} at {self.timestamp}>'
    
    @classmethod
    def get_company_messages(cls, company_name, limit=50):
        company = Company.query.filter_by(company_name=company_name).first()
        if not company:
            return []
        return cls.query.filter_by(company_id=company.id).order_by(cls.timestamp.desc()).limit(limit).all()
    
    @classmethod
    def mark_as_read(cls, message_id):
        message = cls.query.get(message_id)
        if message:
            message.is_read = True
            db.session.commit()
            return True
        return False
    
    @classmethod
    def save_company_message(cls, company_name, content, subject=None, sender='SHAIRE', recipient=None, message_type=None, reply_to_message_id=None):
        if not company_name or not content:
            return None
        company = Company.query.filter_by(company_name=company_name).first()
        if not company:
            return None
        new_message = cls(
            company_name=company_name,
            subject=subject,
            content=content,
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            reply_to_message_id=reply_to_message_id
        )
        db.session.add(new_message)
        db.session.commit()
        return new_message
    

""" 
Appointments scheduled between SHAIRE and companies
Need to manage appointment details, status, timestamps, and messages
The table will be connected to SHAIRE Dashboard for managing appointments and other actions
What is the foreign key company_name and company_id
"""
class AppointmentsWithCompanies(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    appointment_date = db.Column(db.DateTime, nullable=False)
    company_id = db.Column(db.Integer, db.ForeignKey('company.id'), nullable=True)
    company_name = db.Column(db.String(200), nullable=True)
    company_email = db.Column(db.String(120), nullable=True)
    company_contact = db.Column(db.String(200), nullable=True)
    message = db.Column(db.Text, nullable=True)
    purpose = db.Column(db.String(200), nullable=True)
    status = db.Column(db.String(50), nullable=False, default='scheduled') # e.g., scheduled, completed, canceled
    created_at = db.Column(db.DateTime, server_default=db.func.now())

    def __repr__(self):
        return f'<Appointments for User ID {self.user_id} on {self.appointment_date}>'
    
    @classmethod
    def get_shaire_appointments_per_period(cls, start_date, end_date):
        return cls.query.filter(cls.appointment_date >= start_date, cls.appointment_date <= end_date).all()
    

class SystemSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    setting_name = db.Column(db.String(100), unique=True, nullable=False)
    setting_value = db.Column(db.Text, nullable=False)
    description = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f'<SystemSettings {self.setting_name}>'