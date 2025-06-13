# USER INFORMATIONS
USER_ID_LENGTH = 5
USER_ID_MIN_MATCHING_CHARACTERS = 4
USER_ID_TEST = "Z95M6"

# PATHS
DIRECTORY_VIDEOS = '../Uploads/'
POLICY_DATA_FILE = 'verifications/cahier_des_charges.json'

# NB OF FRAMES TO ANALYZE IN THE VIDEO
NB_SNAPSHOTS_VIDEO_ANALYSIS = 7

# FACE DETECTION MODEL AGE GENDER
FACE_DETECTION_MODEL = 'retinaface'
FACE_CONFIDENCE_THRESHOLD = 0.5
MODEL_MEAN_VALUES = (104.0, 177.0, 123.0)
BLOB_MEAN_VALUES = (78.426, 87.768, 114.895)
BLOB_SIZE = (227, 227)
BLOB_FRAME_SCALE = 3.0

# SPEECH EXTRACTION
DEFAULT_SUMMARY_TEXT = "No text to summarize"
SENTENCE_SCORE_THRESHOLD = 0.7

# TEXT PROCESSING
REPLACE_WORDS = {
    'coral': 'car',
    'railway': 'reliable',
    'Butcher': 'budget',
    'butcher': 'budget',
    'scorers': 'car',
    'scorer': 'car',
    'Scarface': 'scaler'
}

# TEXT VALIDATION VS AI GENERATED TEXT
THRESHOLD_SIMILARITY_RATIO = 0.65
GENERATED_TEXT_MODEL = 'gpt-3.5-turbo'
GENERATED_TEXT_MODEL_MAX_TOKENS = 1000
AI_GENERATED_SPEECH = "One two three four MD reveals Next Generation AI ships with open a ICU summit AMD on Thursday in Wales new details about Next Generation AI chips " \
"Instinct am I 400 series that will be shipped next year see you later the launch events in San Jose California the chips will be able to be used as part of" \
"the Rock scale system MD said it's important for customers that want to hyper scaler of AI computers that can spend entire data centers open aicu simultane appear on the stage and set a company would use the AMD chips it's got it's going to be an amazing thing pmd rock scale setup will make the chips look to use like one system which is important for most artificial intelligence like that providers and companies that develop large language model those customers want hyper scales clusters of the eye computers that can spend in Toyota centuries and use massive amounts of power open AI in a table"

# TEXT RECOGNITION 
TEXT_DETECTION_CONFIDENCE_THRESHOLD = 0.5

# TEXT RECOGNITION in YOLO
LIST_BRANDS = ['Nescafe', 'Apple', 'Samsung', 'Honor', 'Oppo', 'Huawei', 'Xiaomi']
LIST_PRODUCTS = ['phone', 'laptop', 'tablet', 'watch', 'headphones']

# OBJECT DETECTION
LISTE_OBJETS = ['person', 'cup', 'dish', 'knife', 'bottle', 'scissor', 'cake', 'plate', 'punnet', 'basket', 'eye', 'carrot',
                'bowl', 'fork', 'spoon', 'bag', 'glove', 'book', 'board', 'strawberry', 'hand', 'socket', 'sink', 'handle',
                'cabinet', 'switch', 'lamp', 'banana', 'tree', 'canvas', 'frame', 'chair','glasses','smartphone','laptop',
                'monitor','keyboard','mouse','headphone','earphone','speaker','tablet','camera','projector','printer', 'radio',
                'television','remote','clock','watch','calculator','scale','tape measure','ruler','pencil','pen',
                'lion','box']  # not used yet

BOX_DIM_TOLERANCE = 10

# VERIFCATION AND MARKING OF THE VIDEO
COMPLIANCE_ACCEPTED_THRESHOLD = 0.5
COMPLIANCE_DIRECTORY = 'compliance_metrics'
COMPLIANCE_BASEFILE = 'compliance_metrics.json'
WEIGHT_FOR_AUTH = 3
WEIGHT_FOR_DURATION_NBWORDS = 3