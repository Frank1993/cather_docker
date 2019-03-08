SEQUENCE_ID = 'sequence_id'
IMAGE_INDEX = 'sequence_index'
LONGITUDE = 'longitude'
LATITUDE = 'latitude'
IMAGE_WIDTH = 'photo_width'
IMAGE_HEIGHT = 'photo_height'
HEADING = 'heading'
ROI_ID = 'detection_id'
ROI_X = 'x'
ROI_Y = 'y'
ROI_WIDTH = 'detection_width'
ROI_HEIGHT = 'detection_height'
CONFIDENCE = 'confidence_level'
VALIDATION_STATUS = 'validation_status'
DETECTION_MODE = 'mode'
SIGN_TYPE = 'internal_name'
REGION = 'region'
PARENT_ID = 'parent_id'
FILE_PATH = 'file_path'
FILE_NAME = 'file_name'
SHOT_DATE = 'shot_date'

IMAGE_PROTO = 'image_proto'

MANUAL = 'MANUAL'
AUTOMATIC = 'AUTOMATIC'

CONFIRMED = 'CONFIRMED'
REMOVED = 'REMOVED'
TO_BE_CHECKED = 'TO_BE_CHECKED'

YES = 'yes'
OSV = 'osv'
NOW = 'now'
BOT = 'bot'  # beginning of time
PHOTOS = 'photos'
METADATA_FILENAME = 'meta_data_filename'

IMAGE_FIELDS = [SEQUENCE_ID, IMAGE_INDEX, LONGITUDE, LATITUDE, IMAGE_WIDTH, IMAGE_HEIGHT, HEADING, REGION, FILE_PATH, SHOT_DATE]

validation_correspondence = {'CONFIRMED': 'TRUE_POSITIVE', 'REMOVED': 'FALSE_POSITIVE', 'TO_BE_CHECKED': 'UNKNOWN'}

OSV_URL_DETAILS = 'http://openstreetview.com/details'
OSV_URL_PHOTO_LIST = 'http://openstreetview.com/1.0/sequence/photo-list/'
