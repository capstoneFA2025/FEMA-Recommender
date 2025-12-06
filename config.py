"""Configuration and constants for FEMA Recommender Application."""

from pathlib import Path
from typing import Dict, List

APP_TITLE = "Responder Assist"
APP_VERSION = "2.0.0"

DATA_DIR = Path(__file__).parent
ESF_MODEL_PATH = DATA_DIR/"trained_model_ESF.pkl"
AR_TOPIC_MODEL_PATH = DATA_DIR/"trained_model_AR_topic.pkl"
AR_TOPICS_FILE = DATA_DIR/"AR_topics.txt"
SOW_TOPICS_FILE = DATA_DIR/"SoW_topics.txt"

STATES = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 
    'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 
    'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 
    'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 
    'New Mexico', 'New York', 'New Jersey', 'North Carolina', 'North Dakota', 'Ohio', 
    'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 
    'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 
    'Wisconsin', 'Wyoming', 'District of Columbia', 'Guam', 'Puerto Rico', 'American Samoa', 
    'Northern Mariana Islands', 'Federated States of Micronesia', 'Marshall Islands', 
    'Palau', 'U.S. Virgin Islands'
]

STATE_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH',
    'New Mexico': 'NM', 'New York': 'NY', 'New Jersey': 'NJ', 'North Carolina': 'NC',
    'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN',
    'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC',
    'Guam': 'GU', 'Puerto Rico': 'PR', 'American Samoa': 'AS', 'Northern Mariana Islands': 'MP',
    'Federated States of Micronesia': 'FM', 'Marshall Islands': 'MH', 'Palau': 'PW',
    'U.S. Virgin Islands': 'VI'
}

REGIONS: Dict[int, List[str]] = {
    1: ['CT', 'MA', 'ME', 'NH', 'RI', 'VT'],
    2: ['NJ', 'NY', 'PR', 'VI'],
    3: ['DC', 'DE', 'MD', 'PA', 'VA', 'WV'],
    4: ['AL', 'FL', 'GA', 'KY', 'MS', 'NC', 'SC', 'TN'],
    5: ['IL', 'IN', 'MI', 'MN', 'OH', 'WI'],
    6: ['AR', 'LA', 'NM', 'OK', 'TX'],
    7: ['IA', 'KS', 'MO', 'NE'],
    8: ['CO', 'MT', 'ND', 'SD', 'UT', 'WY'],
    9: ['AS', 'AZ', 'CA', 'FM', 'GU', 'HI', 'MH', 'MP', 'NV', 'PW'],
    10: ['AK', 'ID', 'OR', 'WA']
}

INCIDENT_TYPES = [
    'Biological', 'Chemical', 'Coastal Storm', 'Dam/Levee Break', 'Earthquake',
    'Fire', 'Flood', 'Hurricane', 'Mud/Landslide', 'Other', 'Severe Ice Storm',
    'Severe Storm', 'Snowstorm', 'Straight-Line Winds', 'Terrorist', 'Tornado',
    'Tropical Depression', 'Tropical Storm', 'Typhoon', 'Volcanic Eruption',
    'Winter Storm'
]

DECLARATION_TYPES = ['DR', 'EM']

ESF_COLUMNS = [
    'ESF_0.0', 'ESF_1.0', 'ESF_2.0', 'ESF_3.0', 'ESF_4.0', 'ESF_5.0',
    'ESF_6.0', 'ESF_7.0', 'ESF_8.0', 'ESF_9.0', 'ESF_10.0', 'ESF_11.0',
    'ESF_12.0', 'ESF_13.0', 'ESF_14.0', 'ESF_15.0'
]

PROBABILITY_THRESHOLDS = {
    'highly_likely': 0.75,
    'likely': 0.50,
    'unlikely': 0.25
}

ERROR_MESSAGES = {
    'no_incident': 'Please select at least one incident type',
    'no_esf': 'Please select at least one ESF',
    'model_load_error': 'Error loading prediction model',
    'file_not_found': 'Required data file not found'
}
