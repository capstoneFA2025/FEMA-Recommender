"""This file takes the input from the user and creates a dataframe to be passed into a ML model for prediction"""
import pandas as pd
import regex as re

cols=['state_AK','state_AL','state_AR','state_AS','state_AZ','state_CA','state_CO','state_CT','state_DC','state_DE','state_FL',
 'state_GA','state_GU','state_HI','state_IA','state_ID','state_IL','state_IN','state_KS','state_KY','state_LA','state_MA','state_MD',
 'state_ME','state_MI','state_MN','state_MO','state_MP','state_MS','state_MT','state_NC','state_ND','state_NE','state_NH','state_NJ',
 'state_NM','state_NV','state_NY','state_OH','state_OK','state_OR','state_PA','state_PR','state_RI','state_SC','state_SD',
 'state_TN','state_TX','state_UT','state_VA','state_VI','state_VT','state_WA','state_WI','state_WV','state_WY',
 'declarationType_DR','declarationType_EM','region_1','region_2','region_3','region_4','region_5','region_6','region_7',
 'region_8','region_9','region_10','designatedIncidentTypes_Biological','designatedIncidentTypes_Chemical',
 'designatedIncidentTypes_Coastal Storm','designatedIncidentTypes_Dam/Levee Break','designatedIncidentTypes_Earthquake',
 'designatedIncidentTypes_Fire','designatedIncidentTypes_Flood','designatedIncidentTypes_Hurricane',
 'designatedIncidentTypes_Mud/Landslide','designatedIncidentTypes_Other','designatedIncidentTypes_Severe Ice Storm',
 'designatedIncidentTypes_Severe Storm','designatedIncidentTypes_Snowstorm','designatedIncidentTypes_Straight-Line Winds',
 'designatedIncidentTypes_Terrorist','designatedIncidentTypes_Tornado','designatedIncidentTypes_Tropical Depression',
 'designatedIncidentTypes_Tropical Storm','designatedIncidentTypes_Typhoon','designatedIncidentTypes_Volcanic Eruption',
 'designatedIncidentTypes_Winter Storm']

df = pd.DataFrame(0,index=[0],columns=cols)

def stage_one_input(incident_types,state,dec_type):#,region):
    """Function accepts input from user selection, which is returned as a list"""
    for i in incident_types:
        inc='designatedIncidentTypes_'+i
        df[inc]+=1

    df['state_'+state]+=1
    
    df['declarationType_'+dec_type]+=1

    #df['region_'+str(region)]+=1

    return df  