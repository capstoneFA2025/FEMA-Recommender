"""This file takes the input from the user (ESF selection) and creates a dataframe to be passed into a ML model for prediction"""
import pandas as pd

def get_AR_topics(doc):

    with open(doc,'r') as file:
        lines = file.readlines()

    AR_topics={}
    for line in lines:
        #line=line.strip()
        split_line=line.strip().split('-')
        AR_topics[int(split_line[0].strip())]=split_line[1]

    return AR_topics

def get_SOW_topics(doc):
    with open(doc,'r') as file:
        lines = file.readlines()

    SOW_topics={}
    for line in lines:
        #line=line.strip()
        split_line=line.strip().split('-')
        SOW_topics[int(split_line[0].strip())]=split_line[1]

    return SOW_topics
    
def AR_SoW_connect():
    connect={}
    return connect
