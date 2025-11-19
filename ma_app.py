import streamlit as st
import pickle
from create_input_stage1 import stage_one_input
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
import pandas as pd
from cap_search import doc_search, build_index
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')

states = ['AL','AK','AZ','AR','CA','CO','CT','DE', 'FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI',
          'MN','MS','MO','MT','NE','NV','NH','NM','NY','NJ','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT',
          'VT','VA','WA','WV','WI','WY','DC','GU','PR','AS','MP','FM','MH','PW','VI']

reg_state={1:['CT','MA','ME','NH','RI','VT'],2:['NJ','NY','PR','VI'],3:['DC','DE','MD','PA','VA','WV'],4:['AL','FL','GA','KY','MS','NC','SC','TN'],
           5:['IL','IN','MI','MN','OH','WI'],6:['AR','LA','NM','OK','TX'],7:['IA','KS','MO','NE'],8:['CO','MT','ND','SD','UT','WY'],
           9:['AS','AZ','CA','FM','GU','HI','MH','MP','NV','PW'],10:['AK','ID','OR','WA']}

incident=['Biological','Chemical','Coastal Storm','Dam/Levee Break','Earthquake','Fire','Flood','Hurricane','Mud/Landslide',
          'Other','Severe Ice Storm','Severe Storm','Snowstorm','Straight-Line Winds','Terrorist','Tornado','Tropical Depression',
          'Tropical Storm','Typhoon','Volcanic Eruption','Winter Storm']

dec_type=['EM','DR']

esf=['ESF_0','ESF_1','ESF_2','ESF_3','ESF_4','ESF_5','ESF_6','ESF_7','ESF_8','ESF_9','ESF_10',
     'ESF_11','ESF_12','ESF_13','ESF_14','ESF_15']

with open('trained_model.pkl', 'rb') as file:
    esf_model = pickle.load(file)

st.title("Welcome to Responder Assist!")
sys_select = st.selectbox('Please select an option:',('Recommend mission assignments','Select mission assignments based on capability'),
                         index=0)
if sys_select=='Recommend mission assignments':
    #The following code allows the user to input context with dropdowns.
    #ESFs are suggested based on past data using a pretrained model.
    inc_type=st.multiselect('Please select incident types(s)',
                           incident, default=None)
    dec=st.selectbox('Please select declaration type',dec_type)

    reg=st.selectbox('Please select region', list(range(1,11)))

    state=st.selectbox('Please select a state', reg_state[reg])
    st.markdown(len(inc_type))
    if (inc_type==None):
        st.markdown('Please select at least one incident type'
                    )
    X=stage_one_input(inc_type,state,dec,reg)

    #st.markdown("Suggesting Emergency Support Functions....")
    pred=esf_model.predict(X)

    pred=pred.reshape(1,-1)
    #Create a dataframe with '1' if a ESF is recommended and '0' otherwise
    esf_df=pd.DataFrame(pred,index=[0],columns=esf)

    #Create lists of suggested ESFs to return to user
    pos_list=[]
    neg_list=[]
    
    for e in esf:
        if esf_df.loc[0,e]==1:
            pos_list.append(e)
        else:
            neg_list.append(e)

    pos_esf=" ".join(pos_list)
    neg_esf=" ".join(neg_list)
    #st.markdown(pos_esf)
    st.markdown("Recommended ESFs:")
    st.markdown(pos_esf)
    
    #st.markdown("ESFs not suggested:")  
    #st.markdown(neg_esf)
    
    esf_user_select=st.multiselect('Please select ESFs:', esf[1:])

    esf_user_df=pd.DataFrame(0,index=[0],columns=esf)

    for e in esf_user_select:
        esf_user_df.loc[:,e]=1

    #esf_user_df will be added to X from above, and used for stage 2 predictions

else:
    cap=st.text_input('Please enter a capability',None)
    st.write('The capability is:', cap)

    index, topics = build_index('Topics.txt')

    doc_search(cap, index, topics)