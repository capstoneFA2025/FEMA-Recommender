import streamlit as st
import pickle
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt_tab')
from create_input_stage1 import stage_one_input
from cap_search import doc_search, build_index

states = ['AL','AK','AZ','AR','CA','CO','CT','DE', 'FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI',
          'MN','MS','MO','MT','NE','NV','NH','NM','NY','NJ','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT',
          'VT','VA','WA','WV','WI','WY','DC','GU','PR','AS','MP','FM','MH','PW','VI']

reg_state={1:['CT','MA','ME','NH','RI','VT'],2:['NJ','NY','PR','VI'],3:['DC','DE','MD','PA','VA','WV'],4:['AL','FL','GA','KY','MS','NC','SC','TN'],
           5:['IL','IN','MI','MN','OH','WI'],6:['AR','LA','NM','OK','TX'],7:['IA','KS','MO','NE'],8:['CO','MT','ND','SD','UT','WY'],
           9:['AS','AZ','CA','FM','GU','HI','MH','MP','NV','PW'],10:['AK','ID','OR','WA']}

incident=['Biological','Chemical','Coastal Storm','Dam/Levee Break','Earthquake','Fire','Flood','Hurricane','Mud/Landslide',
          'Other','Severe Ice Storm','Severe Storm','Snowstorm','Straight-Line Winds','Terrorist','Tornado','Tropical Depression',
          'Tropical Storm','Typhoon','Volcanic Eruption','Winter Storm']

dec_type=['DR','EM']

esf=['ESF_0','ESF_1','ESF_2','ESF_3','ESF_4','ESF_5','ESF_6','ESF_7','ESF_8','ESF_9','ESF_10',
     'ESF_11','ESF_12','ESF_13','ESF_14','ESF_15']

esf_input=['supportFunction_0.0','supportFuncton_1.0','supportFuncton_2.0','supportFunction_3.0','supportFunction_4.0','supportFunction_5.0',
           'supportFunction_6.0','supportFunction_7.0','supportFunction_8.0','supportFunction_9.0','supportFunction_10.0','supportFunction_11.0',
           'supportFunction_12.0','supportFunction_13.0','supportFunction_14.0','supportFunction_15.0']

with open('trained_model.pkl', 'rb') as file:
    esf_model = pickle.load(file)

with open('trained_cluster_model.pkl','rb') as file2:
    cluster_model = pickle.load(file2)

st.title("Welcome to Responder Assist!")
sys_select = st.selectbox('Please select an option:',('Recommend mission assignments','Select mission assignments based on capability'),
                         index=0)
if sys_select=='Recommend mission assignments':
    #The following code allows the user to input context with dropdowns.
    #ESFs are suggested based on past data using a pretrained model.
    inc_type=st.multiselect('Please select incident types(s)',
                           incident, default=None)
    
    dec=st.selectbox('Please select declaration type',dec_type)

    #reg=st.selectbox('Please select region', list(range(1,11)))

    state=st.selectbox('Please select a state', states)
    
    if (len(inc_type)==0):
        st.markdown(':red[*Please select at least one incident type]'
                    )
    else:
        if st.button("Generate ESF Prediction"):
            
            X=stage_one_input(inc_type,state,dec)#,reg)
            if 'X' not in st.session_state:
                st.session_state['X']=X
            
            #pred=esf_model.predict(X)
            pred=esf_model.predict_proba(X)

            pred_2=pred.reshape(1,-1)
            #Create a dataframe with '1' if a ESF is recommended and '0' otherwise
            esf_df=pd.DataFrame(pred_2,index=[0],columns=esf)

            top=[] #greater than 75% likelihood
            mid=[] #50-75% likelihood
            bottom=[] #25-50% likelihood
            very_bottom=[] #less than 25% likelihood

            output_dict=esf_df.to_dict('list')
            
            for item in output_dict:
                if output_dict[item][0]>.75:
                    top.append(item)
                elif (output_dict[item][0]<=.75) & (output_dict[item][0]>=.5):
                    mid.append(item)
                elif (output_dict[item][0]<.5) & (output_dict[item][0]>=.25):
                    bottom.append(item)
                elif (output_dict[item][0]<.25):
                    very_bottom.append(item)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.header('Highly Likely (>75%)')
                col1.write(top)
            with col2:
                st.header('Likely (50-75%)')
                col2.write(mid)
            with col3:
                st.header('Unlikely (25-50%)')
                col3.write(bottom)
            with col4:
                st.header('Very Unlikely (<25%)')
                col4.write(very_bottom)
    
            #esf_user_select=st.multiselect('Please select ESFs:', esf[1:])

            #esf_user_df=pd.DataFrame(0,index=[0],columns=esf)

            #temporary, remove when done
            esf_user_select=st.multiselect('Please select ESFs:', esf_input[1:],default=None)

            if (len(esf_user_select)==0):
                st.markdown(':red[*Please select at least one ESF]'
                    )
            else:
                esf_user_df=pd.DataFrame(0,index=[0],columns=esf_input)
                #esf_user_df will be added to X from above, and used for stage 2 predictions
                #new column labels, will remove for the final app
                for e in esf_user_select:
                    esf_user_df.loc[:,e]=1

                X_new = X.join(esf_user_df)

                if st.button("Suggest 'Assistance Requested' Topics"):

                    pred_top=cluster_model.predict_proba(X_new)
                    pred_AR=pred_top.reshape(1,-1)
                    #st.markdown(cluster_model.classes_)
                    #AR_df=pd.DataFrame(pred_AR,index=[0],columns=cluster_model.)

else:
    cap=st.text_input('Please enter a capability',None)
    st.write('The capability is:', cap)
    if cap!=None:
        index, topics = build_index('Topics.txt')

        doc_search(cap, index, topics)