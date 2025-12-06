# FEMA-Recommender
A MADS capstone project based on a recommendation system

## README requirements
Your README is the technical documentation for your repo. Provide an overview of your project, how it's organized, and instructions for using your code. Your repository should be documented with a welcoming, informative README so a data scientist from outside MADS could understand what you did!

## About the project
Federal Emergency Management Agency (FEMA) Incident Management Assistance Teams (IMAT) are the first FEMA personnel to respond to a Federally declared disaster, and are responsible for building out the Federal response in conjunction with the State, Local, Tribal or Territorial (SLTT) government.  One of the problems encountered in response operations is the volume of work required at the start of a disaster coupled with the lack of staff to perform the work.  Hours can be spent generating ideas of what mission assignments are needed, and the potential scope of the operation - let alone the actual scope as that may take days to learn.  For this project, we envision using existing data from disasters to provide a virtual 'assistant' to emergency managers in this sort of situation.  Instead of spending hours determining what is likely needed and then drafting from scratch (or a template), what if we could use machine learning and other data science tools to do some of that basic work and recommend the most likely mission assignments for that disaster and jurisdiction?  Beneficiaries could not only be FEMA IMATs, but also the SLTT emergency managers who frequently do not have much experience, staff, or knowledge of FEMA programs.

This project works off of two datasets available from FEMA. The first contains information about past disaster declarations. It records the disaster number, impacted counties, state, region, and disaster type.  In addition to the disaster type for the declaration, it includes additional disasters occurring in the same area.  For instance, if a wildfire was followed by heavy rain, then flooding and landslides may be additional disasters in the declaration.  All of these types of disasters are important in understanding the mission assignments requested. The second consists of approved mission assignments from October 2012- late 2024.  These are real-world mission assignments that are recorded by type of assistance, Emergency Support Function (ESF), responsible agency, region, state, and disaster number.  We are most interested in the assistance requested (AR), as that is the starting point for all mission assignments.

Our project creates a working user interface that allows users multiple options to help them build mission assignments. They may start by providing the state, type of disaster(s), and declaration type (*need to describe this above*) and our first multilabel classification model will make recommendations of the ESFs most likely to accompany the submitted features. If the user wishes to start with their own list of ESFs, they can move on to the second part of the interface. After they make their selections of ESFs, that information and the previously selected features will be used by our second multilabel classification model to recommend AR topics to them to complete their assignments. These topics and related ... were grouped into clusters by a third model to provide the best related topics as determined by our .... Finally, the user can search for a Statement of Work based on information from the AR topics that were recommended. If the user wishes to start by simply searching for particular topics in both the AR topics and Statements of Work, they can start there instead.

## How the project is organized
Notebooks for this project are split into separate folders, each with a requirements.txt file. This was done due to the very specific combination of libraries and associated versions necessary for the Large Language Model (LLM) notebook to function. 

## How to run the code
PLACE INFO HERE - WHAT'S THE ORDER THAT THESE NEED TO RUN IN ORDER TO MAKE SENSE?

* CLUSTERING
* LLM
* EDA
* MERGE CLEAN SPLIT
* MODEL NOTEBOOKS

### Obtaining the data files
WE PROBABLY NEED TO CREATE A MAKEFILE THAT WILL DOWNLOAD THE MA AND DDS DATASETS FROM GOOGLE DRIVE. JOE'S NOTEBOOKS ARE USING A PARQUET VERSION INSTEAD OF CSV BUT MAYBE THIS SHOULD CHANGE?

### Clustering notebooks?

### LLM notebooks?
NOTE THAT JACOB'S LLM NOTEBOOK WILL REQUIRE AN AWS API KEY AND TAKE 7 HOURS OF COMPUTE TIME.

### Multilabel classification notebooks
The exploratory data analysis notebooks (explore_ma.ipynb and explore_dds.ipynb) are just for exploratory purposes and do not create any essential files for the other notebooks to use. These both create visualizations as part of the analysis and explore_ma does save one visualization to a file. Merge_clean_and_split_stratified.ipynb performs the task of merging the MA, DDS, and AR Topic datasets together, then splits those into a training set and test set. This must be run in order to create the training and test parquet files used by the multilabel classification notebooks to create models for later use. Multilabel_classification_model_selection_ESF.ipynb should be run next to create the ESF prediction model, which will save to a pickle file as part of the notebook. Multilabel_classification_model_selection_AR_topics.ipynb is the next notebook to run as it will create and save the AR topic prediction model to a different pickle file. 

### Streamlit app branch
The files here do not need to be run as part of the project, which is why they exist in a separate branch. They are used by the Streamlit site to provide a user interface to perform our inference and information retrieval functions. The front end is available at https://fema-recommendergit-se8dcweiaepgsg7ashtvna.streamlit.app/


## Does your dataset have any usage restrictions? Please check for a license associated with the dataset. 
No usage restrictions.  This is Federal data that is publicly available.


