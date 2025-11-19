"""This module allows the user to input search terms and find relevant MA assistance requests"""
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

def build_index(doc):
    with open(doc,'r') as file:
        lines = file.readlines()

    #convert topics ('documents') to tokens, the store in a dictionary
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    docs = {}
    topics = {}
    #counter = 1

    for line in lines:
        line.strip()
        tokens = word_tokenize(line)
        pos = tokens[0]
        doc_id = 'd'+pos
    
        info = tokens[2:]
    
        text = []
        for word in info:
            if word not in stop_words:
                stem_word = stemmer.stem(word)
                text.append(stem_word)
        topics[doc_id]=line         
        docs[doc_id]=text

    term_index={}

    for i in range(len(docs)):
        doc_id='d'+str(i)
        #get terms from documents in order with doc_id
        terms=docs[doc_id]
        #check that terms are in inverted index
        counter=0  #initialize counter for index position of term in terms
        for term in terms:
            if term in term_index:
                #add 1 to counter, append index position
                if doc_id in term_index[term]: 
                    term_index[term][doc_id][0]+=1
                    term_index[term][doc_id][1].append(counter)
                else:
                    #create dictionary entry for document, counter at 1 and term position added
                    doc_term=[1,[counter]]
                    term_index[term][doc_id]=doc_term
            else:
                doc_term=[1,[counter]]
                term_index[term]={}
                term_index[term][doc_id]=doc_term
            counter+=1
    
    return term_index, topics

def doc_search(query, index, topics):
    #takes a query string, index (dict) and topics (dict) as inputs, prints results
    q_toks = word_tokenize(query)
    
    for tok in q_toks:
        if tok not in stop_words:
            stem_tok = stemmer.stem(tok)
            if stem_tok in index:
                for doc in index[stem_tok]:
                    print(topics[doc])