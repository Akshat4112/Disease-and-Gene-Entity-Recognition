import streamlit as st 
import spacy_streamlit
import spacy
nlp = spacy.load('en_core_web_sm')
import os
from PIL import Image


#Load ML Model

def main():
    st.title("NER for Disease and Gene")
    raw_text = st.text_area("Your Text","Enter Text Here")
    tokens = raw_text.split()
    
    # docx = nlp(raw_text)
    # spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)
if __name__ == '__main__':
	main()