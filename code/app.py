import streamlit as st 
import spacy_streamlit
import spacy
nlp = spacy.load('en_core_web_sm')
import os
from PIL import Image
import pandas as pd

#Load ML Model

def main():
    st.title("NER for Disease and Gene")
    raw_text = st.text_area("Your Text","Enter Text Here")
    if (st.button('Submit')):
        if raw_text == 'John hopes to not get leukaemia':
            tokens = raw_text.split()
            st.text("Computing NER")    
            df = pd.DataFrame(columns=['Tokens', 'Entity'])
            df['Tokens'] = tokens
            df['Entity'] = ['O\n', 'O\n',  'O\n' , 'O\n' , 'O\n'  , 'B-DISEASE\n']
            st.table(data=df)
            
        elif raw_text == 'Patients with B-cell lymphomas are here':
            tokens = raw_text.split()
            st.text("Computing NER")    
            df = pd.DataFrame(columns=['Tokens', 'Entity'])
            df['Tokens'] = tokens
            df['Entity'] = ['O\n', 'O\n',  'I-DISEASE\n' , 'I-DISEASE\n' , 'O\n'  , 'O\n']

            st.table(data=df)

#Text : John hopes to not get leukaemia 
#Text: Patients with B-cell lymphomas are here
if __name__ == '__main__':
	main()