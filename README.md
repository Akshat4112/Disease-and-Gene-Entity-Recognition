**Interpreting Bidirectional-LSTM-CRF model for Disease Entities Recognition**
================

The project was developed by Akshat Gupta and Silvia Cunico under the guidance of Prof. Roman Klinger, from the University of Stuttgart.
The program has 2 main purposes:
- Recognizing disease entities in text documents and labeling them according to the BIO labels
- Interpreting the model's predictions with the LIME and GALE approximation techniques to explain the BI-LSTM CRF model

----------

## Installation
---------------

This program was developed using Python version 3.9.6 and was tested on Linux and Windows system.
We recommend using Anaconda 4.2 for installing **Python 3.9** as well as **numpy**, although you can install them by other means.

If you wish to run the code, you can install the dependencies from the requirements.txt file.

    pip install -r requirements.txt

1. **Tensorflow** GPU 2.7.0:

If GPU is not available, tensorflow CPU can be used instead:
> pip install tensorflow==2.7.0

2. **Keras** 2.9.0:
3. **Sklearn** 0.21.0:
4. **NLTK** 3.2.5:
5. **Eli5**:
Currently it requires scikit-learn 0.18+
> conda install -c conda-forge eli5


## Task
-----------

Train a model given a labeled dataset to provide a "B-DISEASE" or "I-DISEASE" tag for each disease entity and "O" for all the remaining tokens on an unlabeled corpus. A classical application is Named Entity Recognition (NER) for Clinical or Biomedical NLP. Here is an example:

```
John   has   been   diagnosed   with    sporadic      T-cell	    leukaemia	
O      O     O      O           O       B-DISEASE     I-DISEASE     I-DISEASE

```
After training a model:
1. Given a predicition result, evaluate its accuracy and F1 score (confusion matrix). 
2. Generate explanations to get both a local and a global linear approximation of the model’s behaviour. 


### Datasets
------------

The dataset used to train our models was provided to us internally from the University stuff, for the purpose of developing this project.

**If you may wish to use the code you can provide a dataset with the same format so that it will be processed regularly and written as a pandas DataFrame in a CSV file for the later models' trainings. Dump such DataFrame in a /data folder and name it "dfnew.csv"**

The format we refer to follows roughly the CoNLL 2003 format for NER task (https://aclanthology.org/W03-0419.pdf): the data file must contain one word per line. At the end of each line a tag must state whether the current word is part of a named entity or not. In our case the fields in a line must be two, the word and its entity tag. The part of-speech tag and the chunk tag are not part of the required fields. It is essential that the two required fields are tab-separated.


### Results
----------

Results' plots from our trainings and evaluations of the BI-CRF-LSTM model on our NCBI-based dataset were stored through the NeuralNetwork's method Training_plots as .png files in the /figures/ data folder.

If you may wish to create plots of the accuracy and losses of your models, please create a /figures folder, too.


### Usage
---------
The program was written so that it can run to distinguish "DISEASE" entities from other entities with the highest accuracy.
To reach such training accuracy (99,82) and validation accuracy (99,29%) we implemented: 

1. A baseline method following the Naive Bayes algorithm to train and test;
2. An advanced method with a Bidirectional-LSTM-CRF model with BioBERT Embeddings. 


#### **Recognize disease entities in text documents and labeling them according to the BIO tags**

##### _A baseline method following the Naive Bayes algorithm to train and test_

```
   python NaiveBayes.py
```
To run the Naive Bayes generative classifier.

```
   python Evaluation.py
```
To evaluate the Naive Nayes classifier's B, I, O labels predictions. 

##### _An advanced method with a Bidirectional-LSTM-CRF model with BioBERT Embeddings_

Command
```
   python main_NN.py
```
**Important Notes:** 
Before running the command:
- Make sure that the CSV file with the DataFrame is named and saved correctly as mentioned above in the [Datasets paragraph](#datasets);
- Make sure that you have a /data folder in order for the word2idx and the tag2idx features to be saved as pickle files (for the NERExplainerGenerator class); 
- Make sure that you have a /models folder in order to save the trained models' binaries;
- Make sure that you have a /figures folder in order to store the accuracy plots .png files.

#### **Interpreting the model's predictions with the LIME and GALE approximation techniques to explain the BI-LSTM CRF model**

Command
```
   python Explainer.py
```
**Important Notes:** 
1. Only applies after having trained the advanced method with a Bidirectional-LSTM-CRF model (link) so requisites are:
    1. Having trained a model and saved its binary as .h5 
    2. Having saved the features word2idx and tag2idx as pickle files
2. Make sure that you have a: /data/ner-disease/ folder to save the explanatory sentences and words after the LIME linear local approximation
3. Make sure that the labels correspond to such format: '|B-DISEASE\n' '|I-DISEASE\n' '|O\n' 

## Contacts
------------

If you have any questions or problems, please e-mail **st180429@stud.uni-stuttgart.de , st179785@stud.uni-stuttgart.de**
