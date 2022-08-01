**Interpreting Bidirectional-LSTM-CRF model for Disease Entities Recognition**
================

The project was developed by Akshat Gupta and Silvia Cunico under the guidance of Prof. Roman Klinger, from the University of Stuttgart.
The program has 2 main purposes:
- Recognizing disease entities in text documents and labeling them according to the BIO labels
- Interpreting the model's predictions with the LIME and GALE approximation techniques to explain the BI-LSTM CRF model

----------

## Contacts
------------

If you have any questions or problems, please e-mail **st180429@stud.uni-stuttgart.de , st179785@stud.uni-stuttgart.de**


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

## Datasets
------------




## Results
----------



## Usage
---------
The program was written so that it can run to distinguish "DISEASE" entities from other entities.  
The NaiveBayes.py contains our Naive Bayes algorithm, a probabilistic and generative classifier, trained on 
```


#### **Recognize disease entities in text documents and labeling them according to the BIO tags**


#### **Interpreting the model's predictions with the LIME and GALE approximation techniques to explain the BI-LSTM CRF model**

> **Important Note:** The pre-trained models  

