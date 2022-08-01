**Interpreting Bidirectional-LSTM-CRF model for Disease Entities Recognition**
================

The project was developed by Akshat Gupta and Silvia Cunico undet the guidance of Prof. Roman Klinger, from the University of Stuttgart.
The program has 2 main purposes:
- Recognizing disease entities in text documents and labeling them according to the BIO labels.
- Interpreting the model's predictions with the LIME and GALE approximation techniques to explain LSTM CRF model. 

----------

## Contacts
------------

If you have any questions or problems, please e-mail **st180429@stud.uni-stuttgart.de , st179785@stud.uni-stuttgart.de**


## Installation
---------------

This program was developed using Python version 3.9.6 and was tested on Linux and Windows system.
We recommend using Anaconda 4.2 for installing **Python 3.9** as well as **numpy**, although you can install them by other means.

Other requirements:

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
---------------

Given a sentence, provide a "B-DISEASE" or "I-DISEASE" tag for each disease entity and "O" for all the remaining tokens. A classical application is Named Entity Recognition (NER) for Clinical or Biomedical NLP. Here is an example:

```
John   has   been   diagnosed   with    sporadic      T-cell	    leukaemia	
O      O     O      O           O       B-DISEASE     I-DISEASE     I-DISEASE
```



## Usage
---------
The program was written so that it can run for any type of entity. All the entity types presented in the data set must be 
>```
><Entity type 1>
><Entity type 2>
><...>
>```
#### **Recognize disease entities in text documents and labeling them according to the BIO tags**
Command:

    python main_NN.py
    
    Disease Entities classifier: Recognize disease entities in text documents and interpreting predictions

> **Note:**
> - "model" is the direct child folder within the "models/" folder
> - "dataset" is the direct child folder within the "dataset/" folder

#### **Evaluating pre-trained models**

> **Important Note:** The pre-trained models we included with the program were trained on Windows system. For that reason, evaluating those models on different OSs might give very different results.

> Evaluating on Windows will give these results:
> - For the BioCreative V CDR dataset:
>```
>                P       R       F1
>        Dis:    83.85   85.92   84.87
>        Chem:   92.78   93.54   93.16
>```
> - For the NCBI dataset:
>```
>                P       R       F1
>        Dis:    86.72   86.35   86.53
>```

#### **Training new models**
[**Step 0** - Download the pre-trained word embedding model]

Any pre-trained word embedding model can be used but it must be the same format as our provided word embedding model, which we highly recommend to be used. You can download the model file named "**wikipedia-pubmed-and-PMC-w2v.bin**" at this link: http://evexdb.org/pmresources/vec-space-models/

After the file was downloaded, place it within the "**pre_trained_models/**" folder.

[**Step 1** - Prepare the data]

All the data must be placed inside the `"data/<dataset_name>/"` folder, with `<dataset_name>` is replaced appropriately. The corpus that will be used for training must follow the BioCreative V format and be divided into at most three dataset: **train**, **development** and **test**, or at least one dataset: **train**.

The data will be prepared when running the follow command:

    python -m train.build_data <dataset> <train_set> <word_embedding> <ab3p_path> -dev <dev_set> -test <test_set>

> **Note:** Building data could take a while so be patient!

For more details, run:

    python -m train.build_data -h
    usage: build_data.pyc [-h] [-dev DEV_SET] [-test TEST_SET] dataset train_set word_embedding ab3p

        Build necessary data for model training and evaluating.

        positional arguments:
          dataset         the name of the dataset that the model will be trained on, i.e: cdr
          train_set       path to the training dataset, i.e: data/cdr/cdr_train.txt
          word_embedding  path to the word embedding pre-trained model,
                          i.e: pre_trained_models/wikipedia-pubmed-and-PMC-w2v.bin
          ab3p            path to the Ab3P program.

        optional arguments:
          -h, --help      show this help message and exit
          -dev DEV_SET, --dev_set DEV_SET
                          path to the development dataset, i.e: data/cdr/cdr_dev.txt
          -test TEST_SET, --test_set TEST_SET
                          path to the test dataset, i.e: data/cdr/cdr_test.txt

[**Step 2** - Training]

Command:

    python -m train.run -h
    usage: run.pyc [-h] [-dev DEV_SET] [-es | -e EPOCH] [-v] [-ds DISPLAY_STEP]
                    model dataset train_set

    Train new model.

    positional arguments:
      model                 the name of the model, i.e: d3ner_cdr
      dataset               the name of the dataset that the model will be trained on, i.e: cdr
      train_set             path to the training dataset, i.e: data/cdr/cdr_train.txt

    optional arguments:
      -h, --help            show this help message and exit
      -dev DEV_SET, --dev_set DEV_SET
                            path to the development dataset, i.e: data/cdr/cdr_dev.txt
      -es, --early_stopping
                            use early stopping
      -e EPOCH, --epoch EPOCH
                            number of epochs to train
      -v, --verbose         print training process
      -ds DISPLAY_STEP, --display_step DISPLAY_STEP
      						number of steps before displaying

Example: Training new model on CDR corpus with early stopping option

    python -m train.run d3ner_cdr cdr data/cdr/cdr_train.txt -dev data/cdr/cdr_dev.txt -es

  [1]: Curated chemical and disease vocabulary data were retrieved from the Comparative Toxicogenomics Database (CTD), MDI Biological Laboratory, Salisbury Cove, Maine, and NC State University, Raleigh, North Carolina. World Wide Web (URL: http://ctdbase.org/). October, 2017
  
  [2]: Mikolov,T. et al. (2013) Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, pp. 3111-3119.
  
  [3]: Sohn,S. et al. (2008) Abbreviation definition identification based on automatic precision estimates. BMC bioinformatics, 9(1), pp. 402.
