# imports
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# dataset input
# preferably take the online input of the dataset


# preprocessing the dataset
# define the max samples to be taken from the input
max_samples = 10000

# preprocessing(Multiple methods can be used here)


def preprocess_sample(sample):
    # extract sentence from a sample
    # change code
    sentence = sample
    # lower the characters and creates the sapce between characters and punction
    sentence = sentence.lower().strip()
    # removing characters other than the alphabets & ". , ? , ! , ',' "
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # adding start and end token to the sentence
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence

# Loading Conversation from the dataset


def load_conversation():
    # Code for the loading conversation depending upon the dataset


def tokenize():
    # tokenizing the sentence
    # tokenizer of choice
    # Or conversion to encoding like universal snetence encoding


max_tok_len = 40


def tokenize_filter():
    # changes to the token to be made


batch_size = 64
buffer_size = 20000


def dataset():
    # define the dataset
