import sys
import nltk
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, models, util
from torch import nn
import tensorflow as tf

from transformers import logging

logging.set_verbosity_error()


def read_txt(text_file, question_file):
    with open(text_file, 'r') as file:
        data = file.read().replace('\n', '')
        text_sentenced = nltk.sent_tokenize(data)
        sentences = [sentence for sentence in text_sentenced]

    with open(question_file, 'r') as file:
        questions = file.read().split("\n")

    # print(questions)
    # print(sentences)
    return sentences, questions


def get_question_type(questions):
    nlp = spacy.load('en_core_web_sm')
    question_types=[]
    for question in questions:
        doc_question = nlp(question)
        question_type = "YES/NO"

        for token in doc_question:
            if token.tag_ in ["WDT", "WP", "WP$", "WRB"]:
                question_type = token.text.upper()
                break

        question_types.append(question_type)
    # print(question_types)
    return question_types


def select_sentence(model,sentences,question):

    question_embedding = model.encode(question)
    sentences_embedding = model.encode(sentences)
    cosine_scores = util.cos_sim(question_embedding, sentences_embedding)
    list_cosine_sim=tf.reshape(cosine_scores,-1).numpy()
    index = np.argmax(list_cosine_sim)

    # # Output the pairs with their score
    # list_cosineScore = []
    # for i in range(len(sentences)):
    #     for j in range(len(questions)):
    #         print(cosine_scores[j][i])
    #     # list_cosineScore.append(cosine_scores[i])
    return sentences[index]

if __name__ == "__main__":
    text_file = sys.argv[1]
    question_file = sys.argv[2]

    sentences, questions = read_txt(text_file, question_file)
    question_type = get_question_type(questions)
    word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256,activation_function=nn.Tanh())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model,dense_model])
    for i in range(len(questions)):
        select_sentence = select_sentence(model,sentences,questions[i])
        # print(select_sentence)
    print("Question : ",questions)
    print("Question Type: ",question_type)
    print("Closest sentence using Cosine Similarity : ",select_sentence)
