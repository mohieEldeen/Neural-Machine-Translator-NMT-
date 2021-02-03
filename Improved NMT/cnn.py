#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    a cnn networ that's used to encode the character embeddings and calculate the word embeddings from it

    """
    ### YOUR CODE HERE for part 1g
    def __init__(self , word_embed_size, char_embed_size ):

        """
        a cnn networ that's used to encode the character embeddings and calculate the word embeddings from it

        args:
            word_embed_size : the size of the word embeddings
            char_embed_size : the size of the char embeddings
        """

        super(CNN, self).__init__()

        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        self.conv1 = nn.Conv1d(char_embed_size , word_embed_size , kernel_size = 5 , padding = 1)


    def forward(self , sents_char_embedded):

        """
        calculate the word embeddings out of a char embeddings input

        args:
            sents_char_embedded: an input tensor of shape (max_seq_len , batch_size , max_word_len , char_embed)

        returns:
            xconv: the output tensor of shape (max_seq_len , batch_size , word_embedding_size)

        """

        max_seq_len = sents_char_embedded.shape[0]
        batch_size = sents_char_embedded.shape[1]
        max_word_len = sents_char_embedded.shape[2]

        

        #reshaping the input to prepare it for the convNet
        sents_char_embedded = sents_char_embedded.contiguous().reshape((max_seq_len * batch_size   , max_word_len , self.char_embed_size)).transpose(1,2) # (max_seq_len * batch_size , char_embed , max_word_len)

        xconv = nn.functional.relu(self.conv1(sents_char_embedded))# (max_seq_len * batch_size , word_embed , max_word_len-2)

        xconv = torch.max( xconv , dim = 2)[0]  # (max_seq_len * batch_size , word_embed )

        xconv = xconv.contiguous().reshape((max_seq_len , batch_size , self.word_embed_size)) #shape (max_seq_len , batch_size , word_embedding_size)


        return xconv


    ### END YOUR CODE
