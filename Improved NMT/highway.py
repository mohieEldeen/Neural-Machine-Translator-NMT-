#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    """a class network that computes the highway of the output of the CNN character encoder"""

    def __init__ (self, word_embed_size):
        """
         a network that computes the highway of the output of the CNN character encoder

         args:
            word_embed_size: the embedding size of the words
        """
        super(Highway, self).__init__()

        self.highProj = nn.Linear(word_embed_size , word_embed_size)
        self.highGate = nn.Linear(word_embed_size , word_embed_size)


    def forward(self , xconv):
        """computes the highway of the output of the convert

        Args:
            xconv : the output of the cnn with shape of (max_seq_len , batch_size , word_embedding_size)

        return:

            xHigh: the output of the highway network with shape of (max_seq_len , batch_size , word_embedding_size)
        """

        xProj = torch.nn.functional.relu(   self.highProj(xconv)) #shape (max_seq_len , batch_size , word_embedding_size)

        xGate = torch.sigmoid(self.highGate(xconv)) #shape (max_seq_len , batch_size , word_embedding_size)

        xHigh = xGate * xProj + ( 1 - xGate) * xconv #shape (max_seq_len , batch_size , word_embedding_size)

        return xHigh

    ### END YOUR CODE
