#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                                           padding_idx=self.target_vocab.char_pad)

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.

        char_embedded = self.decoderCharEmb(input) # shape (word_len , batch_size , char_embed_size)

        if dec_hidden:
            (hidden_states , (last_hidden , last_cell)) = self.charDecoder(char_embedded , dec_hidden)
        else :
            (hidden_states , (last_hidden , last_cell)) = self.charDecoder(char_embedded )
            #hidden_states shape (word_len , batch_size , hidden_size)
            #last_hidden shape (1 , batch_size , hidden_size)
            #last_cell shape (1 , batch_size , hidden_size)

        scores = self.char_output_projection(hidden_states) # shape (word_len , batch_size , vocab_size)


        return  scores ,  (last_hidden , last_cell)


        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss

        # Chop of the <END> token for max length words.
        input_char_sequence = char_sequence[:-1] #shape (word_length - 1, batch_size)
        target_char_sequence = char_sequence[1:] #shape (word_length - 1, batch_size)
        target_char_sequence_mask = (target_char_sequence != self.target_vocab.char_pad).float() #mask with values of ones for the padded parts shape (word_length - 1, batch_size)

        scores ,  (last_hidden , last_cell) = self.forward( input_char_sequence , dec_hidden)
        #scores shape (word_len-1 , batch_size , vocab_size)

        scores = nn.functional.log_softmax(scores , dim = 2) #scores shape (word_len-1 , batch_size , vocab_size)

        #getting the props of the targetted words
        scores_target = torch.gather(scores, 2, index = target_char_sequence.unsqueeze(2)).squeeze(2) #shape (word_len-1 , batch_size )

        #descarding the padding so it doesn't effect the value of loss
        scores_padded = scores_target * target_char_sequence_mask #shape (word_len-1 , batch_size )

        cross_entropy_loss = -scores_padded.sum()

        return cross_entropy_loss





        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        batch_size = initialStates[0].shape[1]

        #intializing the first character to enter the decoder which is a tensor of start tokens in batch_size
        cur_char = torch.tensor([self.target_vocab.start_of_word for _ in range(batch_size)], dtype=torch.long, device=device).unsqueeze(0) #shape (1,batch_size)

        #intializing the tensor that will hold the decoded characters
        word_chars = torch.zeros((max_length , batch_size), dtype=torch.long, device=device) #shape (max_length,batch_size)

        #intializing the decoder states
        dec_state = initialStates

        for index in range(max_length):

            char_scores ,  dec_state = self.forward( cur_char , dec_state)
            #char_scores: shape (1 , batch_size , vocab_size)

            char_scores_squeezed = char_scores.squeeze(0) # shape (batch_size , vocab_size)
            cur_char = char_scores_squeezed.argmax(1) # shape (batch_size)

            #assigning the predicted characters in the tensor
            word_chars[index] = cur_char.contiguous()

            cur_char = cur_char.contiguous().unsqueeze(0)


        word_chars = word_chars.contiguous().transpose(1,0) #shape (batch_size , max_length)

        words = []

        for example in word_chars:

            word = ''

            for char_id in example :

                if char_id == self.target_vocab.end_of_word:
                    break

                word = word + self.target_vocab.id2char[char_id.item()]

            words.append(word)


        return words










        ### END YOUR CODE
