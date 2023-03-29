"""
Transform the raw data into training data for the neural network
"""
import re
import numpy as np
from encoding import MTGStandardEncoder
import random

random.seed(1337)

with open('../data/decks.txt') as f:
    raw_deck_txt = f.read()

encoder = MTGStandardEncoder()

# convert decks to card indexes
enc_decks = encoder.encode_decks(raw_deck_txt)

# print(enc_decks[0])
# print(decks[0])

# construct test cases by removing cards from the deck, where the value if a probability distribution of removed cards
deck_size = 60
inputs, values = [], []
for deck in enc_decks:
    # remove cards one at a time and make an example each time all the way to 0 cards
    # repeat this process 3 times to make different examples with shuffling
    for _ in range(3):
        shuffled_deck = deck.copy()
        random.shuffle(shuffled_deck)
        removed_cards = []
        for i in range(deck_size - 1, -1, -1):  # blank cards are at the end
            removed_cards.append(shuffled_deck[i])
            shuffled_deck[i] = 0

            # create values probability distribution from removed cards
            values_dist = np.zeros(len(encoder))
            for card in removed_cards:
                values_dist[card] += 1
            #values_dist /= len(removed_cards)  # cross entropy takes counts not probabilities

            # store training inputs/values
            inputs.append(shuffled_deck.copy())
            values.append(values_dist)

# shuffle the training examples
combined = list(zip(inputs, values))
random.shuffle(combined)
inputs, values = zip(*combined)

# split into training and validation sets
n = len(inputs)
train_inputs = inputs[:int(n*0.9)]
train_values = values[:int(n*0.9)]
val_inputs = inputs[int(n*0.9):]
val_values = values[int(n*0.9):]


# save training examples to file (as binary)
train_inputs = np.array(train_inputs, dtype=np.uint16)
val_inputs = np.array(val_inputs, dtype=np.uint16)
# train_inputs.tofile('../data/train_inputs.bin')
# val_inputs.tofile('../data/val_inputs.bin')
np.save('../data/train_inputs.npy', train_inputs)
np.save('../data/val_inputs.npy', val_inputs)

train_values = np.array(train_values, dtype=np.uint16)
val_values = np.array(val_values, dtype=np.uint16)
# train_values.tofile('../data/train_values.bin')
# val_values.tofile('../data/val_values.bin')
np.save('../data/train_values.npy', train_values)
np.save('../data/val_values.npy', val_values)

print(train_inputs.shape, train_values.shape)