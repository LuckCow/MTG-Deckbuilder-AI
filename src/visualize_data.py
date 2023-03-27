"""
Check the data and outputs for debugging
"""

import numpy as np
import matplotlib.pyplot as plt

from src.encoding import MTGStandardEncoder

train_inputs = np.load('../data/train_inputs.npy')
train_values = np.load('../data/train_values.npy')
val_inputs = np.load('../data/val_inputs.npy')
val_values = np.load('../data/val_values.npy')

enc = MTGStandardEncoder()

def visualize_value_probs(prob_dist):
    """Visualize the probability of each card in the value row. only show cards that have a non-zero probability"""

    # get indices of non-zero probabilities and sort them in descending order of probability
    nonzero_indices = np.where(prob_dist > 0)[0]
    sorted_indices = np.argsort(prob_dist[nonzero_indices])[::-1]
    sorted_nonzero_indices = nonzero_indices[sorted_indices]

    # get the corresponding non-zero probabilities and outcomes
    nonzero_probs = prob_dist[sorted_nonzero_indices]

    # convert the card indices to card names
    decode = lambda i: enc.decode(i) if len(enc.decode(i)) < 18 else enc.decode(i)[:18] + '-'
    outcomes = [decode(i) for i in sorted_nonzero_indices]
    #outcomes = [str(i) for i in sorted_nonzero_indices]

    print(outcomes, nonzero_probs)

    # plot a bar chart of the non-zero probabilities
    fig = plt.figure(figsize=(8, 8))
    plt.bar(range(len(nonzero_probs)), nonzero_probs)
    plt.xticks(range(len(nonzero_probs)), outcomes, rotation=90)
    plt.xlabel('Card')
    plt.ylabel('Probability')
    plt.title('Probability Distribution')
    plt.subplots_adjust(bottom=0.2)  # adjust the margin at the bottom for card names
    plt.show()

if __name__ == '__main__':
    # visualize the probability of each card in the value row


    for v in train_values[:10]:
        visualize_value_probs(v)

