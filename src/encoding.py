"""
Convert card pool to indexes, so they can be encoded by the neural network
"""

import json
import re

class MTGStandardEncoder:
    """"""
    def __init__(self):
        self.cards = []
        self.card_to_index = {}
        self.index_to_card = {}
        self._load_cards('../data/oracle-cards-20230321090221.json')

    def _load_cards(self, filepath):
        with open(filepath, encoding="utf8") as f:
            oracle_cards = json.load(f)

        for card in oracle_cards:
            # pick out only standard legal cards
            if card['legalities']['standard'] == 'legal':
                self.cards.append(card['name'])

        # sort cards alphabetically
        self.cards.sort()
        self.cards.insert(0, 'Blank') # add a blank card place holding in generation

        for i, card in enumerate(self.cards):
            self.card_to_index[card] = i
            self.index_to_card[i] = card

            # for double-sided cards, add another encoding mapping for the shortened version
            # that uses only the first side name (separated by //)
            if '//' in card:
                self.card_to_index[card.split('//')[0].strip()] = i


    def encode(self, card):
        """ convert card name to index """
        return self.card_to_index[card]

    def decode(self, index):
        """ convert index to card name """
        return self.index_to_card[index]

    def __len__(self):
        return len(self.cards)

    def decode_deck(self, encoded_deck):
        """ convert a list of indexes to a standard deck format string """
        # count the number of occurrences of each card id
        card_counts = {}
        for card_id in encoded_deck:
            if card_id in card_counts:
                card_counts[card_id] += 1
            else:
                card_counts[card_id] = 1

        # convert card id to card name
        deck = []
        for card_id, count in card_counts.items():
            deck.append((count, self.decode(card_id)))

        deck.sort(key=lambda x: x[1])  # sort by card name
        deck_str = 'Deck\n'
        for count, card in deck:
            deck_str += '{} {}\n'.format(count, card)

        return deck_str

    def encode_decks(self, deck_str):
        """ convert a standard deck format string to a list of encoded decks """
        # Parse text into decks
        is_sideboard = False
        decks = []
        deck, sideboard = [], []
        for line in deck_str.splitlines():
            # Start a new deck
            if line == 'Deck':
                if deck:
                    decks.append((deck, sideboard))
                is_sideboard = False
                deck = []
                sideboard = []
            elif line == 'Sideboard':
                is_sideboard = True
            elif line == '':
                pass
            else:
                # card lines always follow r'(\d+) (.*)' where g1 is counts and g2 is the card name
                match = re.match(r'(\d+) (.*)', line)
                count = int(match.group(1))
                card = match.group(2)

                if is_sideboard:
                    sideboard.append((count, card))
                else:
                    deck.append((count, card))

        # convert decks to encodings
        enc_decks = []
        for deck, sideboard in decks:
            # ignore sideboard for now
            enc_deck = []
            for count, card in deck:
                enc_deck.extend([self.encode(card)] * count)

            if len(enc_deck) == 60:
                enc_decks.append(enc_deck)
            elif len(enc_deck) > 60:
                print('Throwing out deck with {} cards'.format(len(enc_deck)))
            elif len(enc_deck) < 60:
                # pad deck with blank cards
                enc_deck.extend([0] * (60 - len(enc_deck)))
                enc_decks.append(enc_deck)

        return enc_decks



if __name__ == '__main__':
    enc = MTGStandardEncoder()
    print('Total cards: {}'.format(len(enc.cards)))  # we have 1881 cards + 1 placeholder blank card
    print(enc.encode('Plains'))
    print(enc.decode(0))
    print(enc.decode(1))