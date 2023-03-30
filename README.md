# MTG-Deckbuilder-AI
Transformer Based Neural Network trained to generate deck lists for the card game Magic the Gathering

## Overview
The cards are represented by an encoding which is a 1xN vector where N is the number of cards in the game plus a placeholder token for a missing card (1882). Each index represents a unique card and the cards are represented by 1-hot vectors, meaning that the index of the card is 1 and all other indices are 0. 

The dataset consists of a list of 46 top performing decks in the standard format from recent tournaments. These deck lists were used to create many training examples. The deck is shuffled, and then one card is removed at a time. Each iteration of this produces a training input-target pair. The input is the remaining cards in the deck and the output is a single vector 1882 length vector with the counts of each removed card in their corresponding index in order to represent the probability of which card should be added next.

The network itself consists of an encoder layer to convert the 1-hot card representation into an embedding space. Then, it is fed through a transformer encoder layer to understand the structure and relationship between the cards. Finally, that memory is decoded by a single linear layer to represent the probability of each potential card to add next. This can then be sampled from to finish out a deck from any starting point.

Figure: Example of the probability distribution for selecting the next card at one step of generation.
![selection_dist1.png](img%2Fselection_dist1.png)

## Example Output
(https://www.moxfield.com/decks/8RqO8KlOhkOgCmiVHAiUjQ)
```
Deck
2 Atraxa, Grand Unifier
4 Blackcleave Cliffs
2 Bloodtithe Harvester
3 Corpse Appraiser
4 Darkslick Shores
4 Fable of the Mirror-Breaker // Reflection of Kiki-Jiki
4 Go for the Throat
1 Haunted Ridge
1 Island
1 Make Disappear
1 Mountain
3 Otawara, Soaring City
1 Overgrown Farmland
1 Phoenix Chick
3 Plains
1 Razorlash Transmogrant
4 Reckoner Bankbuster
1 Roadside Reliquary
2 Seachrome Coast
1 Shattered Sanctum
4 Sheoldred, the Apocalypse
1 Shipwreck Marsh
2 Shivan Reef
1 Sokenzan, Crucible of Defiance
1 Stormcarved Coast
1 Sundown Pass
2 Swamp
1 Thalia, Guardian of Thraben
1 The Cruelty of Gix
2 Xander's Lounge
```

