from src.encoding import MTGStandardEncoder
import model
import torch

load_path = 'model_a1.pt'

mtgDeckGenerator = model.MTGDeckGenerator()
mtgDeckGenerator.load_state_dict(torch.load(load_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = MTGStandardEncoder()

partial_deck = """Deck
13 Plains
4 Reckoner Bankbuster
2 Roadside Reliquary
2 Sanctuary Warden
2 Serra Paragon
2 Spirited Companion
2 The Eternal Wanderer
3 The Restoration of Eiganjo
4 The Wandering Emperor
4 Wedding Announcement
"""
partial_deck_enc = enc.encode_decks(partial_deck)
# convert lists to 2d tensor
partial_deck_enc = torch.tensor(partial_deck_enc, dtype=torch.long, device=device)
print(enc.decode_deck(mtgDeckGenerator.generate(partial_deck_enc)[0].tolist()))

# generate from the model
# add as many cards as you want to the starting deck, leave the rest as placeholders (id: 0),
# then the model will fill in the blanks
blank_deck = torch.zeros((1, 60), dtype=torch.long, device=device)
print(enc.decode_deck(mtgDeckGenerator.generate(blank_deck)[0].tolist()))