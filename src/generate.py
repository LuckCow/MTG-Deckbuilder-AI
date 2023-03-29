from src.encoding import MTGStandardEncoder
import model
import torch

load_path = 'model2_a1.pt'

mtgDeckGenerator = model.MTGDeckGenerator()
mtgDeckGenerator.load_state_dict(torch.load(load_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = MTGStandardEncoder()

"""2 Cathar Commando
4 Demolition Field
2 Destroy Evil
1 Eiganjo, Seat of the Empire
1 Elspeth Resplendent
2 Farewell
4 Field of Ruin
4 Lay Down Arms
2 Loran of the Third Path"""
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
mtgDeckGenerator.visualize_generate(partial_deck_enc)
print(enc.decode_deck(mtgDeckGenerator.generate(partial_deck_enc)[0].tolist()))


"""3 Lightning Strike
6 Mountain
2 Play with Fire
4 Bloodthirsty Adversary
4 Go for the Throat"""
partial_deck2 = """Deck
1 Razorlash Transmogrant
4 Bloodtithe Harvester
4 Fable of the Mirror-Breaker
4 Blackcleave Cliffs
1 Takenuma, Abandoned Mire
1 Sokenzan, Crucible of Defiance
4 Sulfurous Springs
4 Haunted Ridge
3 Swamp
3 Archfiend of the Dross
2 Cut Down
4 Monastery Swiftspear
2 Squee, Dubious Monarch
4 Kumano Faces Kakkazan"""
partial_deck_enc2 = enc.encode_decks(partial_deck2)
# convert lists to 2d tensor
partial_deck_enc2 = torch.tensor(partial_deck_enc2, dtype=torch.long, device=device)
mtgDeckGenerator.visualize_generate(partial_deck_enc2)
print(enc.decode_deck(mtgDeckGenerator.generate(partial_deck_enc2)[0].tolist()))

# generate from the model
# add as many cards as you want to the starting deck, leave the rest as placeholders (id: 0),
# then the model will fill in the blanks
blank_deck = torch.zeros((1, 60), dtype=torch.long, device=device)
print(enc.decode_deck(mtgDeckGenerator.generate(blank_deck)[0].tolist()))

mtgDeckGenerator.visualize_generate(blank_deck)