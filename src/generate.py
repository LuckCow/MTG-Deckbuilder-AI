from src.encoding import MTGStandardEncoder
import model
import torch

load_path = 'model2.pt'

mtgDeckGenerator = model.MTGDeckGenerator()
mtgDeckGenerator.load_state_dict(torch.load(load_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = MTGStandardEncoder()

# generate from the model
# add as many cards as you want to the starting deck, leave the rest as placeholders (id: 0),
# then the model will fill in the blanks
starting_deck = torch.zeros((1, 60), dtype=torch.long, device=device)
print(enc.decode_deck(mtgDeckGenerator.generate(starting_deck)[0].tolist()))