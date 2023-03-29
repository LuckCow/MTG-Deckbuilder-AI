import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import encoding
from src.visualize_data import visualize_value_probs

enc = encoding.MTGStandardEncoder()

vocab_size = len(enc)
n_embd = 32
n_head = 4
n_layer = 3
dropout = 0.2
lr = 1e-3
block_size = 60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MTGDeckGenerator(nn.Module):
    """ Simple Transformer encoder model """
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.decoder = nn.Linear(n_embd, vocab_size)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.to(device)


    def forward(self, idx, targets=None):
        # idx is (batch_size, 60, 1882) array of one hot vectors
        # convert 1 hot tokens to embedding domain
        tok_emb = self.token_embedding_table(idx)

        # encode with transformer
        x = self.transformer_encoder(tok_emb)

        # decode with linear layer - no cross attention needed
        logits = self.decoder(x)

        # calculate loss
        # targets are (batch_size, 1882) of probability vectors
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=60):
        # idx is (B, T) array of indices in the current context
        # T is the deck size of 60 cards, containing an arbitrary number of placeholder tokens (0) which will
        # be iteratively replaced by cards recommended by the model

        # first, sort the deck so that blank cards are at the end
        # sort idx by number in each B dimension
        idx, _ = idx.sort(1, True)

        for _ in range(max_new_tokens):
            # find first index of placeholder token (0) in each B dimension
            matches = (idx[0] == 0).nonzero()  # TODO: do for all batches
            if matches.size(0) == 0:
                # no placeholder tokens, so we can't generate any more cards
                return idx
            else:
                # get the index of the first placeholder token
                first_blank_idx = matches[0].item()

            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_nexts = torch.multinomial(probs, num_samples=20, replacement=False) # (B, 1)

            # find the first card that is legal to add to the deck
            for idx_next in idx_nexts[0]:
                next_counts = len((idx[0] == idx_next).nonzero()) + 1
                if enc.is_legal_addition(idx_next.item(), next_counts):
                    # wrap back into (1, 1) tensor
                    idx_next = idx_next.unsqueeze(0).unsqueeze(0)
                    break
            else:
                raise ValueError("Could not find a legal card to add to the deck from first 20 samples")

            # append sampled card into next blank slot
            idx = torch.cat([idx[:, :first_blank_idx], idx_next, idx[:, first_blank_idx+1:]], dim=1)
        return idx

    @torch.no_grad()
    def visualize_generate(self, idx):
        idx, _ = idx.sort(1, True)

        matches = (idx[0] == 0).nonzero()  # TODO: do for all batches
        if matches.size(0) == 0:
            # no placeholder tokens, so we can't generate any more cards
            return idx
        else:
            # get the index of the first placeholder token
            first_blank_idx = matches[0].item()

        # get the predictions
        logits, loss = self(idx)
        # focus only on the last time step
        logits = logits[:, -1, :]  # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)  # (B, C)

        visualize_value_probs(probs[0].cpu().numpy())