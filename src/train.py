import model
import numpy as np
import torch

from src.encoding import MTGStandardEncoder

enc = MTGStandardEncoder()
mtgDeckGenerator = model.MTGDeckGenerator()
device = model.device

# print the number of parameters in the model
print(sum(p.numel() for p in mtgDeckGenerator.parameters()), ' parameters')

# load the data, should be (7290, 60) and (7290, 1882)
# train_inputs = np.fromfile('../data/train_inputs.bin', dtype=np.uint16)
# train_values = np.fromfile('../data/train_values.bin', dtype=np.uint16)
# val_inputs = np.fromfile('../data/val_inputs.bin', dtype=np.uint16)
# val_values = np.fromfile('../data/val_values.bin', dtype=np.uint16)
train_inputs = np.load('../data/train_inputs.npy')
train_values = np.load('../data/train_values.npy')
val_inputs = np.load('../data/val_inputs.npy')
val_values = np.load('../data/val_values.npy')

print(train_inputs.shape, train_values.shape)

batch_size = 8
max_iters = 1000
eval_interval = max_iters // 10
eval_iters = 200

def get_batch(split):
    if split == 'train':
        inputs = train_inputs
        values = train_values
    else:
        inputs = val_inputs
        values = val_values

    # sample a random batch of data

    indices = np.random.randint(0, len(inputs), batch_size)
    xb = inputs[indices]
    yb = values[indices]

    # convert to tensors
    xb = torch.from_numpy((xb).astype(np.int64)).to(device)
    yb = torch.from_numpy((yb).astype(np.int64)).to(device)

    return xb, yb

@torch.no_grad()
def estimate_loss():
    out = {}
    mtgDeckGenerator.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = mtgDeckGenerator(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    mtgDeckGenerator.train()
    return out

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = mtgDeckGenerator(xb, yb)
    mtgDeckGenerator.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    mtgDeckGenerator.optimizer.step()

# generate from the model
context = torch.zeros((1, 60), dtype=torch.long, device=device)
print(enc.decode_deck(mtgDeckGenerator.generate(context)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))