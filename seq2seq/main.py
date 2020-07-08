import torch
import torch.nn as nn
import torch.optim as optim
# Dataset
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from model import*
import random
import time
from utils import*


# Random initialization
SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

source = Field(tokenize = tokenize_de_rev, init_token = '<sos>', eos_token = '<eos>', lower = True)
# source = Field(tokenize = tokenize_fr_rev, init_token = '<sos>', eos_token = '<eos>', lower = True)
target = Field(tokenize = tokenize_en, init_token = '<sos>',eos_token = '<eos>', lower = True)
train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),fields = (source, target))
# train_data, valid_data, test_data = Multi30k.splits(exts = ('.fr', '.en'),fields = (source, target))

# See total data size
print("Number of training examples: %d"%(len(train_data.examples)))
print("Number of validation examples: %d"%(len(valid_data.examples)))
print("Number of testing examples: %d"%(len(test_data.examples)))

# Build Vocabulary for each source and target
source.build_vocab(train_data, min_freq = 2)
target.build_vocab(train_data, min_freq = 2)
print("Unique Words in source vocabulary: %d"%(len(source.vocab)))
print("Unique Words in target vocabulary: %d"%(len(target.vocab)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
INPUT_DIM = len(source.vocab)
OUTPUT_DIM = len(target.vocab)
ENC_EMB_DIM = 1000   #1000, 256
DEC_EMB_DIM = 1000   #1000, 256
HID_DIM = 1000       #1000, 512
N_LAYERS = 4         #4, 2
N_EPOCHS = 8         #  following the paper
CLIP = 1

train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size = BATCH_SIZE, device = device)
encoder_multi = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS)
decoder_multi = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS)
model = seq2seq(encoder_multi, decoder_multi, device).to(device)

# encoder_dropout = Encoder_with_dorpout(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS)
# decoder_dropout = Decoder_with_dropout(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS)
# model = seq2seq(encoder_dropout, decoder_dropout, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)

parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Model parameters: %d'%(parameters))

optimizer = optim.Adam(model.parameters())
PAD_IDX = target.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_bleu = train(model, train_iterator, optimizer, criterion, CLIP, source, target)
    valid_loss, valid_bleu, srcs, tars, preds = evaluate(model, valid_iterator, criterion, source, target)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'Epoch: {epoch + 1} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Train BLEU: {train_bleu:7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} | Val. BLEU: {valid_bleu:7.3f}')



# Final testing
model.load_state_dict(torch.load('tut1-model.pt'))
test_loss, test_bleu, test_srcs, test_tars, test_preds = evaluate(model, test_iterator, criterion, source, target)
print(f'==> Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | Test BLEU: {test_bleu:7.3f}')


print("Followings are %d samples from test cases" %(3))
for i in range(3):
    print("=========== Case %d ============" % (i + 1))
    print("\tOriginal Source :")
    print(test_srcs[i])
    print("\tGround Truth    : ")
    print(test_tars[i])
    print("\tOur Predictions : ")
    print(test_preds[i])


if __name__ == "__main__":
    print("End of running")