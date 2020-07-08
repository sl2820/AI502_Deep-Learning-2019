import torch
import torch.nn as nn
from calculate_bleu import*
import spacy
from google_bleu import *  # did not work

# Language Tokenizers
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')
spacy_fr = spacy.load('fr')
# German forward
def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

# German Reversed
def tokenize_de_rev(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

# English forward
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

# English Reversed
def tokenize_fr_rev(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]


def get_words (input, vocabs, tag = 's'):
    temp = input.permute(1, 0)
    if tag == 's':
        words = [[vocabs.vocab.itos[word] for word in sentence] for sentence in temp]
    else:
        words = [[vocabs.vocab.itos[word] for word in sentence] for sentence in temp]
    words = [' '.join(s) for s in words]
    words = [s.split('<eos>')[0] for s in words]
    return words


def train(model, iterator, optimizer, criterion, clip, src_vocab,trg_vocab):
    model.train()
    epoch_loss = 0
    counter = 0
    bleu = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg)

        # Remove <sos> tag
        output = output[1:]
        trg = trg[1:]
        prediction_embed = torch.argmax(output,-1)

        src_words = get_words(src, src_vocab, 's')
        trg_words = get_words(trg, trg_vocab, 't')
        prediction_words = get_words(prediction_embed,trg_vocab, 't')
        bleu += get_bleu(prediction_words, trg_words)*len(trg_words)
        counter +=len(trg_words)

        output = output.view(-1, output.shape[-1])
        trg = trg.view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator), bleu / counter


def evaluate(model, iterator, criterion, src_vocab,trg_vocab):
    model.eval()
    epoch_loss = 0
    counter = 0
    bleu = 0
    sources = []
    targets = []
    predictions = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg)
            # Remove <sos> tag
            output = output[1:]
            trg = trg[1:]
            src = src[1:]
            prediction_embed = torch.argmax(output, -1)

            src_words = get_words(src,src_vocab, 's')
            trg_words = get_words(trg,trg_vocab, 't')
            prediction_words = get_words(prediction_embed, trg_vocab,'t')
            bleu += get_bleu(prediction_words, trg_words) * len(trg_words)

            counter += len(trg_words)

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
            sources.append(src_words)
            targets.append(trg_words)
            predictions.append(prediction_words)

    return epoch_loss / len(iterator), bleu / counter, sources, targets, predictions

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
