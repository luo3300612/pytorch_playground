import torch
import torchtext
from torchtext.datasets import text_classification
from .model import myLSTM

NGRAMS = 2
import os

if not os.path.isdir('./.data'):
    os.mkdir('./.data')
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root='./.data', ngrams=NGRAMS, vocab=None)
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUN_CLASS = len(train_dataset.get_labels())

model = myLSTM(input_size=128, hidden_size=128, vocab_size=VOCAB_SIZE, embedding_dim=128, max_length=20,
               batch_size=batch_size, num_classes=NUN_CLASS).to(device)
