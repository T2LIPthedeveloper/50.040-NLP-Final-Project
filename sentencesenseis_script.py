import os
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from d2l import torch as d2l
import spacy
import gensim
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import sentencepiece as spm
import nltk
import random
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

devices = d2l.try_all_gpus()
print(f"Device: {devices}")

# Load GloVe vectors
glove_input_file = './data/glove.6B.200d/glove.6B.200d.txt'
word2vec_output_file = './data/glove.6B.200d/glove.6B.200d.word2vec.txt'

if not os.path.exists(word2vec_output_file):
    glove2word2vec(glove_input_file, word2vec_output_file)

word_vectors = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
spacy_en = spacy.load('en_core_web_sm')

def tokenizer(text):
    return [
        tok.text.lower() 
        for tok in spacy_en.tokenizer(text) 
        if not tok.is_punct and not tok.is_space
    ]

def read_imdb(data_dir, is_train):
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test', label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

def load_data_imdb(batch_size, num_steps=500):
    data_dir = os.path.join('.', 'data', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    
    train_tokens = [tokenizer(review) for review in train_data[0]]
    test_tokens = [tokenizer(review) for review in test_data[0]]
    
    vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
    
    train_features = torch.tensor([
        d2l.truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in train_tokens
    ])
    test_features = torch.tensor([
        d2l.truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in test_tokens
    ])
    
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size, is_train=False)
    return train_iter, test_iter, vocab

batch_size = 64
train_iter, test_iter, vocab = load_data_imdb(batch_size)
embedding_dim = 200
embedding_matrix = np.zeros((len(vocab), embedding_dim))

unk_count = 0
for i, token in enumerate(vocab.idx_to_token):
    if token in word_vectors:
        embedding_matrix[i] = word_vectors[token]
    else:
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
        unk_count += 1

print(f"Number of OOV words: {unk_count} out of {len(vocab)}")

embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

class HybridCNNRNN200(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 lstm_hidden_size, num_lstm_layers, dropout=0.5, **kwargs):
        super(HybridCNNRNN200, self).__init__(**kwargs)
        # Embedding Layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Constant Embedding Layers
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        # Dropout Layer
        self.dropout = nn.Dropout(dropout)
        # Convolutional Layers
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            padding = (k - 1) // 2
            self.convs.append(
                nn.Conv1d(
                    in_channels=2 * embed_size,
                    out_channels=c,
                    kernel_size=k,
                    padding=padding
                )
            )
        # ReLU Activation
        self.relu = nn.ReLU()
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=sum(num_channels),
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True
        )
        # Attention Layer
        self.attention = nn.Linear(2 * lstm_hidden_size, 1)
        # Fully Connected Layer
        self.decoder = nn.Linear(2 * lstm_hidden_size, 2)

    def forward(self, inputs):
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        embeddings = self.dropout(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = self.relu(conv(embeddings))
            conv_outputs.append(conv_out)
        conv_outputs = torch.cat(conv_outputs, dim=1)
        conv_outputs = conv_outputs.permute(0, 2, 1)

        lstm_out, _ = self.lstm(conv_outputs)
        
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), dim=1)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        outputs = self.decoder(self.dropout(context_vector))
        return outputs

embed_size = 200
kernel_sizes = [3, 5, 7]
num_channels = [100, 100, 100]
lstm_hidden_size = 150
num_lstm_layers = 2
dropout = 0.5
devices = d2l.try_all_gpus()

net = HybridCNNRNN200(
    vocab_size=len(vocab),
    embed_size=embed_size,
    kernel_sizes=kernel_sizes,
    num_channels=num_channels,
    lstm_hidden_size=lstm_hidden_size,
    num_lstm_layers=num_lstm_layers,
    dropout=dropout
)
net.apply(init_weights)
net = net.to(devices[0])

net.embedding.weight.data.copy_(embedding_matrix)
net.constant_embedding.weight.data.copy_(embedding_matrix)
net.constant_embedding.weight.requires_grad = False

if not os.path.exists("./model/original/HybridCNNRNN200/best/model.pth"):
    lr, num_epochs = 0.0005, 10
    trainer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    loss = nn.CrossEntropyLoss(reduction='none')
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
    os.makedirs("./model/original/HybridCNNRNN200", exist_ok=True)
    torch.save(net, "./model/original/HybridCNNRNN200/model.pth")
else:
    net = torch.load("./model/original/HybridCNNRNN200/best/model.pth").to(device)

def predict_sentiment(net, vocab, sequence):
    sequence = tokenizer(sequence)
    sequence = torch.tensor(vocab[sequence], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'

def cal_metrics(net, test_iter, test_texts, locs):
    net.eval()
    device = next(net.parameters()).device

    all_preds = []
    all_labels = []
    all_texts = []

    sample_idx = 0

    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(device)
            y = y.to(device)

            outputs = net(X)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            batch_size = X.size(0)
            for i in range(batch_size):
                all_texts.append(test_texts[sample_idx])
                sample_idx += 1

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    F1_Score = f1_score(all_labels, all_preds)

    print(f'Accuracy:  {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1 Score:  {F1_Score:.4f}')

    results = pd.DataFrame({
        'Text': all_texts,
        'True Label': all_labels,
        'Predicted Label': all_preds
    })
    results.to_csv(locs, index=False)

    return F1_Score, precision, recall, accuracy

if not os.path.exists("./model/original/HybridCNNRNN200/best/test-result.txt"):
    data_dir = os.path.join('.', 'data', 'aclImdb')
    test_data = read_imdb(data_dir, False)
    F1_Score, precision, recall, accuracy = cal_metrics(net, test_iter, test_data[0], './model/original/HybridCNNRNN200/prediction_results.csv')
else:
    with open("./model/original/HybridCNNRNN200/best/test-result.txt") as file:
        content = file.read()
    print(content)

def read_csv_dataset(file_path):
    dataset = pd.read_csv(file_path)
    data = dataset['text'].tolist()
    labels = dataset['label'].tolist()
    return data, labels

test_data = read_csv_dataset('./data/test_data_movie.csv')
num_steps = 500

test_tokens = [tokenizer(review) for review in test_data[0]]
test_features = torch.tensor([
    d2l.truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in test_tokens
])
test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                           batch_size, is_train=False)

F1_Score, precision, recall, accuracy = cal_metrics(net, test_iter, test_data[0], 'results.csv')
