import torch.nn as nn
import torch
from weightdrop import WeightDrop

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.4, dropouth=0.25, dropouti=0.4, dropoute=0.1, wdrop=0.5, tie_weights = True):
        super(RNNModel, self).__init__()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
        if wdrop:
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]

        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self, input, hidden=None):
        # Assuming input is a tensor representing multiple sequences
        # input shape: (sequence_length, batch_size)
    
        # Embedding layer
        emb = self.encoder(input)
        
        # LSTM layers
        raw_output = emb
        for l, rnn in enumerate(self.rnns):
            if isinstance(rnn, torch.nn.LSTM):
                rnn.flatten_parameters()
            raw_output, new_h = rnn(raw_output, hidden[l])
        
        # Output layer
        output = self.decoder(raw_output[-1])  # Use only the last output
        
        # Softmax to get probabilities
        output_probabilities = torch.softmax(output, dim=1)
        
        # Get top-K predicted word indices
        K = 2
        topk_values, topk_indices = torch.topk(output_probabilities, K, dim=1)

        # Get index of word with highest probability
        _, predicted_word_indices = torch.max(output_probabilities, 1)
        
        return topk_indices, predicted_word_indices


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                for l in range(self.nlayers)]
    
    
    def locked_dropout(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        mask = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x
    
    
    def embedded_dropout(self, embed, words, dropout=0.1, scale=None):
        if dropout:
            mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
            masked_embed_weight = mask * embed.weight
        else:
            masked_embed_weight = embed.weight
        if scale:
            masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

        padding_idx = embed.padding_idx
        if padding_idx is None:
            padding_idx = -1

        X = torch.nn.functional.embedding(words, masked_embed_weight,
        padding_idx, embed.max_norm, embed.norm_type,
        embed.scale_grad_by_freq, embed.sparse
        )
        return X