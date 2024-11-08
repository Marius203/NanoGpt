import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------------------------------------

# Check if CUDA is available
# if torch.cuda.is_available():
#     print("CUDA is available. Using GPU.")
#     device = torch.device('cuda')
# else:
#     print("CUDA is not available. Using CPU.")
#     device = torch.device('cpu')


class GPTConfig:
    def __init__(self, n_layer=12, n_head=12, n_embd=768, vocab_size=50257, block_size=1024, bias=True, dropout=0.1):
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.bias = bias
        self.dropout = dropout

# ----------------------------------------

class CausalSelfAttention(nn.Module):
    # Multi headed attention
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd 
        # not really a "bias", more of a mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1 , config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimentionality(n_embd)
        # calculate query, key, value for all heads in batch and move head forward to be batch dim
        # query, key and value are 3 vectors emitted from each token
        # query and key multiply each other to determine how relevant they are for each other
        qkv = self.c_attn(x) # qcv = query,key,value (maybe not intuitive)
        q, k, v = qkv.split(self.n_embd, dim = 2)

        k = k.view(B,T,self.n_head, C//self.n_head).transpose(1,2)  #they all have to change dimentions so they become (B, num of heads, T, head size)
        q = q.view(B,T,self.n_head, C//self.n_head).transpose(1,2)  #in order to be able to multiply
        v = v.view(B,T,self.n_head, C//self.n_head).transpose(1,2)

        #attention creates the giant (T,T) matrix of queries and keys
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # This makes sure that the preddiction is not going to be influenced by future characters,
        att = torch.clamp(att, min=-10, max=10)  # Clamp the attention scores to avoid extreme values
        att = F.softmax(att, dim = 1) # only the ones that precede the current one 

        y = att @ v #(B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1,2).contiguous().view(B, T, C) #reassemble head outputs through concatenation
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) 
        self.gelu = nn.GELU(approximate = 'tanh')                           # Smoother ReLU
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 


    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self,config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        """
        Forward pass
        Attention -> tokens communicate
        MLP -> operation that happens to each token, independently of the other tokens 
        """
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

class GPT(nn.Module):
    def __init__ (self,config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # weights of the token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # weights of the position embeddings
            hidden_modules = nn.ModuleList([Block(config) for _ in range (config.n_layer)]), # initialize the layers
            final_layer_norm = nn.LayerNorm(config.n_embd) 
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        # weight sharing for memory efficiency (~ 30% of the parameters would be stored twice otherwise)
        self.transformer.wte.weight = self.lm_head.weight

        # init parameters
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5   
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self,idx, targets=None): #idx = indices of the shape (B,T), T - length sequences, B - independent sequences
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward a vector of length {T}"
        # forward token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device = idx.device)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, num of embeddings)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, num of embeddings)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.hidden_modules:
            x = block(x)
        # forward the final layer norm and the classifier
        x = self.transformer.final_layer_norm(x)
        logits = self.lm_head(x) # shape (B, T, vocab size)
        loss = None 
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
num_return_sequences = 5
max_length = 30
device = 'cuda'

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Prefix tokens
import tiktoken

class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        #load data from input file at initialization
        with open('input.txt','r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.currnet_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.currnet_position : self.currnet_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance position
        self.currnet_position += B*T

        #reset if next batch is out of bounds 
        if self.currnet_position + (B * T + 1) > len(self.tokens):
            self.currnet_position = 0
        return x, y

torch.manual_seed(7)
torch.cuda.manual_seed(7)

train_loader = DataLoader(B = 4, T = 256)

torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig())
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print("all good")