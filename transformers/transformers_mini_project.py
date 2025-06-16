# Step-by-step Breakdown with Dummy Data
# Step 1: Tokenization and Input Embedding
# We turn words into numbers (vectors).

import torch
import torch.nn as nn

# Dummy vocabulary and sentence
vocab = {"I": 0, "am": 1, "GPT": 2, "model": 3, "pad": 4}
sentence = ["I", "am", "GPT"]

# Step 1: Convert words to token indices
tokens = torch.tensor([vocab[word] for word in sentence])  # [0, 1, 2]
print("Tokens:", tokens)

# Step 2: Create embedding layer (vocab_size=5, embed_dim=4)
embedding = nn.Embedding(num_embeddings=5, embedding_dim=4)
embedded = embedding(tokens)  # Shape: [3, 4]
print("Embedded:\n", embedded)


# # Step 2: Add Positional Encoding
# # Since transformers don’t use RNNs, they need a way to know word order.

import math

def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
            if i+1 < d_model:
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * i)/d_model)))
    return pe

pos_enc = positional_encoding(seq_len=3, d_model=4)
print("Positional Encoding:\n", pos_enc)

# Add embedding + pos encoding
x = embedded + pos_enc
print("Input to Transformer:\n", x)

# # Step 3: Self-Attention (Single Head)
# # Key formula:

# # Attention(Q, K, V) = softmax(QKᵀ / √d_k) V

d_model = 4
# Initialize weights for Q, K, V
W_Q = nn.Linear(d_model, d_model, bias=False)
W_K = nn.Linear(d_model, d_model, bias=False)
W_V = nn.Linear(d_model, d_model, bias=False)

Q = W_Q(x)  # Shape: [3, 4]
K = W_K(x)
V = W_V(x)

# Compute attention scores
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model)
attn_weights = torch.softmax(scores, dim=-1)
output = torch.matmul(attn_weights, V)

print("Attention Weights:\n", attn_weights)
print("Attention Output:\n", output)


# Step 4: Add & Norm + Feed Forward
# This is a residual connection + layer norm + small neural net.

# # Add & Norm
# x = x + output
# norm1 = nn.LayerNorm(d_model)
# x = norm1(x)

# # Feed Forward
# ff = nn.Sequential(
#     nn.Linear(d_model, d_model * 2),
#     nn.ReLU(),
#     nn.Linear(d_model * 2, d_model)
# )

# ff_out = ff(x)

# # Add & Norm again
# x = x + ff_out
# x = norm1(x)  # Re-use same norm for demo

# print("Final Encoder Output:\n", x)
