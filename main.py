import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)

        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(context)

        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class BasicBERT(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=6, d_ff=3072, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = PositionalEncoding(d_model, max_len)
        self.segment_embedding = nn.Embedding(2, d_model)  # For sentence pairs

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Output heads
        self.mlm_head = nn.Linear(d_model, vocab_size)  # Masked Language Modeling
        self.nsp_head = nn.Linear(d_model, 2)  # Next Sentence Prediction

        # Special tokens
        self.cls_token_id = 0  # [CLS]
        self.sep_token_id = 1  # [SEP]
        self.mask_token_id = 2  # [MASK]
        self.pad_token_id = 3  # [PAD]

    def create_padding_mask(self, x):
        return (x != self.pad_token_id).unsqueeze(1).unsqueeze(2)

    def forward(self, input_ids, segment_ids=None, masked_positions=None):
        batch_size, seq_len = input_ids.size()

        # Create embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(token_emb)

        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        seg_emb = self.segment_embedding(segment_ids)

        # Combine embeddings
        embeddings = token_emb + pos_emb + seg_emb
        embeddings = self.dropout(embeddings)

        # Create attention mask
        attention_mask = self.create_padding_mask(input_ids)

        # Pass through transformer layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)

        # Output heads
        mlm_logits = self.mlm_head(hidden_states)
        cls_output = hidden_states[:, 0]  # [CLS] token representation
        nsp_logits = self.nsp_head(cls_output)

        return {
            'last_hidden_state': hidden_states,
            'mlm_logits': mlm_logits,
            'nsp_logits': nsp_logits,
            'cls_output': cls_output
        }

    def get_embeddings(self, input_ids, segment_ids=None):
        """Get contextualized embeddings for input tokens"""
        with torch.no_grad():
            outputs = self.forward(input_ids, segment_ids)
            return outputs['last_hidden_state']

# Example usage and training utilities
class BERTTrainer:
    def __init__(self, model, vocab_size):
        self.model = model
        self.vocab_size = vocab_size
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.nsp_criterion = nn.CrossEntropyLoss()

    def mask_tokens(self, inputs, mask_prob=0.15):
        """Apply random masking for MLM training"""
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, mask_prob)

        # Don't mask special tokens
        special_tokens = [0, 1, 2, 3]  # [CLS], [SEP], [MASK], [PAD]
        for token_id in special_tokens:
            probability_matrix.masked_fill_(inputs == token_id, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80% of the time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.model.mask_token_id

        # 10% of the time, replace with random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def compute_loss(self, outputs, mlm_labels, nsp_labels):
        mlm_loss = self.mlm_criterion(outputs['mlm_logits'].view(-1, self.vocab_size), mlm_labels.view(-1))
        nsp_loss = self.nsp_criterion(outputs['nsp_logits'], nsp_labels)
        total_loss = mlm_loss + nsp_loss
        return total_loss, mlm_loss, nsp_loss

# Example initialization
def create_bert_model(vocab_size=30000, device='cpu'):
    """Create a basic BERT model with reasonable defaults"""
    model = BasicBERT(
        vocab_size=vocab_size,
        d_model=768,
        num_heads=12,
        num_layers=6,
        d_ff=3072,
        max_len=512,
        dropout=0.1
    )
    return model.to(device)  # Move model to specified device

# Demo usage
if __name__ == "__main__":
    # Set device to CPU
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create model
    vocab_size = 30000
    model = create_bert_model(vocab_size)
    model = model.to(device)  # Move model to CPU

    # Example input (batch_size=2, seq_len=10)
    input_ids = torch.randint(4, vocab_size, (2, 10), device=device)  # Create tensor on CPU
    input_ids[:, 0] = 0  # Set first token to [CLS]

    # Forward pass
    outputs = model(input_ids)

    print("Model created successfully!")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output hidden states shape: {outputs['last_hidden_state'].shape}")
    print(f"MLM logits shape: {outputs['mlm_logits'].shape}")
    print(f"NSP logits shape: {outputs['nsp_logits'].shape}")
    print(f"CLS output shape: {outputs['cls_output'].shape}")

    # Example of getting embeddings
    embeddings = model.get_embeddings(input_ids)
    print(f"Embeddings shape: {embeddings.shape}")