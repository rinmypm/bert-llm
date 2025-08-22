# Basic BERT Language Model - Complete Study Guide

A simplified PyTorch implementation of BERT (Bidirectional Encoder Representations from Transformers) for educational purposes. This comprehensive guide covers both the implementation and the theoretical foundations of BERT, optimized for CPU usage and designed as a complete learning resource.

## 📚 Table of Contents

1. [Quick Start](#-quick-start)
2. [Understanding BERT](#-understanding-bert)
3. [Architecture Deep Dive](#-architecture-deep-dive)
4. [Training Methodology](#-training-methodology)
5. [Implementation Guide](#-implementation-guide)
6. [Performance & Optimization](#-performance--optimization)
7. [Advanced Topics](#-advanced-topics)
8. [Study Exercises](#-study-exercises)

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv bert_env

# Activate virtual environment
# On Windows:
bert_env\Scripts\activate
# On macOS/Linux:
source bert_env/bin/activate

# Install CPU-only PyTorch (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy>=1.21.0

# Or install from requirements.txt
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

### Basic Usage

```python
import torch
from bert_model import create_bert_model

# Create model on CPU
device = torch.device('cpu')
vocab_size = 30000
model = create_bert_model(vocab_size, device='cpu')

# Prepare input
input_ids = torch.randint(4, vocab_size, (2, 10), device=device)  # Batch size 2, sequence length 10
input_ids[:, 0] = 0  # Set first token to [CLS]

# Forward pass
outputs = model(input_ids)

print(f"Hidden states: {outputs['last_hidden_state'].shape}")  # [2, 10, 768]
print(f"MLM logits: {outputs['mlm_logits'].shape}")           # [2, 10, 30000]
print(f"NSP logits: {outputs['nsp_logits'].shape}")           # [2, 2]
```

## 🧠 Understanding BERT

### What is BERT?

**BERT (Bidirectional Encoder Representations from Transformers)** revolutionized natural language processing by introducing true bidirectional understanding. Unlike previous models that read text sequentially (left-to-right or right-to-left), BERT reads in both directions simultaneously.

### Key Innovations

#### 1. **Bidirectional Context**
Traditional language models process text sequentially:
```
"The cat sat on the [MASK]"
Left-to-right: Uses "The cat sat on the" to predict [MASK]
Right-to-left: Uses "mat" to predict [MASK] (if we knew the end)
```

BERT processes bidirectionally:
```
"The cat sat on the [MASK]"
BERT: Uses both "The cat sat on the" AND "mat" to predict [MASK]
```

#### 2. **Pre-training + Fine-tuning Paradigm**
- **Pre-training**: Learn general language representations on large unlabeled corpus
- **Fine-tuning**: Adapt to specific tasks with minimal task-specific architecture changes

#### 3. **Transformer Architecture**
Built entirely on attention mechanisms, allowing parallel processing and better long-range dependencies.

### Historical Context

#### Before BERT (Sequential Models)
- **Word2Vec (2013)**: Static word embeddings, no context
- **ELMo (2018)**: Bidirectional but shallow, concatenated representations
- **GPT-1 (2018)**: Transformer-based but unidirectional (left-to-right)

#### BERT's Breakthrough (2018)
- **True bidirectionality**: Deep bidirectional representations
- **Transfer learning**: Pre-train once, fine-tune for many tasks
- **State-of-the-art results**: Achieved new records on 11 NLP tasks

#### After BERT
- **RoBERTa (2019)**: Optimized BERT training
- **ALBERT (2019)**: Parameter sharing for efficiency
- **DeBERTa (2020)**: Disentangled attention mechanisms

## 🏗️ Architecture Deep Dive

### Overall Structure

```
Input: "Hello [MASK] world"
    ↓
Token Embeddings: [2, 103, 1000]
    ↓
+ Position Embeddings: [0, 1, 2]
    ↓
+ Segment Embeddings: [0, 0, 0]
    ↓
Transformer Layer 1
    ↓
Transformer Layer 2
    ↓
... (6-12 layers)
    ↓
Output: Contextualized representations
```

### 1. Embedding Layer

#### Token Embeddings
- **Purpose**: Convert discrete tokens to continuous vectors
- **Size**: `[vocab_size, hidden_size]` (e.g., 30,000 × 768)
- **Learning**: Learned during training

```python
# Token "hello" (ID: 7592) → 768-dimensional vector
token_emb = self.token_embedding(input_ids)  # [batch, seq_len, hidden_size]
```

#### Position Embeddings
- **Problem**: Transformers have no inherent sequence order
- **Solution**: Add positional information to each token
- **BERT approach**: Learned position embeddings (vs. sinusoidal in original Transformer)

```python
# Position 0, 1, 2, ... → position-specific vectors
pos_emb = self.position_embedding(token_emb)
```

#### Segment Embeddings
- **Purpose**: Distinguish between sentence A and sentence B
- **Usage**: Next Sentence Prediction task
- **Values**: 0 for sentence A, 1 for sentence B

```python
# All tokens from sentence A get segment ID 0
# All tokens from sentence B get segment ID 1
seg_emb = self.segment_embedding(segment_ids)
```

### 2. Transformer Blocks

#### Multi-Head Self-Attention

**Core Concept**: Each token attends to all other tokens in the sequence.

```python
# For each token, compute attention with every other token
# "The cat sat on the mat"
# Token "cat" attends to: "The"(0.1), "cat"(0.6), "sat"(0.2), "on"(0.05), "the"(0.03), "mat"(0.02)
```

**Mathematical Foundation**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V

Where:
Q = Query matrix (what we're looking for)
K = Key matrix (what to compare against)  
V = Value matrix (what information to extract)
d_k = dimension of key vectors (for scaling)
```

**Multi-Head Mechanism**:
- **Purpose**: Capture different types of relationships
- **Heads**: 12 parallel attention mechanisms
- **Each head**: Focuses on different aspects (syntax, semantics, etc.)

```python
# Head 1: Might focus on subject-verb relationships
# Head 2: Might focus on adjective-noun relationships
# Head 3: Might focus on long-range dependencies
# ... combine all heads for final representation
```

#### Feed-Forward Network
- **Purpose**: Add non-linearity and transform representations
- **Structure**: Linear → GELU → Linear
- **Size**: Typically 4× hidden size (768 → 3072 → 768)

#### Residual Connections & Layer Normalization
- **Residual**: `output = LayerNorm(input + sublayer(input))`
- **Purpose**: Enable deep networks, prevent vanishing gradients
- **Layer Norm**: Stabilize training, normalize across features

### 3. Special Tokens

| Token | Purpose | Position | Usage |
|-------|---------|----------|-------|
| **[CLS]** | Classification | First position | Sequence-level tasks (sentiment, classification) |
| **[SEP]** | Separator | Between sentences | Mark sentence boundaries |
| **[MASK]** | Masked token | Random positions | Pre-training MLM objective |
| **[PAD]** | Padding | End of sequences | Batch processing with uniform length |

### 4. Output Heads

#### Masked Language Model (MLM) Head
- **Input**: Hidden states from all positions
- **Output**: Vocabulary probability distribution
- **Size**: `[batch_size, seq_len, vocab_size]`

```python
mlm_logits = self.mlm_head(hidden_states)  # Predict masked tokens
```

#### Next Sentence Prediction (NSP) Head
- **Input**: [CLS] token representation
- **Output**: Binary classification (IsNext / NotNext)
- **Size**: `[batch_size, 2]`

```python
nsp_logits = self.nsp_head(cls_output)  # Predict sentence relationship
```

## 🎯 Training Methodology

### Pre-training Objectives

#### 1. Masked Language Model (MLM)

**Concept**: Randomly mask tokens and predict them using bidirectional context.

**Masking Strategy**:
- **15%** of tokens are selected for masking
- Of selected tokens:
    - **80%** → Replace with [MASK]
    - **10%** → Replace with random token
    - **10%** → Keep original (but still predict)

**Example**:
```
Original: "The cat sat on the mat"
Masked:   "The [MASK] sat on the mat"
Target:   Predict "cat" at masked position
```

**Why this strategy?**
- **80% [MASK]**: Main learning signal
- **10% random**: Prevent over-reliance on [MASK] token
- **10% unchanged**: Learn from all positions, not just [MASK]

**Code Implementation**:
```python
def mask_tokens(self, inputs, mask_prob=0.15):
    labels = inputs.clone()
    
    # Select 15% of tokens
    probability_matrix = torch.full(labels.shape, mask_prob)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # 80% → [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = self.mask_token_id
    
    # 10% → random token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(self.vocab_size, labels.shape)
    inputs[indices_random] = random_words[indices_random]
    
    return inputs, labels
```

#### 2. Next Sentence Prediction (NSP)

**Concept**: Learn sentence-level relationships by predicting if sentence B follows sentence A.

**Data Preparation**:
- **50%** of examples: B actually follows A (IsNext)
- **50%** of examples: B is random sentence (NotNext)

**Example**:
```
IsNext:
Sentence A: "I went to the store."
Sentence B: "I bought some groceries."
Label: 1 (IsNext)

NotNext:
Sentence A: "I went to the store."  
Sentence B: "The weather is nice today."
Label: 0 (NotNext)
```

**Input Format**:
```
[CLS] I went to the store [SEP] I bought some groceries [SEP]
```

### Training Process

#### Phase 1: Pre-training
1. **Data**: Large text corpus (Wikipedia, BookCorpus, etc.)
2. **Duration**: Days/weeks on multiple GPUs
3. **Objective**: Learn general language representations
4. **Output**: Pre-trained BERT model

#### Phase 2: Fine-tuning
1. **Data**: Task-specific labeled dataset
2. **Duration**: Hours on single GPU
3. **Objective**: Adapt to specific task
4. **Output**: Task-specific model

### Loss Function

```python
def compute_loss(self, outputs, mlm_labels, nsp_labels):
    # MLM Loss: Cross-entropy over vocabulary
    mlm_loss = F.cross_entropy(
        outputs['mlm_logits'].view(-1, self.vocab_size), 
        mlm_labels.view(-1), 
        ignore_index=-100  # Ignore non-masked tokens
    )
    
    # NSP Loss: Binary cross-entropy
    nsp_loss = F.cross_entropy(outputs['nsp_logits'], nsp_labels)
    
    # Combined loss
    total_loss = mlm_loss + nsp_loss
    return total_loss, mlm_loss, nsp_loss
```

## 💻 Implementation Guide

### Core Components Explained

#### 1. MultiHeadAttention Implementation

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model          # 768
        self.num_heads = num_heads      # 12
        self.d_k = d_model // num_heads # 64 per head
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)  # Output projection
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # 1. Linear projections and reshape for multi-head
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 3. Apply mask (for padding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 4. Softmax and apply to values
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # 5. Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(context)
        
        return output
```

**Step-by-step breakdown**:
1. **Linear Projections**: Transform input into Query, Key, Value matrices
2. **Multi-head Reshape**: Split into multiple attention heads
3. **Attention Scores**: Compute similarity between queries and keys
4. **Masking**: Prevent attention to padding tokens
5. **Softmax**: Normalize attention weights
6. **Context**: Weight values by attention scores
7. **Concatenate**: Combine all heads
8. **Output Projection**: Final linear transformation

#### 2. Positional Encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Sinusoidal encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even positions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd positions
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

**Mathematical intuition**:
- **Problem**: Transformers have no inherent position understanding
- **Solution**: Add position-specific vectors to embeddings
- **Sinusoidal pattern**: Allows model to learn relative positions
- **Different frequencies**: Each dimension encodes position differently

#### 3. Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),    # 768 → 3072
            nn.GELU(),                   # Activation
            nn.Linear(d_ff, d_model)     # 3072 → 768
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

**Key design principles**:
- **Residual connections**: `output = norm(input + sublayer)`
- **Layer normalization**: Applied after residual addition
- **GELU activation**: Smoother than ReLU, better for NLP
- **Dropout**: Regularization to prevent overfitting

### Model Architecture Decisions

#### Why 768 dimensions?
- **Computational efficiency**: Divisible by common head counts (8, 12, 16)
- **Capacity**: Large enough for complex representations
- **Memory**: Balanced with available hardware constraints

#### Why 12 attention heads?
- **Diversity**: Each head can focus on different relationships
- **Parallelization**: Efficient on modern hardware
- **Empirical**: Found to work well across many tasks

#### Why 6-12 layers?
- **Depth vs. efficiency**: More layers = more capacity but slower training
- **BERT-base**: 12 layers, BERT-large: 24 layers
- **Diminishing returns**: Performance gains plateau after certain depth

## 🔧 Performance & Optimization

### CPU Optimization Strategies

#### 1. Model Size Reduction

```python
# Small BERT for CPU training
small_config = {
    'vocab_size': 30000,
    'd_model': 256,      # Reduced from 768
    'num_heads': 4,      # Reduced from 12
    'num_layers': 3,     # Reduced from 6
    'd_ff': 1024,        # Reduced from 3072
    'max_len': 128,      # Reduced from 512
}

model = BasicBERT(**small_config).to('cpu')
```

**Memory savings**:
- **Parameters**: ~15M (vs. 110M for BERT-base)
- **Memory usage**: ~500MB (vs. 1.5GB for BERT-base)
- **Speed**: 3-4x faster training

#### 2. Batch Size Optimization

```python
# CPU-friendly batch sizes
batch_sizes = {
    'training': 4,      # Small batches for CPU
    'inference': 16,    # Larger batches OK for inference
    'evaluation': 8     # Moderate batch size
}
```

#### 3. Gradient Accumulation

```python
# Simulate larger batch sizes
accumulation_steps = 8
effective_batch_size = batch_size * accumulation_steps

for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps  # Scale loss
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 4. Mixed Precision (CPU)

```python
# Use automatic mixed precision for CPU
scaler = torch.cuda.amp.GradScaler()

with torch.autocast(device_type='cpu', dtype=torch.float16):
    outputs = model(input_ids)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Memory Management

#### Efficient Inference

```python
def efficient_inference(model, input_ids, max_memory_mb=1000):
    """Memory-efficient inference with batching"""
    model.eval()
    batch_size = input_ids.size(0)
    
    # Estimate memory per sample
    sample_memory = estimate_memory_usage(model, input_ids[0:1])
    max_batch_size = max_memory_mb // sample_memory
    
    results = []
    with torch.no_grad():
        for i in range(0, batch_size, max_batch_size):
            batch = input_ids[i:i+max_batch_size]
            output = model(batch)
            results.append(output)
    
    return torch.cat(results, dim=0)
```

#### Memory Profiling

```python
def profile_memory_usage(model, input_shape):
    """Profile memory usage for different model configurations"""
    import psutil
    import gc
    
    # Baseline memory
    gc.collect()
    baseline = psutil.virtual_memory().used
    
    # Create dummy input
    dummy_input = torch.randint(0, 1000, input_shape)
    
    # Forward pass
    output = model(dummy_input)
    forward_memory = psutil.virtual_memory().used
    
    # Backward pass
    loss = output['mlm_logits'].sum()
    loss.backward()
    backward_memory = psutil.virtual_memory().used
    
    print(f"Baseline: {baseline / 1e6:.1f} MB")
    print(f"Forward: {(forward_memory - baseline) / 1e6:.1f} MB")
    print(f"Backward: {(backward_memory - forward_memory) / 1e6:.1f} MB")
```

## 🔬 Advanced Topics

### 1. Attention Pattern Analysis

#### Visualizing Attention

```python
def visualize_attention(model, input_ids, tokenizer, layer=0, head=0):
    """Extract and visualize attention patterns"""
    model.eval()
    
    # Hook to capture attention weights
    attention_weights = []
    
    def attention_hook(module, input, output):
        # Capture attention weights from specific layer/head
        attention_weights.append(output[1])  # attention weights
    
    # Register hook
    hook = model.layers[layer].attention.register_forward_hook(attention_hook)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Remove hook
    hook.remove()
    
    # Extract attention for specific head
    attn = attention_weights[0][0, head]  # [seq_len, seq_len]
    
    # Visualize
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    plot_attention_heatmap(attn, tokens)
```

#### Attention Head Specialization

Different attention heads often specialize in different linguistic patterns:

- **Head 1**: Subject-verb relationships
- **Head 2**: Adjective-noun pairs
- **Head 3**: Long-range dependencies
- **Head 4**: Punctuation and structure
- **Head 5**: Coreference resolution

```python
def analyze_head_specialization(model, test_sentences):
    """Analyze what different attention heads focus on"""
    patterns = {
        'syntactic': [],     # Grammar relationships
        'semantic': [],      # Meaning relationships  
        'positional': [],    # Distance-based patterns
        'structural': []     # Punctuation, formatting
    }
    
    for sentence in test_sentences:
        attentions = extract_all_attention(model, sentence)
        
        for head in range(12):
            pattern_type = classify_attention_pattern(attentions[head])
            patterns[pattern_type].append(head)
    
    return patterns
```

### 2. Transfer Learning Strategies

#### Task-Specific Fine-tuning

```python
class BERTForClassification(nn.Module):
    def __init__(self, bert_model, num_classes, dropout=0.1):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert_model.d_model, num_classes)
        
    def forward(self, input_ids, segment_ids=None):
        outputs = self.bert(input_ids, segment_ids)
        
        # Use [CLS] token for classification
        cls_output = outputs['cls_output']
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        return logits
```

#### Layer-wise Learning Rates

```python
def create_optimizer_with_layer_decay(model, base_lr=2e-5, decay_rate=0.9):
    """Apply different learning rates to different layers"""
    parameters = []
    
    # Embedding layers - lowest learning rate
    parameters.append({
        'params': [p for n, p in model.named_parameters() if 'embedding' in n],
        'lr': base_lr * (decay_rate ** 12)
    })
    
    # Transformer layers - decreasing learning rate for lower layers
    for i in range(12):
        layer_params = [p for n, p in model.named_parameters() if f'layers.{i}' in n]
        parameters.append({
            'params': layer_params,
            'lr': base_lr * (decay_rate ** (11 - i))
        })
    
    # Output layers - highest learning rate
    parameters.append({
        'params': [p for n, p in model.named_parameters() if 'head' in n],
        'lr': base_lr
    })
    
    return torch.optim.AdamW(parameters)
```

#### Gradual Unfreezing

```python
class GradualUnfreezing:
    def __init__(self, model, unfreeze_schedule):
        self.model = model
        self.schedule = unfreeze_schedule
        self.current_epoch = 0
        
    def step(self, epoch):
        """Gradually unfreeze layers based on schedule"""
        if epoch in self.schedule:
            layers_to_unfreeze = self.schedule[epoch]
            for layer_name in layers_to_unfreeze:
                for param in self.model.get_submodule(layer_name).parameters():
                    param.requires_grad = True
            print(f"Unfroze layers: {layers_to_unfreeze}")

# Usage
schedule = {
    0: ['classifier'],                    # Epoch 0: Only classification head
    2: ['layers.11', 'layers.10'],       # Epoch 2: Top 2 layers
    4: ['layers.9', 'layers.8'],         # Epoch 4: Next 2 layers
    6: ['layers.7', 'layers.6'],         # Continue unfreezing
    8: ['layers.5', 'layers.4', 'layers.3', 'layers.2', 'layers.1', 'layers.0']
}

unfreezer = GradualUnfreezing(model, schedule)
```

### 3. Interpretability and Analysis

#### Probing Tasks

```python
def create_probing_classifier(bert_model, probe_task='pos_tagging'):
    """Create classifier to probe BERT representations"""
    
    probe_configs = {
        'pos_tagging': {'num_classes': 45, 'layer': 6},      # Part-of-speech
        'ner': {'num_classes': 9, 'layer': 8},               # Named entities
        'dependency': {'num_classes': 37, 'layer': 4},       # Dependency relations
        'sentiment': {'num_classes': 3, 'layer': 11}         # Sentiment
    }
    
    config = probe_configs[probe_task]
    
    class ProbingClassifier(nn.Module):
        def __init__(self, bert, layer_idx, num_classes):
            super().__init__()
            self.bert = bert
            self.layer_idx = layer_idx
            self.classifier = nn.Linear(bert.d_model, num_classes)
            
        def forward(self, input_ids):
            with torch.no_grad():
                # Extract representations from specific layer
                hidden_states = self.bert(input_ids)['last_hidden_state']
            
            # Use specific layer for probing
            layer_output = hidden_states  # Simplified - would need layer extraction
            logits = self.classifier(layer_output)
            return logits
    
    return ProbingClassifier(bert_model, config['layer'], config['num_classes'])
```

#### Representation Analysis

```python
def analyze_bert_representations(model, sentences, analysis_type='similarity'):
    """Analyze BERT's internal representations"""
    model.eval()
    
    representations = []
    
    with torch.no_grad():
        for sentence in sentences:
            input_ids = tokenize_sentence(sentence)
            outputs = model(input_ids)
            
            # Extract different types of representations
            if analysis_type == 'cls':
                repr = outputs['cls_output']
            elif analysis_type == 'mean_pool':
                repr = outputs['last_hidden_state'].mean(dim=1)
            elif analysis_type == 'max_pool':
                repr = outputs['last_hidden_state'].max(dim=1)[0]
            
            representations.append(repr)
    
    representations = torch.stack(representations)
    
    # Compute similarity matrix
    similarity_matrix = torch.cosine_similarity(
        representations.unsqueeze(1), 
        representations.unsqueeze(0), 
        dim=2
    )
    
    return similarity_matrix, representations
```