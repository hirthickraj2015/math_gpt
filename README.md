# Math and Boolean GPT
## CS7CS4 Machine Learning - Final Assignment 2025-26

This repository implements transformer-based models for solving:
1. **Arithmetic expressions** (Math GPT)
2. **Boolean logic expressions** (Boolean GPT)

## Project Structure

```
├── 1_dataset_generation.ipynb    # Generate training/testing datasets
├── 2_math_gpt.ipynb              # Part 1: Math GPT implementation
├── 3_boolean_gpt.ipynb           # Part 2: Boolean GPT implementation
├── dataset/
│   ├── math/
│   │   ├── training/math_train.txt
│   │   └── testing/math_test.txt
│   └── boolean/
│       ├── training/boolean_train.txt
│       └── testing/boolean_test.txt
├── model_weights_part1.pth       # Trained Math GPT model
├── model_weights_part2.pth       # Trained Boolean GPT model (after running notebook 3)
├── requirements.txt              # Python dependencies
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hirthickraj2015/math_gpt.git
cd math_gpt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Generate Datasets (Optional - datasets included)
```bash
jupyter notebook 1_dataset_generation.ipynb
```
Run all cells to generate fresh datasets for both tasks.

### Step 2: Train Math GPT (Part 1)
```bash
jupyter notebook 2_math_gpt.ipynb
```
Run all cells to train the arithmetic model. This will:
- Load the math dataset
- Train a transformer model
- Evaluate on test set
- Save `model_weights_part1.pth`

### Step 3: Train Boolean GPT (Part 2)
```bash
jupyter notebook 3_boolean_gpt.ipynb
```
Run all cells to train the boolean logic model. This will:
- Load the boolean dataset
- Train a transformer model
- Evaluate on test set
- Save `model_weights_part2.pth`

## Model Architecture

### Math GPT
- **Embedding dimension**: 64
- **Layers**: 2
- **Attention heads**: 4
- **Block size**: 32
- **Parameters**: ~166K

### Boolean GPT
- **Embedding dimension**: 32
- **Layers**: 2
- **Attention heads**: 2
- **Block size**: 48
- **Parameters**: ~45K

Both models use transformer architecture with:
- Multi-head self-attention
- Feed-forward networks
- Layer normalization
- Residual connections

## Dataset Details

### Math Dataset
- **Training**: ~2,000 expressions
- **Testing**: ~200 expressions
- **Operations**: +, -, *, //, %
- **Range**: Single-digit (0-9) focused
- **Complexity**: Simple to moderate (with parentheses)

### Boolean Dataset
- **Training**: ~1,000 expressions
- **Testing**: ~100 expressions
- **Operations**: AND, OR, XOR, NOT
- **Values**: True, False
- **Complexity**: Simple to complex (with parentheses)

## Evaluation Metrics

Both models are evaluated using:
1. **Exact Match Accuracy**: Percentage of completely correct answers
2. **Character-Level Accuracy**: Partial credit for partially correct answers
3. **Operation-Specific Accuracy**: Breakdown by operation type
4. **Error Analysis**: Common failure patterns

## Loading Pre-trained Models

```python
import torch
from model_code import GPTLanguageModel

# Load Math GPT
model = GPTLanguageModel()
model.load_state_dict(torch.load('model_weights_part1.pth'))
model.eval()

# Load Boolean GPT
model = GPTLanguageModel()
model.load_state_dict(torch.load('model_weights_part2.pth'))
model.eval()
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib
- Seaborn
- Jupyter

## Assignment Tasks Completed

### Part 1: Math GPT
- ✅ Task 1.1: Built appropriate training/testing datasets
- ✅ Task 1.2: Defined evaluation metrics (exact match, character-level, operation-specific)
- ✅ Task 1.3: Explored architectural adaptations (embedding size, layers, heads, tokenization)
- ✅ Task 1.4: Analyzed performance across different arithmetic operations

### Part 2: Boolean GPT
- ✅ Task 2.1: Built appropriate training/testing datasets
- ✅ Task 2.2: Defined evaluation metrics
- ✅ Task 2.3: Explored architectural adaptations
- ✅ Task 2.4: Analyzed performance across different boolean operations

### Part 3: Discussion
- Architectural comparison between Math and Boolean GPT
- Analysis of what works and what doesn't
- Discussion of design choices and their impact

## Author

Hirthick Raj - Trinity College Dublin

## License

Academic use only - CS7CS4 Final Assignment 2025-26
