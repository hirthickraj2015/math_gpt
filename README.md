# Math and Boolean GPT
## CS7CS4 Machine Learning - Final Assignment 2025-26

Transformer-based models for symbolic reasoning tasks: arithmetic expressions and boolean logic.

## Project Structure

```
├── 1_dataset_generation.ipynb    # Generate math and boolean datasets
├── 2_math_gpt.ipynb              # Train Math GPT model
├── 3_boolean_gpt.ipynb           # Train Boolean GPT model
├── 4_evaluation.ipynb            # Comprehensive evaluation & analysis
├── dataset/
│   ├── math/
│   │   ├── training/math_train.txt
│   │   └── testing/math_test.txt
│   └── boolean/
│       ├── training/boolean_train.txt
│       └── testing/boolean_test.txt
├── model_weights_part1.pth    # Trained Math GPT model
├── model_weights_part2.pth    # Trained Boolean GPT model
└── requirements.txt           # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Datasets

```bash
jupyter notebook 1_dataset_generation.ipynb
```

Run all cells to generate comprehensive datasets for both math and boolean tasks.

### 3. Train Models

Train the Math GPT:
```bash
jupyter notebook 2_math_gpt.ipynb
```

Train the Boolean GPT:
```bash
jupyter notebook 3_boolean_gpt.ipynb
```

### 4. Evaluate and Analyze

```bash
jupyter notebook 4_evaluation.ipynb
```

This generates comprehensive evaluation results addressing all assignment tasks:
- Task 1.2 & 2.2: Evaluation metrics
- Task 1.4 & 2.4: Operation-specific analysis
- Task 3.1: Critical comparison

## Model Architecture

Both models use a GPT-style transformer architecture with **task-specific adaptations**:

### Math GPT:
- **Vocabulary**: Character-level tokenization (19 characters)
- **Embedding dimension**: 64
- **Layers**: 2 transformer blocks
- **Attention heads**: 4 multi-head attention
- **Block size**: 32 (context window)
- **Dropout**: 0.1
- **Parameters**: ~0.2M (small, efficient)

### Boolean GPT:
- **Vocabulary**: Character-level tokenization (15 characters)
- **Embedding dimension**: 32
- **Layers**: 2 transformer blocks
- **Attention heads**: 2 multi-head attention
- **Block size**: 48 (context window)
- **Dropout**: 0.05
- **Parameters**: ~0.03M (very small, efficient)

### Design Rationale:

1. **Character-level tokenization**: Each digit/operator/boolean is atomic and meaningful
2. **Small embeddings**: Limited vocabulary doesn't require large embeddings
3. **Shallow architecture**: Symbolic tasks simpler than natural language
4. **Small block size**: Most expressions < 32 characters
5. **Task-specific sizing**: Boolean logic is simpler, so smaller model suffices

## Dataset Statistics

### Math Dataset:
- **Training**: ~90,000 expressions
- **Testing**: ~10,000 expressions
- **Operations**: +, -, *, //, %
- **Complexity**: Single digit to multi-operation with parentheses

### Boolean Dataset:
- **Training**: ~9,000 expressions
- **Testing**: ~1,000 expressions
- **Operations**: AND, OR, XOR, NOT
- **Complexity**: Single to nested operations with parentheses

## Evaluation Metrics

See `4_evaluation.py` for comprehensive metrics:

1. **Exact Match Accuracy**: Percentage of completely correct answers
2. **Character-Level Accuracy**: Partial credit for partially correct answers
3. **Operation-Specific Accuracy**: Breakdown by operation type
4. **Error Pattern Analysis**: Common failure modes
5. **Generalization Metrics**: Test vs. training performance

## Outputs for Report

The evaluation notebook (`4_evaluation.ipynb`) generates:

1. **evaluation_results.txt**: Detailed results with example predictions
2. **evaluation_comparison.png**: Visualizations comparing both models
3. **Markdown-formatted analysis**: Can be copied directly into the report

These outputs directly address all evaluation questions from the assignment PDF.

## Assignment Tasks Coverage

### Part 1: Math GPT (46 marks)

- **Task 1.1** (8 marks): Dataset generation - See `1_dataset_generation.ipynb`
- **Task 1.2** (8 marks): Evaluation metrics - See `4_evaluation.ipynb`
- **Task 1.3** (15 marks): Architectural adaptations - See `2_math_gpt.ipynb`
- **Task 1.4** (15 marks): Operation analysis - See `4_evaluation.ipynb`

### Part 2: Boolean GPT (46 marks)

- **Task 2.1** (8 marks): Dataset generation - See `1_dataset_generation.ipynb`
- **Task 2.2** (8 marks): Evaluation metrics - See `4_evaluation.ipynb`
- **Task 2.3** (15 marks): Architectural adaptations - See `3_boolean_gpt.ipynb`
- **Task 2.4** (15 marks): Operation analysis - See `4_evaluation.ipynb`

### Part 3: Discussion (8 marks)

- **Task 3.1** (8 marks): Critical comparison - See `4_evaluation.ipynb`

## Submission Checklist

- [ ] PDF report (5-8 pages) with analysis
- [ ] `model_weights_part1.pth` (Math GPT)
- [ ] `model_weights_part2.pth` (Boolean GPT)
- [ ] Appendix with code (from notebook exports)
- [ ] Appendix with prompt-output examples (from `evaluation_results.txt`)
- [ ] ZIP file with all code and data

## Key Features

- **Clean structure**: 4 focused Jupyter notebooks
- **Comprehensive evaluation**: Addresses all assignment tasks
- **Report-ready output**: Formatted text and visualizations for direct use
- **Well-documented**: Clear explanations and reasoning
- **Reproducible**: Fixed random seeds, saved models

## Dependencies

See `requirements.txt` for full list. Main dependencies:
- `torch` - Deep learning framework
- `jupyter` - Interactive notebook environment
- `numpy` - Numerical computing
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations

## Author

CS7CS4 Student - Trinity College Dublin

## License

Academic use only - CS7CS4 Final Assignment 2025-26
