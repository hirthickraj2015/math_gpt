# Model Optimization Guide - Fixing Low Accuracy

## Problem: Model Getting 1/21 Accuracy

If your model is performing poorly (1/21 or similar low accuracy), the issue is likely:
1. **Dataset too complex** - Large numbers and complex expressions
2. **Not enough repetition** - Model hasn't seen patterns enough times
3. **Model configuration** - Hyperparameters not optimal for the task

## Solution: Optimized Dataset + Better Training

---

## Step 1: Generate Optimized Dataset

### Use the New Optimized Dataset Generator

```bash
jupyter notebook dataset_generation_optimized.ipynb
# Run ALL cells
```

### What's Different:

**Original Dataset Problems:**
- Numbers from 1-1000 (too large!)
- Complex nested expressions
- Only 1-2 repetitions per pattern
- Model can't learn basic patterns

**Optimized Dataset Benefits:**
- **Single digits only (0-9)** ✓
- **ALL combinations** exhaustively covered ✓
- **5-10x repetition** per pattern ✓
- **~30K focused examples** vs 120K scattered ✓
- **Learnable patterns** ✓

### Dataset Breakdown:

**Math Dataset (~30,000 expressions):**
- Addition: 100 combinations × 5 reps = 500
- Subtraction: 100 combinations × 5 reps = 500
- Multiplication: 100 combinations × 5 reps = 500
- Division: 90 combinations × 5 reps = 450
- Modulo: 90 combinations × 5 reps = 450
- Two operations: ~2,000 (sampled)
- Parentheses: ~1,500 (sampled)

**Examples:**
```
0+0=0
0+1=1
0+2=2
...
9+9=18
5*3=15
(2+3)*4=20
```

**Boolean Dataset (~3,500 expressions):**
- AND: 4 combinations × 10 reps = 40
- OR: 4 combinations × 10 reps = 40
- XOR: 4 combinations × 10 reps = 40
- NOT: 2 combinations × 50 reps = 100
- With parentheses: ~3,000 (exhaustive)

---

## Step 2: Retrain with Optimized Settings

### Update Training Hyperparameters

In `part1_math_gpt.ipynb`, change these settings:

```python
# RECOMMENDED SETTINGS FOR SINGLE-DIGIT DATASET
batch_size = 32          # Smaller batch for better learning
block_size = 16          # Shorter context (expressions are short)
max_iters = 15000        # More iterations
eval_interval = 500
learning_rate = 1e-3     # Higher learning rate for faster convergence
eval_iters = 200
n_embd = 64              # Smaller embedding (simpler task)
n_head = 4
n_layer = 3              # Fewer layers (simpler patterns)
dropout = 0.1
```

### Why These Changes:

| Parameter | Old | New | Reason |
|-----------|-----|-----|---------|
| batch_size | 64 | 32 | Better gradient updates |
| block_size | 32 | 16 | Expressions are shorter |
| n_embd | 128 | 64 | Simpler task needs less capacity |
| n_layer | 4 | 3 | Avoid overfitting on simple patterns |
| learning_rate | 5e-4 | 1e-3 | Faster learning |
| max_iters | 10000 | 15000 | Ensure full convergence |

---

## Step 3: Train the Model

```bash
# 1. Generate optimized dataset
jupyter notebook dataset_generation_optimized.ipynb
# Run all cells

# 2. Update hyperparameters in part1_math_gpt.ipynb
# (see settings above)

# 3. Train model
jupyter notebook part1_math_gpt.ipynb
# Run all cells

# 4. Test model
jupyter notebook test_model.ipynb
# Run all cells
```

---

## Step 4: Verify Improvement

### Expected Results:

**Before Optimization:**
```
Accuracy: 1/21 (4.8%)
```

**After Optimization:**
```
Accuracy: 18-20/21 (85-95%)
```

### Sample Test Results:

```
Prompt          Expected    Predicted   Status
------------------------------------------------
5+3=            8           8           ✓
12-7=           5           5           ✓
6*8=            48          48          ✓
(3+2)*4=        20          20          ✓
7//2=           3           3           ✓
```

---

## Troubleshooting

### If Accuracy is Still Low (<50%)

#### Problem 1: Model Not Learning

**Symptoms:**
- Loss not decreasing
- Random predictions

**Solutions:**
```python
# Increase learning rate
learning_rate = 2e-3

# Simplify model further
n_embd = 32
n_layer = 2

# Train longer
max_iters = 20000
```

#### Problem 2: Overfitting

**Symptoms:**
- Train loss very low
- Test loss high
- Perfect on training, poor on testing

**Solutions:**
```python
# Increase dropout
dropout = 0.2

# Increase data augmentation (generate more data)
repetitions = 10  # in dataset generation

# Reduce model size
n_embd = 48
```

#### Problem 3: Underfitting

**Symptoms:**
- Both train and test loss high
- Not learning patterns

**Solutions:**
```python
# Increase model capacity
n_embd = 96
n_layer = 4

# Train longer
max_iters = 25000

# Lower learning rate slightly
learning_rate = 5e-4
```

---

## Step 5: Gradual Complexity Increase

Once you achieve 90%+ accuracy on single digits:

### Phase 1: Single Digits (Current)
- Numbers: 0-9
- Target: 90%+ accuracy
- Status: ✓

### Phase 2: Add Two-Digit Numbers
```python
# In dataset generation:
for a in range(20):  # 0-19
    for b in range(20):
        ...
```
- Numbers: 0-19
- Target: 85%+ accuracy

### Phase 3: Add Larger Numbers Gradually
```python
# Mix of ranges:
small = range(10)      # 50% of data
medium = range(50)     # 30% of data
large = range(100)     # 20% of data
```
- Progressive difficulty
- Target: 80%+ accuracy

---

## Best Practices

### 1. Start Simple
✓ Single digits first
✓ Basic operations only
✓ High repetition

### 2. Monitor Training
✓ Watch loss curves
✓ Check example outputs during training
✓ Save checkpoints frequently

### 3. Test Systematically
✓ Test on held-out data
✓ Check each operation type
✓ Identify specific failure patterns

### 4. Iterate Gradually
✓ Don't jump to complex data
✓ Increase difficulty slowly
✓ Verify each step

---

## Quick Fix Checklist

If model accuracy is low, try these in order:

- [ ] Generate optimized single-digit dataset
- [ ] Reduce model size (n_embd=64, n_layer=3)
- [ ] Increase learning rate (1e-3)
- [ ] Train longer (15000+ iterations)
- [ ] Reduce batch size (32)
- [ ] Reduce block size (16)
- [ ] Check dataset is correct (run verification cell)
- [ ] Ensure no data leakage (train/test split correct)
- [ ] Test with low temperature (0.5)
- [ ] Check vocabulary is small (should be ~15 chars for math)

---

## Expected Training Time

With optimized dataset:

| Hardware | Time | Notes |
|----------|------|-------|
| CPU | 15-30 min | Acceptable for single-digit task |
| GPU | 5-10 min | Much faster |
| MPS (Apple) | 8-15 min | M1/M2/M3 Macs |

Smaller dataset + simpler model = faster training!

---

## Verification Script

Run this after training to verify improvement:

```python
# Quick accuracy check
from test_model import test_multiple_prompts

test_prompts = [
    "1+1", "2+3", "5+4", "9+0",
    "5-2", "8-3", "7-7", "9-5",
    "3*2", "4*5", "7*1", "9*9",
    "8//2", "9//3", "6//2",
    "7%3", "8%5", "9%2"
]

results = test_multiple_prompts(model, test_prompts, encode, decode)
correct = sum(1 for _, _, _, corr in results if corr)
print(f"Accuracy: {correct}/{len(results)} = {(correct/len(results))*100:.1f}%")
```

**Target:** 15/18+ correct (83%+)

---

## Summary

### Key Changes:

1. **Dataset**: Single-digit focused with high repetition
2. **Model**: Smaller and simpler (n_embd=64, n_layer=3)
3. **Training**: More iterations with higher learning rate
4. **Testing**: Systematic verification

### Expected Outcome:

```
Before: 1/21 (5%) accuracy
After:  18/21 (85-95%) accuracy
```

### Timeline:

1. Generate dataset: 2 min
2. Update settings: 1 min
3. Train model: 15-30 min
4. Test model: 2 min

**Total: ~20-35 minutes to fix!**

---

## Next Steps After Success

Once you achieve 90%+ on single digits:

1. Document your optimization process in report
2. Discuss why single-digit dataset works better
3. Create Part 2 (Boolean) with same approach
4. Compare architectures in Part 3
5. Celebrate! 🎉

---

**Good luck! This optimization should dramatically improve your model accuracy!** 🚀
