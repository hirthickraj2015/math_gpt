# Model Testing Guide
## CS7CS4 Machine Learning - Final Assignment

Quick guide for testing your trained models.

---

## Quick Start

```bash
jupyter notebook test_model.ipynb
# Run all cells in order
```

---

## What the Testing Notebook Does

The `test_model.ipynb` notebook provides 5 different ways to test your models:

### 1. Predefined Test Sets
- **Math GPT**: 22 test cases covering all operations
- **Boolean GPT**: 17 test cases covering all logic operations
- Shows accuracy and individual results
- Saves results to file automatically

### 2. Interactive Testing
- Type prompts one at a time
- See immediate results
- Great for exploring model behavior
- Optional: save custom test results

### 3. Batch Testing from File
- Load many prompts from a text file
- Test all at once
- Useful for large-scale testing

### 4. Report Example Generation
- Automatically finds correct and incorrect predictions
- Generates examples for your report appendix
- Shows both strengths and weaknesses
- Ready-to-use format

### 5. Single Prompt Testing
- Test any individual expression
- Adjust temperature and output length
- Debug specific cases

---

## Usage Examples

### Test Math Model

```python
# In Section 5, use:
checkpoint_path = 'checkpoints/best_model.pt'
model_type = 'Math GPT'

# Then run Section 6 to test math prompts
```

**Example output:**
```
Prompt                    Expected        Predicted       Status
----------------------------------------------------------------
5+3=                      8               8               ✓
(3+2)*4=                  20              20              ✓
12-5=                     7               7               ✓

Accuracy: 21/22 (95.5%)
```

### Test Boolean Model

```python
# In Section 5, use:
checkpoint_path = 'checkpoints_boolean/best_model.pt'
model_type = 'Boolean GPT'

# Then run Section 7 to test boolean prompts
```

### Interactive Mode

```python
# Run Section 8, then uncomment:
interactive_test()

# Example session:
# Enter prompt: 15+7
#   Input:    15+7=
#   Output:   22 ✓

# Enter prompt: (5*3)+2
#   Input:    (5*3)+2=
#   Output:   17 ✓
```

### Generate Report Examples

```python
# Run Section 10:
generate_report_examples(num_correct=15, num_incorrect=15)

# Generates file: math_gpt_report_examples.txt
```

---

## File Outputs

The testing notebook generates these files:

### 1. `math_test_results.txt`
- Results from predefined math test set
- Shows input, expected, predicted, status
- Overall accuracy

### 2. `boolean_test_results.txt`
- Results from predefined boolean test set
- Same format as math results

### 3. `*_report_examples.txt`
- Examples for your report appendix
- Divided into:
  - **Strengths** (correct predictions)
  - **Weaknesses** (incorrect predictions)
- Ready to copy into report

### 4. `custom_*_results.txt`
- Results from interactive testing
- Your custom prompts and results

---

## Testing Parameters

### Temperature
Controls output randomness:

```python
temperature=0.5   # Very deterministic (recommended for testing)
temperature=0.7   # Balanced (default)
temperature=1.0   # More random
```

**Lower temperature** = more consistent outputs
**Higher temperature** = more varied outputs

### Max Tokens
Maximum length of generated output:

```python
max_tokens=20    # Good for most expressions
max_tokens=30    # For longer/complex expressions
```

---

## Common Tasks

### Task: Test model accuracy

1. Load model (Section 5)
2. Run predefined tests (Section 6 or 7)
3. Check accuracy output

### Task: Find examples for report

1. Load model (Section 5)
2. Run Section 10 (generate_report_examples)
3. Open generated `*_report_examples.txt`
4. Copy examples into your report appendix

### Task: Test specific expressions

```python
# Use test_single_prompt function:
prompt = "15+7"
full_output, answer = test_single_prompt(
    model, prompt, encode, decode,
    max_tokens=20, temperature=0.7
)
print(f"Answer: {answer}")
```

### Task: Compare multiple model configurations

1. Train multiple models with different hyperparameters
2. Save each with different checkpoint names
3. Load and test each one
4. Compare accuracies

---

## Tips for Testing

### 1. Start Simple
Test basic cases first:
- Math: `5+3`, `10-2`, `4*5`
- Boolean: `True AND False`, `NOT True`

### 2. Gradually Increase Complexity
Then test harder cases:
- Math: `(3+2)*4`, `((5+3)*2)-1`
- Boolean: `(True OR False) AND NOT True`

### 3. Look for Patterns
Notice what the model handles well:
- Does it handle addition better than division?
- Are nested parentheses problematic?
- Does expression length matter?

### 4. Test Edge Cases
Try unusual inputs:
- Very large numbers
- Many nested parentheses
- Long expression chains

### 5. Use Temperature Wisely
- Testing accuracy: use low temperature (0.5-0.7)
- Exploring behavior: use higher temperature (0.8-1.0)

---

## Troubleshooting

### Error: Checkpoint not found

**Problem:** Model hasn't been trained yet

**Solution:**
```bash
# Train the model first:
jupyter notebook part1_math_gpt.ipynb
# Run all cells

# Then test:
jupyter notebook test_model.ipynb
```

### Error: Character not in vocabulary

**Problem:** Testing with wrong model type

**Solution:** Make sure you're testing:
- Math expressions with Math GPT
- Boolean expressions with Boolean GPT

### Low accuracy

**Possible causes:**
1. Model needs more training
2. Testing on out-of-distribution examples
3. Temperature too high

**Solutions:**
1. Train for more iterations
2. Test on similar examples to training data
3. Lower temperature to 0.5-0.7

### Inconsistent outputs

**Problem:** High temperature causing randomness

**Solution:** Lower temperature:
```python
temperature=0.5  # More consistent
```

---

## Creating Custom Test Files

### Format with answers:
```
5+3=8
12-7=5
(3+2)*4=20
```

### Format without answers:
```
5+3
12-7
(3+2)*4
```

### Load and test:
```python
test_from_file('my_tests.txt', has_answers=True)
```

---

## For Your Report

### What to include:

1. **Methodology**
   - How you tested the model
   - Test set size and composition
   - Evaluation metrics used

2. **Results**
   - Overall accuracy
   - Operation-specific accuracy
   - Example predictions (correct and incorrect)

3. **Analysis**
   - What operations work well
   - What operations fail
   - Why (your interpretation)
   - Error patterns

### Example table for report:

```
| Expression Type | Accuracy |
|-----------------|----------|
| Addition        | 95.2%    |
| Subtraction     | 93.1%    |
| Multiplication  | 89.7%    |
| Division        | 87.3%    |
| Parentheses     | 81.5%    |
| Nested          | 74.2%    |
```

### Example prompts for appendix:

**Use the output from Section 10** - it's already formatted perfectly for your report!

---

## Advanced Testing

### Test with different temperatures:

```python
for temp in [0.5, 0.7, 0.9]:
    results = test_multiple_prompts(
        model, prompts, encode, decode,
        temperature=temp
    )
    print(f"Temperature {temp}: ...")
```

### Test robustness:

```python
# Test same prompt multiple times
prompt = "15+7"
answers = []
for i in range(10):
    _, answer = test_single_prompt(
        model, prompt, encode, decode,
        temperature=0.8
    )
    answers.append(answer)

# Check consistency
print(f"Unique answers: {set(answers)}")
```

### Stress test with long expressions:

```python
long_prompts = [
    "((10+5)*2)-((3+7)//2)",
    "(((2+3)*4)+1)*((5-2)+1)",
]
```

---

## Checklist

Before submitting your assignment, test:

- [ ] Overall accuracy on test set
- [ ] Each operation type individually
- [ ] Edge cases (large numbers, nested expressions)
- [ ] Generate examples for report (correct + incorrect)
- [ ] Save all test results
- [ ] Document testing methodology
- [ ] Analyze and explain results

---

## Quick Reference

| Task | Section | Command |
|------|---------|---------|
| Load Math model | 5 | `checkpoint_path = 'checkpoints/best_model.pt'` |
| Load Boolean model | 5 | `checkpoint_path = 'checkpoints_boolean/best_model.pt'` |
| Test Math prompts | 6 | Run cell |
| Test Boolean prompts | 7 | Run cell |
| Interactive testing | 8 | `interactive_test()` |
| Generate examples | 10 | `generate_report_examples()` |

---

## Summary

The testing notebook provides everything you need to:

✅ Evaluate model accuracy
✅ Test custom prompts
✅ Generate report examples
✅ Analyze strengths and weaknesses
✅ Create professional documentation

**Remember:** Understanding why the model succeeds or fails is more important than achieving perfect accuracy!

---

*Happy Testing!* 🧪
