# Assignment Submission Checklist
## CS7CS4 Machine Learning - Final Assignment 2025-26

Use this checklist to ensure you have everything ready for submission.

---

## Before You Start

- [ ] Python 3.10 or 3.11 installed
- [ ] All dependencies installed (`pip install -r requirement.txt`)
- [ ] Environment tested (`python test_environment.py`)
- [ ] Datasets generated (120K+ math, 105K+ boolean expressions)

---

## Part 1: Math GPT (46 marks)

### Dataset (8 marks)

- [ ] Training dataset exists (`dataset/math/training/math_train.txt`)
- [ ] Testing dataset exists (`dataset/math/testing/math_test.txt`)
- [ ] Dataset covers all required operations (+, -, *, //, %, parentheses)
- [ ] Documented dataset statistics in notebook
- [ ] Explained dataset generation methodology

### Evaluation Metrics (8 marks)

- [ ] Implemented exact match accuracy
- [ ] Implemented operation-specific accuracy
- [ ] Implemented additional metrics (digit-level accuracy, etc.)
- [ ] Explained why each metric is appropriate
- [ ] Metrics calculated on test set

### Architecture Exploration (15 marks)

- [ ] Tested at least 3 different configurations
- [ ] Varied embedding dimensions
- [ ] Varied number of layers
- [ ] Varied number of attention heads
- [ ] Documented hyperparameter choices with justification
- [ ] Created comparison table of configurations
- [ ] Explained trade-offs (accuracy vs. complexity vs. training time)
- [ ] Selected final architecture with clear reasoning

### Performance Analysis (15 marks)

- [ ] Tested on all operation types
- [ ] Calculated per-operation accuracy
- [ ] Identified operations that work well
- [ ] Identified operations that fail
- [ ] Provided quantitative evidence (tables, charts)
- [ ] Included 15+ correct example predictions
- [ ] Included 15+ incorrect example predictions
- [ ] Explained why certain operations succeed/fail

### Training

- [ ] Model trained for sufficient iterations (8000-10000+)
- [ ] Training loss converging
- [ ] Test loss not diverging (no severe overfitting)
- [ ] Checkpoints saved during training
- [ ] Best model saved
- [ ] Training curves plotted

### Files Generated

- [ ] `model_weights_part1.pth` (final model weights)
- [ ] `part1_training_loss.png` (training curves)
- [ ] `part1_operation_accuracy.png` (operation accuracy chart)
- [ ] `part1_metrics.json` (summary metrics)
- [ ] Example outputs saved to file

---

## Part 2: Boolean GPT (46 marks)

### Dataset (8 marks)

- [ ] Training dataset exists (`dataset/boolean/training/boolean_train.txt`)
- [ ] Testing dataset exists (`dataset/boolean/testing/boolean_test.txt`)
- [ ] Dataset covers all required operations (AND, OR, XOR, NOT)
- [ ] Documented dataset statistics
- [ ] Explained dataset generation

### Evaluation Metrics (8 marks)

- [ ] Same metrics as Part 1 implemented
- [ ] Metrics appropriate for boolean logic
- [ ] Results documented

### Architecture Exploration (15 marks)

- [ ] Tested multiple configurations
- [ ] Compared to Part 1 architecture
- [ ] Justified any differences from Math GPT
- [ ] Explained why certain hyperparameters were chosen
- [ ] Documented experiments

### Performance Analysis (15 marks)

- [ ] Tested all boolean operations
- [ ] Per-operation accuracy calculated
- [ ] Strengths identified
- [ ] Weaknesses identified
- [ ] 15+ correct examples
- [ ] 15+ incorrect examples
- [ ] Analysis of error patterns

### Training

- [ ] Model trained sufficiently
- [ ] Convergence achieved
- [ ] Checkpoints saved
- [ ] Best model saved
- [ ] Training documented

### Files Generated

- [ ] `model_weights_part2.pth` (final model weights)
- [ ] Part 2 training curves
- [ ] Part 2 accuracy charts
- [ ] Part 2 metrics
- [ ] Example outputs saved

---

## Part 3: Discussion (8 marks)

### Comparative Analysis

- [ ] Compared Math vs Boolean GPT architectures
- [ ] Identified shared architectural elements
- [ ] Identified task-specific adaptations
- [ ] Explained why adaptations were necessary

### Task Analysis

- [ ] Discussed similarities between tasks
- [ ] Discussed differences between tasks
- [ ] Analyzed which task was easier for the model
- [ ] Explained why (with evidence)

### Impact Analysis

- [ ] Evaluated impact on model performance
- [ ] Discussed generalizability
- [ ] Considered computational costs
- [ ] Compared training times

### Failure Analysis

- [ ] Identified how models fail
- [ ] Explained failure patterns
- [ ] Discussed attempted solutions
- [ ] Identified remaining issues
- [ ] Suggested future improvements

---

## Report (5-8 pages + appendix)

### Structure

- [ ] Title page (if required)
- [ ] Introduction (0.5 pages)
- [ ] Part 1: Math GPT (2 pages)
- [ ] Part 2: Boolean GPT (2 pages)
- [ ] Part 3: Discussion (1 page)
- [ ] Conclusion (0.5 pages)
- [ ] References (if applicable)
- [ ] Appendix: Code
- [ ] Appendix: Examples

### Content Quality

- [ ] All figures have numbers and captions
- [ ] All tables have numbers and captions
- [ ] Figures are clear and legible
- [ ] Consistent notation throughout
- [ ] Professional writing style
- [ ] No AI-generated feel
- [ ] Proper grammar and spelling
- [ ] Logical flow

### Understanding Demonstrated

- [ ] Explained transformer architecture
- [ ] Explained attention mechanism
- [ ] Explained embeddings
- [ ] Explained loss function
- [ ] Showed understanding of hyperparameters
- [ ] Discussed trade-offs
- [ ] Critical analysis included
- [ ] Not just reporting numbers

### Figures and Tables

- [ ] Figure 1: Part 1 training curves
- [ ] Figure 2: Part 1 operation accuracy
- [ ] Figure 3: Part 2 training curves
- [ ] Figure 4: Part 2 operation accuracy
- [ ] Table 1: Part 1 hyperparameter comparison
- [ ] Table 2: Part 1 results by operation
- [ ] Table 3: Part 2 hyperparameter comparison
- [ ] Table 4: Part 2 results by operation
- [ ] Table 5: Comparative analysis

### Citations

- [ ] NanoGPT lecture cited
- [ ] Any external resources cited
- [ ] Citation format consistent
- [ ] No uncited sources

### Code Appendix

- [ ] Relevant code sections included
- [ ] Code is readable (proper formatting)
- [ ] Code has comments
- [ ] Not entire notebook (just key sections)
- [ ] Model architecture shown
- [ ] Training loop shown
- [ ] Evaluation functions shown

### Examples Appendix

- [ ] Part 1 correct examples (10-15)
- [ ] Part 1 incorrect examples (10-15)
- [ ] Part 2 correct examples (10-15)
- [ ] Part 2 incorrect examples (10-15)
- [ ] Clear formatting
- [ ] Shows input, expected, predicted

---

## Submission Files

### PDF Report (Separate File)

- [ ] Report is in PDF format
- [ ] PDF is correctly formatted
- [ ] All figures visible and clear
- [ ] PDF size reasonable (<10 MB)
- [ ] File named appropriately
- [ ] **IMPORTANT:** PDF uploaded as separate file, NOT in ZIP

### ZIP File Contents

**Model Weights:**
- [ ] `model_weights_part1.pth`
- [ ] `model_weights_part2.pth`

**Code:**
- [ ] `part1_math_gpt.ipynb`
- [ ] `part2_boolean_gpt.ipynb`
- [ ] `dataset_generation.ipynb`
- [ ] `evaluation_utils.py`
- [ ] `gpt.py`
- [ ] Any other custom code files

**Examples:**
- [ ] `part1_examples.txt` (or similar name)
- [ ] `part2_examples.txt` (or similar name)

**Optional but Recommended:**
- [ ] `requirement.txt`
- [ ] `README.md` with instructions
- [ ] Checkpoints (if reasonable size)

### File Verification

- [ ] All files are readable
- [ ] No corrupted files
- [ ] Notebooks run without errors
- [ ] Models load successfully
- [ ] ZIP extracts properly
- [ ] No unnecessary files (cache, .DS_Store, etc.)

---

## Declaration

- [ ] Declaration form completed
- [ ] Signed and dated
- [ ] Confirms work is your own
- [ ] Confirms no collaboration
- [ ] Submitted separately as required

---

## Pre-Submission Testing

### Code Functionality

- [ ] Extracted ZIP to fresh directory
- [ ] Can run notebooks from scratch
- [ ] Notebooks execute without errors
- [ ] Models load successfully
- [ ] Outputs match expected results

### Model Loading Test

```python
# Test Part 1
model = GPTLanguageModel()
model.load_state_dict(torch.load("model_weights_part1.pth"))
model.eval()
# Should work without errors

# Test Part 2
model = GPTLanguageModel()
model.load_state_dict(torch.load("model_weights_part2.pth"))
model.eval()
# Should work without errors
```

### Report Quality Check

- [ ] Spell-checked
- [ ] Grammar-checked
- [ ] Figures numbered correctly
- [ ] References correct
- [ ] Page count within limits (5-8 pages excluding appendix)
- [ ] PDF opens correctly
- [ ] All hyperlinks work (if any)

---

## Plagiarism Check

- [ ] All writing is in your own words
- [ ] All code is your own (except starter code)
- [ ] All external sources cited
- [ ] No copied text from tutorials
- [ ] No copied code from GitHub
- [ ] Declaration form accurate

---

## Final Checks

### Content Completeness

- [ ] All 3 parts addressed
- [ ] All sub-tasks completed
- [ ] All questions answered
- [ ] All requirements met

### Professional Quality

- [ ] Report looks professional
- [ ] Code is clean and commented
- [ ] Results are clearly presented
- [ ] Analysis is thorough

### Submission Ready

- [ ] All files named correctly
- [ ] All files in correct format
- [ ] ZIP file created
- [ ] PDF separate from ZIP
- [ ] Declaration form ready
- [ ] Ready to upload to Blackboard

---

## Upload to Blackboard

- [ ] PDF report uploaded (separate file)
- [ ] ZIP file uploaded
- [ ] Declaration form uploaded
- [ ] Submission confirmation received
- [ ] Checked submission was successful
- [ ] Downloaded and verified submitted files

---

## Recommended Timeline

### 3 Days Before Deadline

- [ ] All training complete
- [ ] All results collected
- [ ] All figures generated
- [ ] Examples selected

### 2 Days Before Deadline

- [ ] Report draft complete
- [ ] Code cleaned up
- [ ] Files organized

### 1 Day Before Deadline

- [ ] Final review of report
- [ ] Plagiarism check
- [ ] Test code functionality
- [ ] Create ZIP file
- [ ] Verify all requirements met

### Deadline Day

- [ ] Final spell check
- [ ] Generate final PDF
- [ ] Upload to Blackboard
- [ ] Verify submission
- [ ] Backup all files

---

## Common Mistakes to Avoid

- ❌ Forgetting to save model weights
- ❌ Not including expected vs predicted in examples
- ❌ Only showing correct predictions
- ❌ No justification for hyperparameter choices
- ❌ Insufficient analysis (just numbers, no explanation)
- ❌ Code appendix is entire notebook
- ❌ Figures without captions
- ❌ PDF included in ZIP file
- ❌ Missing declaration form
- ❌ Not testing code runs fresh
- ❌ Over 8 pages (excluding appendix)
- ❌ Under 5 pages
- ❌ Unclear or missing methodology
- ❌ No critical analysis
- ❌ Not demonstrating understanding

---

## Emergency Contacts

If you have questions:

1. Check assignment specification
2. Check this documentation
3. Check course materials
4. Attend office hours
5. Email course staff (well before deadline!)

---

## Final Confidence Check

Answer honestly:

- [ ] I understand how transformers work
- [ ] I can explain my architecture choices
- [ ] I know why my model succeeds/fails
- [ ] My analysis is critical, not just descriptive
- [ ] My report demonstrates understanding
- [ ] My code is my own work
- [ ] I'm confident in my submission

If you answered "yes" to all, you're ready to submit! 🎉

If any "no", review those areas before submitting.

---

## Post-Submission

- [ ] Keep backup of all files
- [ ] Save confirmation email
- [ ] Don't delete files until grades released
- [ ] Celebrate! 🎊

---

**Good luck!** 🍀

You've got this! Remember: understanding > perfection.

---

*Last Updated: December 2025*
*CS7CS4 Machine Learning - Trinity College Dublin*
