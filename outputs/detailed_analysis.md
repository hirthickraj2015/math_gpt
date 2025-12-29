# Comprehensive Evaluation Analysis
## CS7CS4 Machine Learning - Final Assignment 2025-26

---

## Task 1.2 & 2.2: Evaluation Metrics (8 marks each)

**Question**: What metrics are appropriate for evaluating symbolic reasoning models?

**Answer**:

For deterministic symbolic tasks, we use:

1. **Exact Match Accuracy**: Percentage of completely correct predictions
   - Rationale: Symbolic tasks are binary - either correct or incorrect
   - Formula: (Correct predictions / Total predictions) × 100

2. **Character-Level Accuracy**: Granular correctness measurement
   - Rationale: Partial credit for near-correct answers (e.g., '42' vs '43')
   - Formula: (Correct characters / Total characters) × 100

3. **Operation-Specific Accuracy**: Per-operation breakdown
   - Rationale: Identifies operation-specific weaknesses
   - Use: Guides targeted improvements

4. **Error Analysis**: Failure mode categorization
   - Rationale: Understanding HOW models fail informs fixes

**Why These Work**: Arithmetic and boolean logic are deterministic with no ambiguity.
These metrics provide both overall performance and diagnostic details.

---

## Task 1.4: Math GPT Analysis (15 marks)

**Question**: What operations are learned correctly and which are not? Why?

**Overall Performance**: 71.65% exact match accuracy

### Operations Breakdown:

#### DIVISION (93.9% accurate)
- Tested: 214 expressions
- Correct: 201
- Accuracy: 93.9%
- Correct examples:
  - `6//4=1` ✓
  - `0//3=0` ✓
  - `8//2=4` ✓
- Incorrect examples:
  - `9//2+1=1` ✗ (expected: 5)
  - `6//1+5=7` ✗ (expected: 11)
  - `7//4=2` ✗ (expected: 1)

#### MODULO (91.2% accurate)
- Tested: 194 expressions
- Correct: 177
- Accuracy: 91.2%
- Correct examples:
  - `4%2=0` ✓
  - `4%9=4` ✓
  - `4%2=0` ✓
- Incorrect examples:
  - `1%8+3=3` ✗ (expected: 4)
  - `4%4=4` ✗ (expected: 0)
  - `7%3=0` ✗ (expected: 1)

#### ADDITION (82.9% accurate)
- Tested: 480 expressions
- Correct: 398
- Accuracy: 82.9%
- Correct examples:
  - `7+7=14` ✓
  - `47+47=94` ✓
  - `89+22=111` ✓
- Incorrect examples:
  - `37+65=92` ✗ (expected: 102)
  - `16+95=101` ✗ (expected: 111)
  - `3+9=14` ✗ (expected: 12)

#### MULTIPLICATION (81.9% accurate)
- Tested: 199 expressions
- Correct: 163
- Accuracy: 81.9%
- Correct examples:
  - `1*9=9` ✓
  - `5*2=10` ✓
  - `0*0=0` ✓
- Incorrect examples:
  - `9*5=35` ✗ (expected: 45)
  - `9*9=84` ✗ (expected: 81)
  - `4*1=5` ✗ (expected: 4)

#### SUBTRACTION (74.8% accurate)
- Tested: 492 expressions
- Correct: 368
- Accuracy: 74.8%
- Correct examples:
  - `78-35=43` ✓
  - `32-74=-42` ✓
  - `87-22=65` ✓
- Incorrect examples:
  - `25-16=17` ✗ (expected: 9)
  - `8-6=4` ✗ (expected: 2)
  - `1-0=13` ✗ (expected: 1)

#### MIXED_OPS (36.9% accurate)
- Tested: 141 expressions
- Correct: 52
- Accuracy: 36.9%
- Correct examples:
  - `4+0*7=4` ✓
  - `3+6*0=3` ✓
  - `2+9*9=83` ✓
- Incorrect examples:
  - `8*5-9=39` ✗ (expected: 31)
  - `7+4*3=17` ✗ (expected: 19)
  - `6*2+6=16` ✗ (expected: 18)

#### PARENTHESES (26.4% accurate)
- Tested: 280 expressions
- Correct: 74
- Accuracy: 26.4%
- Correct examples:
  - `6*(6+0)=36` ✓
  - `(2+2)-6=-2` ✓
  - `(6*8)+8=56` ✓
- Incorrect examples:
  - `(4-4)*5=-8` ✗ (expected: 0)
  - `(2+5)*(1+1)=12` ✗ (expected: 14)
  - `(3+4)+9=15` ✗ (expected: 16)

### Why These Results?

**Operations with High Accuracy (>70%)**:
- Division and Modulo: Small output space (0-9), easy to memorize
- Parentheses (if trained well): Clear structural patterns

**Operations with Moderate Accuracy (40-70%)**:
- Multiplication: Larger output space (0-81+), times tables
- Mixed operations: Requires BODMAS understanding

**Operations with Low Accuracy (<40%)**:
- Subtraction: Negative numbers confuse character-level models
  - Example error: '1-5=-5' instead of '-4' (magnitude error)
- Addition (if low): Carrying mechanism not learned

### Root Causes:
1. **Pattern Matching vs Computation**: Model memorizes, doesn't calculate
2. **Output Space Size**: Smaller ranges = easier memorization
3. **Character-Level Issues**: Multi-digit numbers treated as sequences
4. **Negative Number Confusion**: '-' is both operator and sign
5. **No Algorithmic Understanding**: No built-in arithmetic circuits

---

## Task 2.4: Boolean GPT Analysis (15 marks)

**Question**: What operations are learned correctly and which are not? Why?

**Overall Performance**: 89.20% exact match accuracy

### Operations Breakdown:

#### AND (100.0% accurate)
- Tested: 44 expressions
- Correct: 44
- Accuracy: 100.0%
- Correct examples:
  - `True AND False=False` ✓
  - `True AND True=True` ✓
  - `True AND True=True` ✓

#### OR (95.7% accurate)
- Tested: 47 expressions
- Correct: 45
- Accuracy: 95.7%
- Correct examples:
  - `True OR True=True` ✓
  - `False OR False OR False=False` ✓
  - `False OR True=True` ✓
- Incorrect examples:
  - `True OR False=False` ✗ (expected: True)
  - `False OR False=True` ✗ (expected: False)

#### NOT_COMBINED (95.5% accurate)
- Tested: 22 expressions
- Correct: 21
- Accuracy: 95.5%
- Correct examples:
  - `NOT False OR True=True` ✓
  - `False AND NOT False=False` ✓
  - `True XOR NOT True=True` ✓
- Incorrect examples:
  - `NOT True XOR False=True` ✗ (expected: False)

#### XOR (94.4% accurate)
- Tested: 54 expressions
- Correct: 51
- Accuracy: 94.4%
- Correct examples:
  - `False XOR True=True` ✓
  - `True XOR False=True` ✓
  - `False XOR True=True` ✓
- Incorrect examples:
  - `True XOR False OR True=False` ✗ (expected: True)
  - `True XOR True XOR True=False` ✗ (expected: True)
  - `False AND False XOR True=False` ✗ (expected: True)

#### PARENTHESES (89.3% accurate)
- Tested: 262 expressions
- Correct: 234
- Accuracy: 89.3%
- Correct examples:
  - `True OR ((False AND False) XOR False)=True` ✓
  - `(True OR False) OR (True OR False)=True` ✓
  - `((True AND False) AND False) AND False=False` ✓
- Incorrect examples:
  - `((False OR False) XOR True) XOR False=False` ✗ (expected: True)
  - `((True XOR True) OR False) XOR True=False` ✗ (expected: True)
  - `(True XOR True) XOR True=False` ✗ (expected: True)

#### NOT (71.8% accurate)
- Tested: 71 expressions
- Correct: 51
- Accuracy: 71.8%
- Correct examples:
  - `NOT NOT True=True` ✓
  - `NOT NOT NOT True=False` ✓
  - `NOT NOT NOT True=False` ✓
- Incorrect examples:
  - `NOT NOT False=True` ✗ (expected: False)
  - `NOT NOT True=False` ✗ (expected: True)
  - `NOT NOT True=False` ✗ (expected: True)

### Why Boolean GPT Performs Better:
1. **Smaller Output Space**: Only 'True' or 'False'
2. **Simpler Patterns**: Boolean algebra has fewer rules than arithmetic
3. **No Numeric Complexity**: No carrying, borrowing, or multi-digit issues
4. **Exhaustive Coverage**: Small input space (2 values) easily covered

---

## Task 3.1: Critical Comparison (8 marks)

**Question**: Compare Math GPT vs Boolean GPT architectures

### Performance Comparison:
- **Math GPT**: 71.65% accuracy
- **Boolean GPT**: 89.20% accuracy
- **Winner**: Boolean GPT by 17.55 percentage points

### Architectural Similarities (What Worked for Both):
1. **Character-Level Tokenization**: Each symbol is atomic and meaningful
2. **Small Embeddings**: Limited vocabulary doesn't need large embeddings
   - Math: 64 dimensions, Boolean: 32 dimensions
3. **Shallow Architecture**: 2 layers sufficient for symbolic tasks
4. **Small Block Size**: Most expressions < 50 characters
   - Math: 32, Boolean: 48 (longer for 'True AND False' format)
5. **Light Dropout**: Minimal regularization (0.05-0.1)

### Task-Specific Adaptations:
#### Math GPT:
- Larger embeddings (64): More complex numeric patterns
- More heads (4): Captures multiple attention patterns
- Higher dropout (0.1): Prevents overfitting on arithmetic patterns

#### Boolean GPT:
- Smaller embeddings (32): Simpler true/false patterns
- Fewer heads (2): Less attention diversity needed
- Longer context (48): Accommodates verbose boolean strings
- Lower dropout (0.05): Small task space, less overfitting risk

### Key Insights:
1. **Task Complexity Matters**: Boolean logic (2 values) is simpler than arithmetic (infinite values)
2. **Output Space Drives Difficulty**: Smaller output space = higher accuracy
3. **Pattern Matching ≠ Understanding**: Models memorize, don't reason
4. **Architecture Should Match Task**: Simpler tasks need simpler models
5. **Symbolic Tasks Suit Small Models**: No need for GPT-3 scale

### Limitations of Both:
- No true algorithmic reasoning
- Struggle with out-of-distribution examples
- Character-level tokenization limits number understanding
- Cannot explain their reasoning
- Memorization-based, not computation-based
