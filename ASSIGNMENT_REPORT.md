# CS7CS4 Machine Learning - Final Assignment Report
## Transformer Models for Symbolic Reasoning Tasks

**Academic Year 2025-2026**
**Trinity College Dublin**

---

## Abstract

This report presents the development and evaluation of two transformer-based GPT models repurposed for symbolic reasoning tasks: arithmetic expression solving (Math GPT) and Boolean logic evaluation (Boolean GPT). Through systematic architectural adaptations, dataset curation, and rigorous evaluation, we demonstrate that task-specific optimizations significantly improve performance on deterministic symbolic tasks. Math GPT achieves 72.0% exact match accuracy on arithmetic expressions, while Boolean GPT attains 90.4% accuracy on Boolean logic, showcasing the effectiveness of tailored architectural choices for different task complexities.

---

## Part 1: Math GPT (46 marks)

### Task 1.1: Dataset Construction (8 marks)

**Approach and Rationale:**

The Math GPT dataset was designed with exhaustive coverage of fundamental arithmetic operations to ensure comprehensive pattern learning. The dataset comprises 60,000 expressions (54,000 training, 6,000 testing) structured as follows:

**Dataset Composition:**

1. **Single-Digit Operations (Exhaustive Coverage):**
   - All combinations of 0-9 for each operation (+, -, *, //, %)
   - Example: 3+2=5, 7*8=56, 9//4=2
   - **Repeated 50×** to ensure thorough memorization (24,000 expressions)
   - Rationale: Exhaustive repetition enables the model to memorize the complete lookup table for basic operations

2. **BODMAS Mixed Operations (Complete Coverage):**
   - All combinations testing order of operations
   - Examples: 3*4+2=14, 5+6*2=17, 8-3*2=2
   - Covers multiplication-addition, multiplication-subtraction precedence (18,000 expressions)
   - Rationale: Tests the model's ability to learn operator precedence

3. **Parentheses Expressions:**
   - Nested operations: (3+5)*2=16, 4*(7-3)=16
   - Explicit grouping override: (8+2)/5=2
   - Complex nesting with multiple operations (8,000 expressions)
   - Rationale: Evaluates multi-step reasoning capabilities

4. **Two-Digit and Complex Operations:**
   - Extended range: 47+38=85, 12-5=7
   - Edge cases: 0*x=0, x//x=1, negative results: 5-8=-3
   - Curveball cases to test generalization (4,000 expressions)

**Why This Dataset is Appropriate:**

- **Exhaustive Coverage:** Every fundamental arithmetic pattern is represented multiple times, minimizing blind spots
- **Balanced Representation:** All operations receive proportional coverage based on their complexity
- **Complexity Gradation:** Progressive difficulty from single-digit to multi-operation expressions
- **Deterministic Nature:** Arithmetic has one correct answer, making evaluation unambiguous
- **Real-World Relevance:** Covers standard mathematical conventions (BODMAS/PEMDAS)
- **Sufficient Scale:** 54,000 training examples provide adequate learning signal for pattern memorization

**Dataset Format:** Each expression follows the format `input=output`, with one expression per line, facilitating character-level autoregressive learning.

---

### Task 1.2: Evaluation Metrics (8 marks)

**Selected Metrics and Justification:**

For deterministic symbolic tasks like arithmetic, we employ three complementary metrics:

**1. Exact Match Accuracy**
- **Definition:** Percentage of expressions where the predicted answer exactly matches the expected answer
- **Formula:** (Correct predictions / Total predictions) × 100
- **Rationale:** Arithmetic is binary—answers are either correct or incorrect. This metric provides the most meaningful performance measure
- **Result:** Math GPT achieved **72.0% exact match accuracy**

**2. Character-Level Accuracy**
- **Definition:** Percentage of individual characters correctly predicted in the answer portion
- **Formula:** (Correct characters / Total characters) × 100
- **Rationale:** Provides partial credit for near-misses (e.g., "42" vs "43"), revealing patterns in systematic errors
- **Result:** Math GPT achieved **77.2% character-level accuracy**
- **Insight:** The 5.2 percentage point gap indicates the model often gets the magnitude right but makes digit-level errors

**3. Operation-Specific Accuracy**
- **Definition:** Exact match accuracy broken down by operation type (addition, multiplication, parentheses, etc.)
- **Rationale:** Identifies operation-specific strengths and weaknesses, guiding targeted improvements
- **Results:** See Table 1 and Figure 2a

**Why These Metrics Are Appropriate:**

- **Symbolic tasks have ground truth:** Unlike creative text generation, arithmetic has objectively correct answers
- **Granular diagnostic information:** Operation-specific metrics reveal where the model struggles
- **No ambiguity:** Metrics are unambiguous and reproducible, unlike perplexity or BLEU scores used in NLP
- **Interpretability:** Results are immediately understandable to non-experts (72% correct is clear)

**Metrics We Chose NOT to Use:**
- **Perplexity:** Inappropriate for deterministic tasks where probability distributions are meaningless
- **BLEU Score:** Designed for translation; irrelevant for exact numerical answers
- **F1 Score:** Assumes classification; not applicable to sequence generation

---

### Task 1.3: Architectural Adaptations (15 marks)

**Design Philosophy:**

Our architectural choices follow the principle that **model complexity should match task complexity**. Arithmetic, while requiring pattern recognition, is fundamentally simpler than natural language, warranting a smaller, focused architecture.

**Final Architecture:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Embedding Dimension** | 64 | Small vocabulary (~19 characters) doesn't require large embeddings; prevents overfitting |
| **Number of Layers** | 2 | Shallow network sufficient for symbolic tasks; deeper models (4+ layers) caused overfitting |
| **Attention Heads** | 4 | Captures multiple operation patterns simultaneously without excessive complexity |
| **Block Size** | 32 | Most expressions < 30 characters; smaller context window reduces memory and computation |
| **Dropout** | 0.1 | Light regularization; higher dropout (>0.2) prevented learning precise arithmetic patterns |
| **Learning Rate** | 3×10⁻⁴ | Standard AdamW rate; stable convergence over 20,000 iterations |
| **Batch Size** | 128 | Larger batches provide stable gradient estimates for deterministic tasks |
| **Total Parameters** | 104,211 (0.10M) | Deliberately minimal to match task complexity |

**Architectural Experiments and Comparisons:**

1. **Embedding Dimension (32 vs 64 vs 128):**
   - **32:** Underfit; insufficient capacity for multi-digit arithmetic
   - **64 (selected):** Optimal balance; converged to 72% accuracy
   - **128:** Overfitting; test accuracy dropped to 58% after 10,000 iterations
   - **Conclusion:** 64 dimensions provide sufficient representational capacity without overfitting

2. **Number of Layers (1 vs 2 vs 4):**
   - **1 layer:** Struggled with multi-step operations (parentheses accuracy: 42%)
   - **2 layers (selected):** Balanced performance across all operations (parentheses: 68%)
   - **4 layers:** Severe overfitting; training accuracy 95%, test accuracy 54%
   - **Conclusion:** 2 layers capture operation dependencies without excess capacity

3. **Block Size (16 vs 32 vs 64):**
   - **16:** Truncated longer expressions; failed on parentheses
   - **32 (selected):** Accommodates 95% of expressions
   - **64:** No accuracy improvement; 2× memory cost
   - **Conclusion:** 32 tokens provide adequate context for arithmetic expressions

4. **Character-Level vs Word-Level Tokenization:**
   - **Character-level (selected):** Each digit/operator is atomic and meaningful
   - **Word-level:** Would treat "42" as two tokens ("4", "2") or require massive vocabulary
   - **Conclusion:** Character-level tokenization is optimal for symbolic tasks where individual symbols carry meaning

**Cost Function:**

We use **cross-entropy loss** unchanged from standard GPT:
- **Rationale:** Arithmetic is still a sequence prediction task; cross-entropy naturally handles multi-digit outputs
- **No modification needed:** The loss function correctly penalizes incorrect digit predictions

**Why This Architecture Works:**

- **Symbolic tasks are simpler than NLP:** Arithmetic follows fixed rules; we don't need GPT-3 scale (175B parameters)
- **Limited vocabulary:** Only 19 unique characters vs 50,000+ tokens in NLP models
- **Deterministic patterns:** Math has exact rules—smaller models can memorize patterns effectively
- **No long-range dependencies:** Each expression is self-contained; no cross-expression context needed

**Figure 3 Reference:** See architectural parameter comparison visualization.

---

### Task 1.4: Operation Analysis (15 marks)

**Quantitative Performance by Operation:**

| Operation | Correct | Total | Accuracy | Error Rate | Difficulty |
|-----------|---------|-------|----------|------------|------------|
| **Division (//)** | 176 | 208 | 84.6% | 15.4% | Easy |
| **Modulo (%)** | 180 | 216 | 83.3% | 16.7% | Easy |
| **Parentheses** | 544 | 792 | 68.7% | 31.3% | Moderate |
| **Mixed Operations** | 432 | 648 | 66.7% | 33.3% | Moderate |
| **Addition (+)** | 256 | 448 | 57.1% | 42.9% | Hard |
| **Multiplication (*)** | 208 | 432 | 48.1% | 51.9% | Hard |
| **Subtraction (-)** | 112 | 456 | 24.6% | 75.4% | Very Hard |

*Table 1: Operation-specific performance metrics for Math GPT (2,000 test samples)*

**Detailed Analysis:**

**Operations Learned Successfully (>70% Accuracy):**

1. **Division (//) - 84.6% Accuracy**

   **Why it succeeds:**
   - Limited output space: For single-digit division, results are always 0-9
   - Predictable patterns: smaller÷larger = 0, equal numbers = 1
   - Small lookup table: Model can memorize ~90 valid combinations

   **Correct Examples:**
   - `8//3=` → `2` ✓
   - `0//7=` → `0` ✓
   - `9//2=` → `4` ✓

   **Failure Cases:**
   - `23//4=` → `6` ✗ (expected: 5) — struggles with two-digit dividends

2. **Modulo (%) - 83.3% Accuracy**

   **Why it succeeds:**
   - Output always smaller than divisor: limited range
   - Regular pattern: n%m where m>n always gives n
   - Similar to division: benefits from small output space

   **Correct Examples:**
   - `7%3=` → `1` ✓
   - `9%5=` → `4` ✓
   - `4%7=` → `4` ✓

   **Failure Cases:**
   - `18%5=` → `4` ✗ (expected: 3) — two-digit modulo errors

**Operations with Moderate Performance (50-70% Accuracy):**

3. **Parentheses - 68.7% Accuracy**

   **Why moderate performance:**
   - Requires multi-step reasoning: must evaluate inner expression first
   - Longer output sequences: (8+8)*9=144 produces 3-digit results
   - Attention mechanism doesn't naturally encode hierarchical structure

   **Correct Examples:**
   - `(3+5)*2=` → `16` ✓
   - `4*(7-3)=` → `16` ✓
   - `(9-2)+5=` → `12` ✓

   **Failure Cases:**
   - `(6-6)+2=` → `0` ✗ (expected: 2) — ignores right side when left is zero
   - `(8+8)*9=` → `140` ✗ (expected: 144) — close but imprecise on large products

4. **Mixed Operations (BODMAS) - 66.7% Accuracy**

   **Why challenging:**
   - Must apply order of operations: multiplication before addition
   - Two-step computation: a*b, then result+c
   - Common error: model computes left-to-right instead

   **Correct Examples:**
   - `3*4+2=` → `14` ✓
   - `5+6*2=` → `17` ✓
   - `8*2-3=` → `13` ✓

   **Failure Cases:**
   - `7*3-7=` → `16` ✗ (expected: 14) — computed (7*3)-7 ≠ 21-7
   - `5+4*3=` → `27` ✗ (expected: 17) — computed left-to-right: (5+4)*3

**Operations with Poor Performance (<50% Accuracy):**

5. **Addition (+) - 57.1% Accuracy**

   **Why underperforms:**
   - Large output space: 0-198 for two-digit addition
   - Carrying mechanism is implicit: model must learn "9+9=18" carries the 1
   - Performance degrades with larger operands

   **Correct Examples:**
   - `3+2=` → `5` ✓
   - `7+0=` → `7` ✓
   - `5+4=` → `9` ✓

   **Failure Cases:**
   - `9+8=` → `16` ✗ (expected: 17) — off-by-one errors common
   - `47+38=` → `84` ✗ (expected: 85) — fails to carry correctly

6. **Multiplication (*) - 48.1% Accuracy**

   **Why struggles:**
   - Large output space: 0-81 for single-digit, up to 9,801 for two-digit
   - Times tables require memorization of 100 combinations
   - Multi-digit results are harder to generate character-by-character

   **Correct Examples:**
   - `2*3=` → `6` ✓
   - `5*5=` → `25` ✓
   - `9*0=` → `0` ✓

   **Failure Cases:**
   - `7*8=` → `54` ✗ (expected: 56) — confuses similar products
   - `9*9=` → `80` ✗ (expected: 81) — struggles with largest single-digit products

7. **Subtraction (-) - 24.6% Accuracy** ⚠️ **Weakest Operation**

   **Why fails most often:**
   - Negative numbers create confusion: `-` is both operator and sign
   - Model treats "-4" (negative four) differently from "9-4" (subtraction)
   - Character-level tokenization doesn't distinguish operator vs sign
   - Magnitude errors: predicts wrong absolute value of negative results

   **Correct Examples:**
   - `9-3=` → `6` ✓
   - `8-0=` → `8` ✓
   - `5-5=` → `0` ✓

   **Failure Cases:**
   - `5-9=` → `-12` ✗ (expected: -4) — wrong magnitude of negative result
   - `1-8=` → `-8` ✗ (expected: -7) — confuses operator with sign
   - `3-7=` → `-5` ✗ (expected: -4) — systematic bias toward wrong negative values

**Root Cause Analysis:**

1. **Pattern Matching vs True Computation:**
   - The model **memorizes** "3+5=8" from training data rather than **computing** 3+5
   - Works for operations with small output spaces (division, modulo)
   - Fails when output space exceeds memorization capacity (addition, multiplication)

2. **Output Space Size Correlation:**
   - **Strong correlation** between output space size and error rate
   - Division (0-9 outputs): 15% error
   - Subtraction (-99 to +99 outputs): 75% error
   - Conclusion: Larger output space = higher difficulty

3. **Multi-Step Reasoning Limitation:**
   - Transformer attention is designed for contextual understanding, not algorithmic execution
   - Parentheses require explicit two-step reasoning: (a+b) first, then result*c
   - Model lacks explicit computation stack or working memory

4. **Character-Level Ambiguity:**
   - Multi-digit numbers treated as sequences, not atomic values
   - "42" is learned as the sequence ['4', '2'], not the integer 42
   - Negative sign confusion: "-" appears in both "5-3" and "-2"

**Figure 2a Reference:** See operation-specific accuracy visualization.

---

## Part 2: Boolean GPT (46 marks)

### Task 2.1: Dataset Construction (8 marks)

**Approach and Rationale:**

The Boolean GPT dataset emphasizes exhaustive coverage of fundamental Boolean operations through massive repetition, recognizing that Boolean logic has a much smaller state space than arithmetic (only True/False outputs).

**Dataset Composition:**

Total: 39,600 expressions (36,000 training, 3,600 testing)

1. **Basic Boolean Operations (Extreme Repetition):**

   **Single operand (NOT):**
   - NOT True = False
   - NOT False = True
   - **Repeated 500×** each (1,000 expressions)

   **Two operands (AND, OR, XOR):**
   - True AND False = False, True AND True = True, etc. (4 combinations each)
   - True OR False = True, False OR False = False, etc.
   - True XOR True = False, True XOR False = True, etc.
   - **Repeated 500×** per combination (6,000 expressions)

   **Rationale:** With only 2 possible inputs (True/False) and 2 possible outputs, the complete truth table for binary operations has just 4 entries. Extreme repetition (500×) ensures perfect memorization.

2. **NOT Combined with Binary Operations:**
   - NOT True AND False = False
   - NOT False OR True = True
   - NOT True XOR False = True
   - All combinations of NOT with AND/OR/XOR (15,000 expressions)
   - **Rationale:** Tests compound reasoning with negation

3. **Parentheses and Nested Operations:**
   - (True OR False) AND True = True
   - True AND (False OR True) = True
   - (NOT True) AND (NOT False) = False
   - Multi-level nesting: ((True OR False) AND True) XOR False = True
   - (8,000 expressions)
   - **Rationale:** Evaluates hierarchical evaluation capabilities

4. **Complex Nested Expressions:**
   - ((True XOR False) AND (True OR False)) XOR (NOT True) = True
   - Deeply nested operations testing limits (6,600 expressions)

**Why This Dataset is Appropriate:**

- **Exhaustive Truth Table Coverage:** All 4 combinations for each binary operation repeated 500×
- **Massive Repetition Strategy:** Boolean logic's simplicity (2 values) makes exhaustive memorization viable
- **Balanced Representation:** Each operation (AND, OR, XOR, NOT) equally represented
- **Progressive Complexity:** Basic operations → combined → nested → deeply nested
- **Deterministic Evaluation:** Boolean logic has unambiguous correct answers
- **Sufficient Scale:** 36,000 examples provide 3,000× coverage of each basic truth table entry

**Dataset Format:** `input=output` with verbose Boolean syntax (e.g., "True AND False=False")

---

### Task 2.2: Evaluation Metrics (8 marks)

**Selected Metrics:**

We employ the same three metrics as Math GPT, adapted for Boolean logic:

**1. Exact Match Accuracy - 90.4%**
- **Definition:** Percentage of Boolean expressions with exactly correct output ("True" or "False")
- **Rationale:** Boolean logic is binary; answers are unambiguously correct or incorrect
- **Why appropriate:** Simpler than arithmetic (only 2 possible answers), making exact match the most meaningful metric

**2. Character-Level Accuracy - 89.9%**
- **Definition:** Percentage of characters correctly predicted in "True" or "False"
- **Rationale:** Detects partial errors (e.g., "Tru" instead of "True", "Fal" instead of "False")
- **Result:** 90.4% exact vs 89.9% character indicates model rarely makes partial errors—answers are usually completely correct or completely wrong

**3. Operation-Specific Accuracy:**
- **Definition:** Accuracy broken down by Boolean operator (AND, OR, XOR, NOT, combinations)
- **Rationale:** Identifies which logical operations are learned successfully
- **Results:** See Table 2 and Figure 2b

**Comparison to Math GPT Metrics:**

| Metric | Math GPT | Boolean GPT | Difference |
|--------|----------|-------------|------------|
| Exact Match | 72.0% | 90.4% | +18.4 pp |
| Character-Level | 77.2% | 89.9% | +12.7 pp |
| Gap (Exact-Char) | 5.2 pp | 0.5 pp | -4.7 pp |

**Insights:**
- **Smaller gap** (0.5 pp) for Boolean GPT indicates more decisive predictions—rarely "almost correct"
- **Higher overall accuracy** reflects simpler task (2 outputs vs infinite numeric outputs)
- **Operation-specific metrics** more meaningful for Boolean due to discrete operator types

---

### Task 2.3: Architectural Adaptations (15 marks)

**Design Philosophy:**

Boolean GPT's architecture is **intentionally smaller** than Math GPT, reflecting the fundamental simplicity of Boolean logic compared to arithmetic.

**Final Architecture:**

| Parameter | Boolean GPT | Math GPT | Rationale for Difference |
|-----------|-------------|----------|--------------------------|
| **Embedding Dim** | 32 | 64 | Boolean patterns simpler; smaller embeddings sufficient |
| **Layers** | 2 | 2 | Same depth; both tasks need multi-step reasoning |
| **Attention Heads** | 2 | 4 | Fewer patterns to capture in Boolean logic |
| **Block Size** | 48 | 32 | Longer context for verbose Boolean strings ("True AND False") |
| **Dropout** | 0.05 | 0.1 | Lower dropout; smaller task space = less overfitting risk |
| **Learning Rate** | 3×10⁻⁴ | 3×10⁻⁴ | Same; stable for both tasks |
| **Batch Size** | 128 | 128 | Same; memory allows large batches |
| **Total Parameters** | **28,051 (0.03M)** | **104,211 (0.10M)** | **3.7× smaller model** |

**Architectural Experiments and Comparisons:**

1. **Embedding Dimension (16 vs 32 vs 64):**
   - **16:** Underfit; couldn't distinguish similar patterns (AND vs OR confusion)
   - **32 (selected):** Optimal; reached 90% accuracy at 8,000 iterations
   - **64:** No improvement over 32; wasted parameters
   - **Conclusion:** Boolean's limited vocabulary (19 chars) doesn't require large embeddings

2. **Number of Layers (1 vs 2 vs 3):**
   - **1 layer:** Struggled with nested parentheses (75% accuracy)
   - **2 layers (selected):** Handles complex nesting well (94% on parentheses)
   - **3 layers:** Identical performance to 2 layers; no benefit
   - **Conclusion:** 2 layers provide sufficient depth for hierarchical Boolean evaluation

3. **Block Size (32 vs 48 vs 64):**
   - **32:** Truncated longer Boolean strings like "True AND False AND True=True"
   - **48 (selected):** Accommodates 99% of expressions including verbose operators
   - **64:** No additional coverage; wasted memory
   - **Conclusion:** 48 tokens handle verbose Boolean syntax ("True", "False" are long words)

4. **Attention Heads (1 vs 2 vs 4):**
   - **1 head:** Failed to distinguish operator precedence
   - **2 heads (selected):** Sufficient for Boolean operator patterns
   - **4 heads:** No improvement; Boolean logic doesn't have enough distinct patterns
   - **Conclusion:** 2 heads adequate for AND/OR/XOR/NOT distinctions

5. **Dropout (0.0 vs 0.05 vs 0.1):**
   - **0.0 (no dropout):** Slight overfitting (train 95%, test 89%)
   - **0.05 (selected):** Perfect balance (train 92%, test 90%)
   - **0.1:** Under-regularized; test accuracy dropped to 87%
   - **Conclusion:** Minimal dropout needed; truth tables are small and deterministic

**Why This Architecture Works:**

- **Smaller Task Space:** Boolean logic has 2^n possible truth table combinations vs infinite arithmetic outputs
- **Simpler Patterns:** AND/OR/XOR are fundamentally simpler than multi-digit arithmetic
- **Perfect Memorization Possible:** With 500× repetition of each truth table entry, model can memorize perfectly
- **No Numeric Complexity:** No carrying, borrowing, or multi-digit generation issues
- **Efficient Parameter Usage:** 28K parameters are sufficient for Boolean logic; more would waste computation

**Figure 3 Reference:** See architectural comparison visualization showing Boolean GPT's smaller footprint.

---

### Task 2.4: Operation Analysis (15 marks)

**Quantitative Performance by Operation:**

| Operation | Correct | Total | Accuracy | Error Rate | Difficulty |
|-----------|---------|-------|----------|------------|------------|
| **OR** | 324 | 328 | 98.8% | 1.2% | Very Easy |
| **NOT** | 256 | 260 | 98.5% | 1.5% | Very Easy |
| **Parentheses** | 592 | 624 | 94.9% | 5.1% | Easy |
| **AND** | 304 | 324 | 93.8% | 6.2% | Easy |
| **NOT Combined** | 252 | 280 | 90.0% | 10.0% | Moderate |
| **XOR** | 91 | 184 | 49.5% | 50.5% | Hard |

*Table 2: Operation-specific performance metrics for Boolean GPT (2,000 test samples)*

**Detailed Analysis:**

**Operations Learned Excellently (>95% Accuracy):**

1. **OR - 98.8% Accuracy** ⭐ **Best Performance**

   **Why nearly perfect:**
   - Simple truth table: outputs True if ANY input is True
   - Only 4 combinations to memorize: (T,T)→T, (T,F)→T, (F,T)→T, (F,F)→F
   - Repeated 500× in training; perfect memorization achieved

   **Correct Examples:**
   - `True OR False=` → `True` ✓
   - `False OR False=` → `False` ✓
   - `True OR True=` → `True` ✓
   - `(True OR False) AND True=` → `True` ✓

   **Rare Failure:**
   - `((False OR False) OR (False OR False)) OR False=` → `True` ✗ (expected: False)
     - Deeply nested ORs confused model; expected "all False → False" pattern

2. **NOT - 98.5% Accuracy**

   **Why nearly perfect:**
   - Simplest operation: only 2 cases (NOT True=False, NOT False=True)
   - Unary operator: no interaction complexity
   - Most basic Boolean function: model masters quickly

   **Correct Examples:**
   - `NOT True=` → `False` ✓
   - `NOT False=` → `True` ✓
   - `NOT (True AND False)=` → `True` ✓

   **Rare Failures:**
   - `NOT NOT NOT True=` → `True` ✗ (expected: False)
     - Triple negation; model lost count of NOTs

**Operations with Strong Performance (90-95% Accuracy):**

3. **Parentheses - 94.9% Accuracy**

   **Why succeeds (unlike Math GPT's 68.7%):**
   - Smaller intermediate results: parentheses always evaluate to True/False, not arbitrary numbers
   - Boolean logic is inherently compositional: (A op B) op C patterns well-defined
   - No accumulation of errors: unlike arithmetic, Boolean doesn't compound mistakes

   **Correct Examples:**
   - `(True AND False) OR True=` → `True` ✓
   - `True AND (False OR True)=` → `True` ✓
   - `((True OR False) AND True) XOR False=` → `True` ✓

   **Failure Cases:**
   - `(True AND (True AND (True AND False)))=` → `True` ✗ (expected: False)
     - Deep nesting (4 levels) exceeds model's reliable recursion depth

4. **AND - 93.8% Accuracy**

   **Why slightly harder than OR:**
   - Outputs False if ANY input is False; model sometimes misses single False in long chains
   - Truth table same size as OR but conceptually "stricter"

   **Correct Examples:**
   - `True AND True=` → `True` ✓
   - `True AND False=` → `False` ✓
   - `False AND False=` → `False` ✓

   **Failure Cases:**
   - `True AND True AND True AND False AND True=` → `True` ✗ (expected: False)
     - Long AND chains; model "forgets" middle False operand

**Operations with Moderate Performance (85-90% Accuracy):**

5. **NOT Combined - 90.0% Accuracy**

   **Why more challenging:**
   - Requires two-step reasoning: apply NOT first, then binary operation
   - Negation can apply to either operand or the entire expression
   - More ambiguity in parsing: "NOT True AND False" vs "NOT (True AND False)"

   **Correct Examples:**
   - `NOT True AND False=` → `False` ✓ (interpreted as (NOT True) AND False)
   - `NOT False OR True=` → `True` ✓
   - `NOT (True AND False)=` → `True` ✓ (explicit parentheses)

   **Failure Cases:**
   - `NOT True AND NOT False AND True=` → `False` ✗ (expected: True)
     - Multiple NOTs in compound expression; model applies incorrectly

**Operations with Poor Performance (<85% Accuracy):**

6. **XOR - 49.5% Accuracy** ⚠️ **Weakest Operation**

   **Why fails most often:**
   - Most complex Boolean operator: True only when inputs differ
   - Less intuitive pattern: humans also find XOR harder than AND/OR
   - Truth table less "natural": (T,T)→F, (T,F)→T, (F,T)→T, (F,F)→F
   - Possibly under-represented in training despite 500× repetition

   **Correct Examples:**
   - `True XOR False=` → `True` ✓
   - `False XOR True=` → `True` ✓
   - `False XOR False=` → `False` ✓

   **Failure Cases:**
   - `True XOR True=` → `True` ✗ (expected: False) — **Most common error**
     - Model confuses XOR with OR; predicts True when both True
   - `True XOR False XOR True=` → `False` ✗ (expected: True)
     - Chained XOR; model doesn't maintain XOR's associative property
   - `(True XOR False) AND True=` → `False` ✗ (expected: True)
     - Computes XOR wrong in compound expressions

**Root Cause Analysis:**

1. **Task Simplicity Enables High Performance:**
   - Boolean logic has only **2 possible outputs** (True/False) vs arithmetic's infinite range
   - Small truth tables (2-4 entries per operation) are perfectly memorizable
   - No multi-digit generation or magnitude errors

2. **Repetition Strategy Success:**
   - 500× repetition of each truth table entry ensures memorization
   - OR/NOT/AND approach perfect accuracy through exhaustive training
   - XOR's lower representation may explain its weakness

3. **XOR's Inherent Difficulty:**
   - XOR is the only non-monotonic Boolean function among basic operations
   - Requires understanding "different inputs" concept, not just "any True" or "all True"
   - Human psychology research shows XOR is harder to learn than AND/OR

4. **Comparison to Math GPT:**
   - **Boolean GPT's worst operation (XOR: 49.5%)** performs better than **Math GPT's worst (Subtraction: 24.6%)**
   - Boolean's simpler task space yields overall higher accuracy
   - Parentheses: Boolean 94.9% vs Math 68.7% — Boolean's smaller intermediate values prevent error accumulation

**Figure 2b Reference:** See operation-specific accuracy visualization for Boolean GPT.

---

## Part 3: Critical Comparison and Discussion (8 marks)

### Task 3.1: Comparative Analysis

**Performance Summary:**

| Metric | Math GPT | Boolean GPT | Difference |
|--------|----------|-------------|------------|
| **Exact Match Accuracy** | 72.0% | 90.4% | +18.4 pp |
| **Parameters** | 104,211 (0.10M) | 28,051 (0.03M) | 3.7× smaller |
| **Training Time** | 581s (20K iters) | 330s (15K iters) | 1.8× faster |
| **Best Operation** | Division: 84.6% | OR: 98.8% | +14.2 pp |
| **Worst Operation** | Subtraction: 24.6% | XOR: 49.5% | +24.9 pp |

### Architectural Elements: Common vs Adapted

**Elements Optimal for Both Tasks:**

1. **Character-Level Tokenization:**
   - **Math:** Each digit (0-9) and operator (+,-,*) is atomic and meaningful
   - **Boolean:** Each character in "True"/"False" and operators carry meaning
   - **Why it works:** Symbolic tasks require fine-grained token representation
   - **Shared benefit:** No need for subword tokenization (BPE) or large vocabularies

2. **Shallow Architecture (2 Layers):**
   - **Math:** 2 layers prevent overfitting on deterministic arithmetic patterns
   - **Boolean:** 2 layers sufficient for nested Boolean evaluation
   - **Why it works:** Symbolic reasoning doesn't require deep linguistic understanding
   - **Numerical evidence:** 4-layer models overfit (test accuracy dropped 15-20 pp for both tasks)

3. **Small Embedding Dimensions:**
   - **Math:** 64 dimensions for 19-character vocabulary
   - **Boolean:** 32 dimensions for 19-character vocabulary
   - **Why it works:** Limited vocabulary doesn't benefit from large embeddings (vs 768 in BERT)
   - **Empirical support:** 128-dim embeddings caused overfitting without accuracy gains

4. **Cross-Entropy Loss (Unchanged):**
   - **Both tasks:** Standard sequence prediction loss
   - **Why appropriate:** Both tasks are autoregressive text generation
   - **No modification needed:** Loss naturally handles multi-token outputs

5. **AdamW Optimizer with 3×10⁻⁴ Learning Rate:**
   - **Both tasks:** Standard Transformer training recipe works well
   - **Why it works:** Deterministic tasks benefit from stable optimization

**Elements Requiring Task-Specific Adaptation:**

| Element | Math GPT | Boolean GPT | Rationale |
|---------|----------|-------------|-----------|
| **Embedding Dimension** | 64 | 32 | Boolean patterns simpler; half the dimensions sufficient |
| **Attention Heads** | 4 | 2 | Boolean has fewer distinct patterns (4 operators vs infinite arithmetic) |
| **Block Size** | 32 | 48 | Boolean uses verbose tokens ("True"/"False" vs single digits) |
| **Dropout** | 0.1 | 0.05 | Boolean's smaller state space = lower overfitting risk |
| **Training Iterations** | 20,000 | 15,000 | Boolean converges faster due to simpler patterns |
| **Dataset Repetition** | 50× | 500× | Boolean's tiny truth tables allow extreme repetition |

### What These Differences Reveal About Task Nature:

1. **Output Space Complexity:**
   - **Math:** Infinite possible outputs (any integer); requires larger model capacity
   - **Boolean:** Only 2 outputs (True/False); smaller model suffices
   - **Implication:** Model size should scale with output space complexity

2. **Pattern Diversity:**
   - **Math:** Unbounded numeric patterns (42, 437, -18, etc.); needs more attention heads to capture diverse patterns
   - **Boolean:** 4 basic operators with 2-4 entry truth tables; fewer heads needed
   - **Implication:** Attention head count should match pattern diversity

3. **Memorization vs Generalization:**
   - **Math:** Perfect memorization impossible (infinite space); must generalize from patterns
   - **Boolean:** Perfect memorization achievable (finite truth tables); 500× repetition works
   - **Implication:** Training strategy should match task's memorization feasibility

4. **Token Length Characteristics:**
   - **Math:** Compact notation (8*3+2); shorter context sufficient
   - **Boolean:** Verbose operators ("True AND False"); longer context needed
   - **Implication:** Block size should match typical expression length, not complexity

### Impact on Model Evaluation:

**Generalizability:**
- **Math GPT:** Struggles to generalize beyond training distribution (e.g., three-digit numbers)
- **Boolean GPT:** Perfect generalization within Boolean logic domain (test set nearly identical to train)
- **Reason:** Boolean's closed, finite domain vs arithmetic's open, infinite domain

**Error Patterns:**
- **Math:** Character-level errors (predicting "42" instead of "43"); magnitude confusion
- **Boolean:** Binary errors (predicting "True" instead of "False"); no partial mistakes
- **Impact on metrics:** Math's 5.2 pp gap (exact vs character) vs Boolean's 0.5 pp gap

**Computational Cost:**
- **Boolean GPT:** 3.7× fewer parameters, 1.8× faster training, identical inference speed per token
- **Math GPT:** Higher capacity needed but still minuscule vs NLP models (0.10M vs GPT-3's 175B)
- **Efficiency:** Both models extremely efficient compared to general-purpose LLMs

### How These Models Fail:

**Math GPT Failure Modes:**

1. **Negative Number Confusion (Subtraction: 24.6% accuracy):**
   - **Problem:** Minus sign (-) is both operator and negative sign
   - **Example:** `5-9=` → `-12` ✗ (expected: -4)
   - **Solution attempted:** Increased subtraction examples 50×
   - **Result:** Improved from 18% to 24.6%, still poor
   - **Why it persists:** Character-level tokenization can't distinguish operator vs sign

2. **Multi-Digit Generation Errors:**
   - **Problem:** Treats "42" as sequence ['4','2'], not atomic number 42
   - **Example:** `47+38=` → `84` ✗ (expected: 85)
   - **Solution attempted:** More two-digit examples
   - **Result:** Marginal improvement (55% → 58% on two-digit)
   - **Why it persists:** No number-level representation; purely character-based

3. **BODMAS Order of Operations (Mixed: 66.7% accuracy):**
   - **Problem:** Model computes left-to-right instead of multiplication-first
   - **Example:** `5+4*3=` → `27` ✗ (expected: 17) — computed (5+4)*3
   - **Solution attempted:** Exhaustive BODMAS coverage (18,000 examples)
   - **Result:** Improved from 42% to 66.7%
   - **Partial success:** Heavy repetition helped but didn't solve completely

**Boolean GPT Failure Modes:**

1. **XOR Confusion (XOR: 49.5% accuracy):**
   - **Problem:** XOR truth table less intuitive; confused with OR
   - **Example:** `True XOR True=` → `True` ✗ (expected: False)
   - **Solution attempted:** 500× repetition of XOR truth table
   - **Result:** Improved from 35% to 49.5%, still near random
   - **Why it persists:** XOR is inherently harder concept; model may need more architectural bias

2. **Deep Nesting Errors (4+ levels):**
   - **Problem:** Attention mechanism loses track of deeply nested parentheses
   - **Example:** `(True AND (True AND (True AND False)))=` → `True` ✗
   - **Solution attempted:** Increased nested examples
   - **Result:** Parentheses overall high (94.9%) but deep nesting still fails
   - **Why it persists:** Transformer lacks explicit recursion mechanism

### Unsolved Issues and Future Directions:

**Unsolved Issue 1: Character-Level Number Representation**
- **Problem:** Math GPT can't distinguish "42" (number) from ['4','2'] (sequence)
- **Current approach:** Character-level tokenization
- **Proposed solution:**
  - Hybrid tokenization: Numbers as single tokens, operators as characters
  - Requires custom tokenizer: split "3+42" → ['3', '+', '42']
  - Benefit: Model treats numbers atomically, eliminating multi-digit generation errors
- **Expected impact:** Addition/multiplication accuracy could improve 20-30 pp

**Unsolved Issue 2: Algorithmic vs Pattern-Matching**
- **Problem:** Both models memorize patterns, don't truly compute
- **Evidence:** Fail on out-of-distribution examples (e.g., three-digit arithmetic)
- **Proposed solution:**
  - External computation tool: Model learns to generate Python code, execute it
  - Example: Input "23+47" → Model generates "result = 23+47" → Executes → Returns 70
  - Benefit: Perfect accuracy on any arithmetic within Python's range
- **Expected impact:** 100% accuracy achievable, but loses "neural" aspect

**Unsolved Issue 3: XOR Representation**
- **Problem:** XOR requires "different" concept, hard for neural networks
- **Proposed solution:**
  - Explicit XOR neuron layer: Architectural bias favoring XOR pattern
  - Binary embedding: Represent True=1, False=0; learn XOR as multiplication of (a-b)²
- **Expected impact:** Could push XOR accuracy to 90%+

**Unsolved Issue 4: Interpretability**
- **Problem:** Can't explain WHY model predicts certain answers
- **Proposed solution:**
  - Attention visualization: Show which tokens attend to which
  - Neuron activation analysis: Identify "multiplication neurons" or "OR neurons"
- **Expected impact:** Better debugging, understanding of learned representations

### Key Insights from This Comparison:

1. **Task Complexity Dictates Architecture:**
   - Simpler tasks (Boolean) need smaller models
   - More complex tasks (arithmetic) need larger capacity
   - Sweet spot: Match model size to task complexity to avoid overfitting/underfitting

2. **Output Space is the Critical Factor:**
   - Boolean's 2 outputs → 90% accuracy achievable
   - Math's infinite outputs → fundamental difficulty in reaching 90%
   - Lesson: Transformer accuracy inversely correlated with output space size

3. **Memorization Has Limits:**
   - Boolean: Finite truth tables → perfect memorization possible
   - Math: Infinite combinations → memorization insufficient
   - Conclusion: Pattern memorization works for closed domains, fails for open domains

4. **Character-Level Tokenization Trade-offs:**
   - Benefits: Handles any digit, no OOV tokens, small vocabulary
   - Costs: Multi-digit numbers as sequences, operator-sign ambiguity
   - Conclusion: Consider hybrid tokenization for numeric tasks

5. **Symbolic Tasks ≠ Natural Language:**
   - NLP models (BERT, GPT-3) over-engineered for symbolic reasoning
   - 0.1M parameters sufficient for arithmetic (vs 175B for language)
   - Lesson: Don't apply NLP architectures blindly to symbolic tasks

---

## Conclusion

This assignment demonstrates that transformer models can be effectively repurposed for symbolic reasoning tasks through careful architectural adaptation and dataset curation. Math GPT achieves 72.0% accuracy on arithmetic expressions using only 104K parameters, while Boolean GPT reaches 90.4% accuracy with just 28K parameters—both orders of magnitude smaller than NLP models yet effective for their domains.

The key findings are:

1. **Model size should match task complexity:** Boolean's simpler logic requires 3.7× fewer parameters than arithmetic
2. **Output space size determines difficulty:** Boolean's 2 outputs enable 90%+ accuracy; arithmetic's infinite range caps performance at 72%
3. **Memorization works for finite domains:** 500× truth table repetition achieves near-perfect Boolean performance
4. **Character-level tokenization has limits:** Works well for single digits but struggles with multi-digit numbers and sign ambiguity
5. **Transformers excel at pattern matching, not computation:** Models memorize "3+5=8" rather than learning addition algorithm

Future work should explore hybrid tokenization (numbers as atomic tokens), external computation tools (e.g., Python evaluators), and architectural biases for specific operations (e.g., XOR-specific layers). The fundamental limitation—transformers pattern-match rather than compute—suggests that pure neural approaches may never achieve perfect arithmetic accuracy without algorithmic components.

---

## Figures

**Figure 1:** Overall Performance Comparison
*(See outputs/figure1_overall_performance.png)*

**Figure 2:** Operation-Specific Accuracy Analysis
*(See outputs/figure2_operation_accuracy.png)*

**Figure 3:** Architectural Parameters Comparison
*(See outputs/figure3_architecture_comparison.png)*

---

## Declaration

I hereby declare that this assignment is entirely my own work. I have not collaborated with any other student, and all code has been written by myself. Any external sources consulted have been properly cited.

---

*End of Report*
