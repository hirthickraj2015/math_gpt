"""
Comprehensive Math Dataset Generator
=====================================
Generates a massive, diverse dataset of arithmetic expressions using multiprocessing.

Features:
- All BODMAS operations (+, -, *, //, %)
- Variable expression lengths (2 to 6 operands)
- Negative numbers
- Parentheses at various depths
- Carry propagation (99+1=100)
- Zero handling
- Reversed expressions (answer=expression)
- Commutative swaps
- Edge cases

Usage:
    python generate_comprehensive_dataset.py
"""

import random
import os
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Set
from collections import Counter
import time

# Configuration
TOTAL_SAMPLES = 500_000  # Total expressions to generate
TRAIN_SPLIT = 0.9
OUTPUT_DIR = "dataset/math_v2"
WORKERS = cpu_count()

# Number ranges for different complexity levels
RANGES = {
    "tiny": (0, 9),
    "small": (1, 50),
    "medium": (1, 100),
    "large": (100, 999),
    "huge": (1000, 9999),
}

OPERATORS = ['+', '-', '*', '//', '%']
OPERATORS_NO_DIV = ['+', '-', '*']  # For expressions where we want to avoid division by zero issues


def safe_eval(expr: str) -> int | None:
    """Safely evaluate an expression and return integer result."""
    try:
        result = eval(expr)
        if isinstance(result, float):
            if result.is_integer():
                return int(result)
            return None  # Skip non-integer results
        if abs(result) > 999999:  # Skip astronomically large results
            return None
        return result
    except:
        return None


def format_number(n: int) -> str:
    """Format a number, wrapping negatives in parentheses."""
    return f"({n})" if n < 0 else str(n)


# =============================================================================
# GENERATOR FUNCTIONS - Each returns a list of "expression=result" strings
# =============================================================================

def generate_simple_two_operand(count: int) -> List[str]:
    """Basic: a op b = result"""
    expressions = []
    for _ in range(count):
        a = random.randint(*RANGES["medium"])
        b = random.randint(*RANGES["medium"])
        op = random.choice(OPERATORS)
        
        # Avoid division by zero
        if op in ['//', '%'] and b == 0:
            b = random.randint(1, 100)
        
        expr = f"{a}{op}{b}"
        result = safe_eval(expr)
        if result is not None:
            expressions.append(f"{expr}={result}")
    return expressions


def generate_tiny_numbers(count: int) -> List[str]:
    """Single digit operations for learning basics."""
    expressions = []
    for _ in range(count):
        a = random.randint(*RANGES["tiny"])
        b = random.randint(*RANGES["tiny"])
        op = random.choice(OPERATORS_NO_DIV)
        
        expr = f"{a}{op}{b}"
        result = safe_eval(expr)
        if result is not None:
            expressions.append(f"{expr}={result}")
    return expressions


def generate_large_numbers(count: int) -> List[str]:
    """Large number arithmetic."""
    expressions = []
    for _ in range(count):
        a = random.randint(*RANGES["large"])
        b = random.randint(*RANGES["small"])
        op = random.choice(OPERATORS)
        
        if op in ['//', '%'] and b == 0:
            b = random.randint(1, 50)
        
        expr = f"{a}{op}{b}"
        result = safe_eval(expr)
        if result is not None:
            expressions.append(f"{expr}={result}")
    return expressions


def generate_huge_numbers(count: int) -> List[str]:
    """Very large number arithmetic (4 digits)."""
    expressions = []
    for _ in range(count):
        a = random.randint(*RANGES["huge"])
        b = random.randint(*RANGES["small"])
        op = random.choice(['+', '-'])  # Only add/sub for huge numbers
        
        expr = f"{a}{op}{b}"
        result = safe_eval(expr)
        if result is not None:
            expressions.append(f"{expr}={result}")
    return expressions


def generate_negative_inputs(count: int) -> List[str]:
    """Expressions with negative numbers as inputs."""
    expressions = []
    for _ in range(count):
        a = random.randint(-50, 50)
        b = random.randint(-50, 50)
        op = random.choice(OPERATORS_NO_DIV)
        
        expr = f"{format_number(a)}{op}{format_number(b)}"
        result = safe_eval(expr)
        if result is not None:
            expressions.append(f"{expr}={result}")
    return expressions


def generate_three_operand(count: int) -> List[str]:
    """a op1 b op2 c = result (tests BODMAS)"""
    expressions = []
    for _ in range(count):
        a = random.randint(*RANGES["small"])
        b = random.randint(*RANGES["small"])
        c = random.randint(*RANGES["small"])
        op1 = random.choice(OPERATORS)
        op2 = random.choice(OPERATORS)
        
        # Avoid division by zero
        if op1 in ['//', '%'] and b == 0:
            b = random.randint(1, 50)
        if op2 in ['//', '%'] and c == 0:
            c = random.randint(1, 50)
        
        expr = f"{a}{op1}{b}{op2}{c}"
        result = safe_eval(expr)
        if result is not None:
            expressions.append(f"{expr}={result}")
    return expressions


def generate_four_operand(count: int) -> List[str]:
    """a op1 b op2 c op3 d = result"""
    expressions = []
    for _ in range(count):
        nums = [random.randint(*RANGES["tiny"]) for _ in range(4)]
        ops = [random.choice(OPERATORS_NO_DIV) for _ in range(3)]
        
        expr = f"{nums[0]}{ops[0]}{nums[1]}{ops[1]}{nums[2]}{ops[2]}{nums[3]}"
        result = safe_eval(expr)
        if result is not None:
            expressions.append(f"{expr}={result}")
    return expressions


def generate_five_operand(count: int) -> List[str]:
    """a op1 b op2 c op3 d op4 e = result"""
    expressions = []
    for _ in range(count):
        nums = [random.randint(*RANGES["tiny"]) for _ in range(5)]
        ops = [random.choice(['+', '-']) for _ in range(4)]  # Keep it simple with long chains
        
        expr = "".join(f"{nums[i]}{ops[i]}" for i in range(4)) + str(nums[4])
        result = safe_eval(expr)
        if result is not None:
            expressions.append(f"{expr}={result}")
    return expressions


def generate_six_operand(count: int) -> List[str]:
    """Very long chains: a+b+c+d+e+f"""
    expressions = []
    for _ in range(count):
        nums = [random.randint(*RANGES["tiny"]) for _ in range(6)]
        ops = [random.choice(['+', '-']) for _ in range(5)]
        
        expr = "".join(f"{nums[i]}{ops[i]}" for i in range(5)) + str(nums[5])
        result = safe_eval(expr)
        if result is not None:
            expressions.append(f"{expr}={result}")
    return expressions


def generate_simple_parentheses(count: int) -> List[str]:
    """(a op1 b) op2 c = result"""
    expressions = []
    for _ in range(count):
        a = random.randint(*RANGES["small"])
        b = random.randint(*RANGES["small"])
        c = random.randint(*RANGES["small"])
        op1 = random.choice(OPERATORS_NO_DIV)
        op2 = random.choice(OPERATORS)
        
        if op2 in ['//', '%'] and c == 0:
            c = random.randint(1, 30)
        
        # Randomly choose left or right parentheses
        if random.random() < 0.5:
            expr = f"({a}{op1}{b}){op2}{c}"
        else:
            expr = f"{a}{op1}({b}{op2}{c})"
        
        result = safe_eval(expr)
        if result is not None:
            expressions.append(f"{expr}={result}")
    return expressions


def generate_nested_parentheses(count: int) -> List[str]:
    """((a op1 b) op2 c) op3 d = result"""
    expressions = []
    for _ in range(count):
        a, b, c, d = [random.randint(*RANGES["tiny"]) for _ in range(4)]
        op1, op2, op3 = [random.choice(OPERATORS_NO_DIV) for _ in range(3)]
        
        patterns = [
            f"(({a}{op1}{b}){op2}{c}){op3}{d}",
            f"{a}{op1}(({b}{op2}{c}){op3}{d})",
            f"{a}{op1}({b}{op2}({c}{op3}{d}))",
            f"({a}{op1}{b}){op2}({c}{op3}{d})",
        ]
        expr = random.choice(patterns)
        result = safe_eval(expr)
        if result is not None:
            expressions.append(f"{expr}={result}")
    return expressions


def generate_deep_nesting(count: int) -> List[str]:
    """(((a op b) op c) op d) op e"""
    expressions = []
    for _ in range(count):
        nums = [random.randint(1, 10) for _ in range(5)]
        ops = [random.choice(['+', '-', '*']) for _ in range(4)]
        
        # Build deeply nested expression
        expr = str(nums[0])
        for i in range(4):
            expr = f"({expr}{ops[i]}{nums[i+1]})"
        
        result = safe_eval(expr)
        if result is not None:
            expressions.append(f"{expr}={result}")
    return expressions


def generate_bodmas_critical(count: int) -> List[str]:
    """Expressions where order of operations is critical."""
    expressions = []
    ops_high = ['*', '//']
    ops_low = ['+', '-']
    
    for _ in range(count):
        a, b, c, d = [random.randint(2, 20) for _ in range(4)]
        
        patterns = [
            f"{a}+{b}*{c}",           # Should be a + (b*c)
            f"{a}*{b}+{c}*{d}",       # (a*b) + (c*d)
            f"{a}+{b}*{c}-{d}",       # a + (b*c) - d
            f"{a}*{b}+{c}",           # (a*b) + c
            f"{a}-{b}//{c}",          # a - (b//c)
            f"{a}+{b}%{c}",           # a + (b%c)
        ]
        expr = random.choice(patterns)
        result = safe_eval(expr)
        if result is not None:
            expressions.append(f"{expr}={result}")
    return expressions


def generate_carry_propagation(count: int) -> List[str]:
    """Expressions that test carry: 99+1, 999+1, 49+51."""
    expressions = []
    carry_bases = [9, 19, 29, 49, 99, 199, 499, 999]
    
    for _ in range(count):
        base = random.choice(carry_bases)
        offset = random.randint(1, 20)
        
        if random.random() < 0.5:
            expr = f"{base}+{offset}"
        else:
            expr = f"{base + offset}-{offset}"
        
        result = safe_eval(expr)
        if result is not None:
            expressions.append(f"{expr}={result}")
    return expressions


def generate_zero_expressions(count: int) -> List[str]:
    """Expressions involving zero."""
    expressions = []
    
    for _ in range(count):
        a = random.randint(1, 100)
        
        patterns = [
            f"0+{a}",
            f"{a}+0",
            f"{a}-0",
            f"{a}*0",
            f"0*{a}",
            f"0//{a}",
            f"0%{a}",
            f"{a}-{a}",
        ]
        expr = random.choice(patterns)
        result = safe_eval(expr)
        if result is not None:
            expressions.append(f"{expr}={result}")
    return expressions


def generate_identity_expressions(count: int) -> List[str]:
    """Expressions that result in the same number: a+0, a*1, a-0."""
    expressions = []
    
    for _ in range(count):
        a = random.randint(1, 1000)
        
        patterns = [
            f"{a}+0",
            f"{a}-0",
            f"{a}*1",
            f"{a}//1",
        ]
        expr = random.choice(patterns)
        result = safe_eval(expr)
        if result is not None:
            expressions.append(f"{expr}={result}")
    return expressions


def generate_reversed_expressions(count: int) -> List[str]:
    """result=expression format."""
    expressions = []
    
    for _ in range(count):
        a = random.randint(*RANGES["small"])
        b = random.randint(*RANGES["small"])
        op = random.choice(OPERATORS_NO_DIV)
        
        expr = f"{a}{op}{b}"
        result = safe_eval(expr)
        if result is not None:
            # Reversed format: result = expression
            expressions.append(f"{result}={a}{op}{b}")
    return expressions


def generate_commutative_pairs(count: int) -> List[str]:
    """Generate both a+b and b+a, a*b and b*a."""
    expressions = []
    
    for _ in range(count // 2):
        a = random.randint(*RANGES["medium"])
        b = random.randint(*RANGES["medium"])
        op = random.choice(['+', '*'])  # Commutative operators
        
        result = safe_eval(f"{a}{op}{b}")
        if result is not None:
            expressions.append(f"{a}{op}{b}={result}")
            expressions.append(f"{b}{op}{a}={result}")
    return expressions


def generate_squares_and_powers(count: int) -> List[str]:
    """a*a (squares) and simple powers."""
    expressions = []
    
    for _ in range(count):
        a = random.randint(1, 30)
        
        if random.random() < 0.5:
            expr = f"{a}*{a}"  # Square
        else:
            b = random.randint(2, 5)
            expr = "*".join([str(a)] * b)  # a*a*a...
        
        result = safe_eval(expr)
        if result is not None and abs(result) < 100000:
            expressions.append(f"{expr}={result}")
    return expressions


def generate_division_remainder_pairs(count: int) -> List[str]:
    """a = (a//b)*b + (a%b) style verification."""
    expressions = []
    
    for _ in range(count):
        a = random.randint(10, 200)
        b = random.randint(2, 20)
        
        # Standard division and modulo
        patterns = [
            f"{a}//{b}",
            f"{a}%{b}",
        ]
        expr = random.choice(patterns)
        result = safe_eval(expr)
        if result is not None:
            expressions.append(f"{expr}={result}")
    return expressions


def generate_subtraction_to_negative(count: int) -> List[str]:
    """Cases where subtraction results in negative numbers."""
    expressions = []
    
    for _ in range(count):
        small = random.randint(1, 50)
        large = random.randint(51, 150)
        
        expr = f"{small}-{large}"
        result = safe_eval(expr)
        if result is not None:
            expressions.append(f"{expr}={result}")
    return expressions


def generate_all_operators_chain(count: int) -> List[str]:
    """Use all operators in one expression."""
    expressions = []
    
    for _ in range(count):
        nums = [random.randint(2, 15) for _ in range(5)]
        
        # Ensure no division by zero
        if nums[2] == 0:
            nums[2] = 1
        if nums[4] == 0:
            nums[4] = 1
        
        expr = f"{nums[0]}+{nums[1]}*{nums[2]}-{nums[3]}//{nums[4]}"
        result = safe_eval(expr)
        if result is not None:
            expressions.append(f"{expr}={result}")
    return expressions


def generate_mixed_complexity(count: int) -> List[str]:
    """Randomly pick from various generators for diversity."""
    generators = [
        generate_simple_two_operand,
        generate_three_operand,
        generate_simple_parentheses,
        generate_bodmas_critical,
    ]
    
    expressions = []
    per_generator = count // len(generators)
    
    for gen in generators:
        expressions.extend(gen(per_generator))
    
    return expressions


# =============================================================================
# MULTIPROCESSING WORKER
# =============================================================================

def worker_generate(args: Tuple[str, int]) -> List[str]:
    """Worker function for multiprocessing."""
    generator_name, count = args
    
    generators = {
        "simple_two_operand": generate_simple_two_operand,
        "tiny_numbers": generate_tiny_numbers,
        "large_numbers": generate_large_numbers,
        "huge_numbers": generate_huge_numbers,
        "negative_inputs": generate_negative_inputs,
        "three_operand": generate_three_operand,
        "four_operand": generate_four_operand,
        "five_operand": generate_five_operand,
        "six_operand": generate_six_operand,
        "simple_parentheses": generate_simple_parentheses,
        "nested_parentheses": generate_nested_parentheses,
        "deep_nesting": generate_deep_nesting,
        "bodmas_critical": generate_bodmas_critical,
        "carry_propagation": generate_carry_propagation,
        "zero_expressions": generate_zero_expressions,
        "identity_expressions": generate_identity_expressions,
        "reversed_expressions": generate_reversed_expressions,
        "commutative_pairs": generate_commutative_pairs,
        "squares_and_powers": generate_squares_and_powers,
        "division_remainder": generate_division_remainder_pairs,
        "subtraction_negative": generate_subtraction_to_negative,
        "all_operators_chain": generate_all_operators_chain,
        "mixed_complexity": generate_mixed_complexity,
    }
    
    return generators[generator_name](count)


# =============================================================================
# MAIN GENERATION PIPELINE
# =============================================================================

def generate_dataset():
    """Main function to generate the complete dataset."""
    print("=" * 70)
    print("COMPREHENSIVE MATH DATASET GENERATOR")
    print("=" * 70)
    print(f"Target samples: {TOTAL_SAMPLES:,}")
    print(f"Workers: {WORKERS}")
    print(f"Output: {OUTPUT_DIR}/")
    print("=" * 70)
    
    # Distribution of expression types (adjust weights as needed)
    distribution = {
        "simple_two_operand": 0.12,
        "tiny_numbers": 0.08,
        "large_numbers": 0.06,
        "huge_numbers": 0.03,
        "negative_inputs": 0.06,
        "three_operand": 0.10,
        "four_operand": 0.06,
        "five_operand": 0.04,
        "six_operand": 0.03,
        "simple_parentheses": 0.10,
        "nested_parentheses": 0.06,
        "deep_nesting": 0.03,
        "bodmas_critical": 0.08,
        "carry_propagation": 0.04,
        "zero_expressions": 0.02,
        "identity_expressions": 0.01,
        "reversed_expressions": 0.02,
        "commutative_pairs": 0.02,
        "squares_and_powers": 0.02,
        "division_remainder": 0.02,
        "subtraction_negative": 0.02,
        "all_operators_chain": 0.02,
        "mixed_complexity": 0.04,
    }
    
    # Verify distribution sums to ~1.0
    total_weight = sum(distribution.values())
    print(f"Distribution weight sum: {total_weight:.2f}")
    
    # Create tasks for multiprocessing
    tasks = []
    for gen_name, weight in distribution.items():
        count = int(TOTAL_SAMPLES * weight / WORKERS)
        for _ in range(WORKERS):
            tasks.append((gen_name, count))
    
    print(f"\nGenerating with {len(tasks)} parallel tasks...")
    start_time = time.time()
    
    # Run multiprocessing
    all_expressions = []
    with Pool(WORKERS) as pool:
        results = pool.map(worker_generate, tasks)
        for result in results:
            all_expressions.extend(result)
    
    print(f"Generated {len(all_expressions):,} raw expressions in {time.time() - start_time:.1f}s")
    
    # Remove duplicates
    print("Removing duplicates...")
    unique_expressions = list(set(all_expressions))
    print(f"Unique expressions: {len(unique_expressions):,}")
    
    # Shuffle
    print("Shuffling...")
    random.shuffle(unique_expressions)
    
    # Split into train/test
    split_idx = int(len(unique_expressions) * TRAIN_SPLIT)
    train_data = unique_expressions[:split_idx]
    test_data = unique_expressions[split_idx:]
    
    print(f"Train set: {len(train_data):,}")
    print(f"Test set: {len(test_data):,}")
    
    # Save to files
    os.makedirs(f"{OUTPUT_DIR}/training", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/testing", exist_ok=True)
    
    with open(f"{OUTPUT_DIR}/training/math_train.txt", 'w') as f:
        f.write('\n'.join(train_data))
    
    with open(f"{OUTPUT_DIR}/testing/math_test.txt", 'w') as f:
        f.write('\n'.join(test_data))
    
    print(f"\nSaved to:")
    print(f"  {OUTPUT_DIR}/training/math_train.txt")
    print(f"  {OUTPUT_DIR}/testing/math_test.txt")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    # Analyze expression lengths
    lengths = [len(expr.split('=')[0]) for expr in unique_expressions[:10000]]
    print(f"Expression length (before '='):")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {sum(lengths)/len(lengths):.1f}")
    
    # Operator distribution
    op_counts = Counter()
    for expr in unique_expressions[:50000]:
        for op in OPERATORS:
            op_counts[op] += expr.count(op)
    
    print(f"\nOperator distribution (sample of 50k):")
    for op, count in op_counts.most_common():
        print(f"  {op}: {count:,}")
    
    # Sample expressions
    print(f"\nSample expressions:")
    for expr in random.sample(unique_expressions, 15):
        print(f"  {expr}")
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    generate_dataset()
