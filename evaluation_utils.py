"""
Evaluation utilities for Math and Boolean GPT models.
CS7CS4 Machine Learning - Final Assignment 2025-26

This module provides comprehensive evaluation metrics and analysis tools
for transformer-based symbolic reasoning models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import List, Tuple, Dict, Callable


def evaluate_exact_match(
    model,
    test_expressions: List[str],
    encode: Callable,
    decode: Callable,
    device: str = 'cpu',
    max_samples: int = 1000,
    temperature: float = 0.8
) -> Tuple[float, List[Tuple[str, str, str, bool]]]:
    """
    Evaluate exact match accuracy on test expressions.

    Args:
        model: trained GPT model
        test_expressions: list of test expressions (format: "input=output")
        encode: function to encode text to integers
        decode: function to decode integers to text
        device: device to run evaluation on
        max_samples: maximum number of samples to evaluate
        temperature: sampling temperature for generation

    Returns:
        accuracy: exact match accuracy (0-100)
        results: list of (input, expected, predicted, correct) tuples
    """
    model.eval()
    results = []
    correct = 0
    total = min(len(test_expressions), max_samples)

    with torch.no_grad():
        for i in range(total):
            expr = test_expressions[i].strip()
            if '=' not in expr:
                continue

            # Split into input and expected output
            parts = expr.split('=')
            input_part = parts[0] + '='
            expected = parts[1] if len(parts) > 1 else ""

            try:
                # Generate prediction
                context = torch.tensor(encode(input_part), dtype=torch.long, device=device).unsqueeze(0)
                generated = model.generate(context, max_new_tokens=len(expected)+10, temperature=temperature)
                prediction = decode(generated[0].tolist())

                # Extract predicted answer
                if '=' in prediction:
                    predicted = prediction.split('=')[1].split('\n')[0].strip()
                else:
                    predicted = ""

                is_correct = predicted == expected
                if is_correct:
                    correct += 1

                results.append((input_part, expected, predicted, is_correct))
            except Exception as e:
                print(f"Error evaluating expression '{expr}': {e}")
                results.append((input_part, expected, "", False))

    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy, results


def categorize_math_operation(expr: str) -> str:
    """
    Categorize a mathematical expression by its operation type.

    Args:
        expr: mathematical expression string

    Returns:
        operation category
    """
    if '(' in expr:
        return 'parentheses'
    elif '*' in expr and ('+' in expr or '-' in expr):
        return 'mixed_ops'
    elif '//' in expr:
        return 'division'
    elif '%' in expr:
        return 'modulo'
    elif '*' in expr:
        return 'multiplication'
    elif '+' in expr:
        return 'addition'
    elif '-' in expr:
        return 'subtraction'
    else:
        return 'other'


def categorize_boolean_operation(expr: str) -> str:
    """
    Categorize a boolean expression by its operation type.

    Args:
        expr: boolean expression string

    Returns:
        operation category
    """
    expr_upper = expr.upper()
    if '(' in expr:
        return 'parentheses'
    elif 'NOT' in expr_upper and ('AND' in expr_upper or 'OR' in expr_upper or 'XOR' in expr_upper):
        return 'not_combined'
    elif 'XOR' in expr_upper:
        return 'xor'
    elif 'AND' in expr_upper:
        return 'and'
    elif 'OR' in expr_upper:
        return 'or'
    elif 'NOT' in expr_upper:
        return 'not'
    else:
        return 'other'


def analyze_by_operation(
    results: List[Tuple[str, str, str, bool]],
    categorize_func: Callable
) -> Dict[str, Dict[str, int]]:
    """
    Analyze accuracy by operation type.

    Args:
        results: list of (input, expected, predicted, correct) tuples
        categorize_func: function to categorize operations

    Returns:
        Dictionary mapping operation types to statistics
    """
    operation_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for inp, exp, pred, corr in results:
        op_type = categorize_func(inp)
        operation_stats[op_type]['total'] += 1
        if corr:
            operation_stats[op_type]['correct'] += 1

    return dict(operation_stats)


def calculate_digit_accuracy(results: List[Tuple[str, str, str, bool]]) -> float:
    """
    Calculate character-level accuracy for predicted vs expected results.

    Args:
        results: list of (input, expected, predicted, correct) tuples

    Returns:
        digit_accuracy: percentage of correctly predicted characters
    """
    total_chars = 0
    correct_chars = 0

    for inp, expected, predicted, _ in results:
        max_len = max(len(expected), len(predicted))
        for i in range(max_len):
            total_chars += 1
            if i < len(expected) and i < len(predicted) and expected[i] == predicted[i]:
                correct_chars += 1

    accuracy = (correct_chars / total_chars) * 100 if total_chars > 0 else 0
    return accuracy


def plot_operation_accuracy(
    operation_stats: Dict[str, Dict[str, int]],
    title: str = 'Model Accuracy by Operation Type',
    save_path: str = None
):
    """
    Plot accuracy by operation type as a bar chart.

    Args:
        operation_stats: operation statistics dictionary
        title: plot title
        save_path: path to save figure (optional)
    """
    # Calculate accuracies
    ops = []
    accs = []
    for op_type, stats in sorted(operation_stats.items()):
        if stats['total'] > 0:
            ops.append(op_type)
            accs.append((stats['correct'] / stats['total']) * 100)

    # Create plot
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(ops)))

    plt.bar(ops, accs, color=colors, edgecolor='black', linewidth=1.5)
    plt.xlabel('Operation Type', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (op, acc) in enumerate(zip(ops, accs)):
        plt.text(i, acc + 2, f'{acc:.1f}%', ha='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved as '{save_path}'")

    plt.show()


def plot_training_curves(
    iterations: List[int],
    train_losses: List[float],
    test_losses: List[float],
    save_path: str = None
):
    """
    Plot training and test loss curves.

    Args:
        iterations: list of iteration numbers
        train_losses: list of training losses
        test_losses: list of test losses
        save_path: path to save figure (optional)
    """
    plt.figure(figsize=(12, 5))

    # Regular scale
    plt.subplot(1, 2, 1)
    plt.plot(iterations, train_losses, label='Train Loss', linewidth=2)
    plt.plot(iterations, test_losses, label='Test Loss', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Test Loss Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Log scale
    plt.subplot(1, 2, 2)
    plt.plot(iterations, train_losses, label='Train Loss', linewidth=2)
    plt.plot(iterations, test_losses, label='Test Loss', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title('Training and Test Loss (Log Scale)', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved as '{save_path}'")

    plt.show()


def plot_expression_length_distribution(
    expressions: List[str],
    block_size: int,
    title: str = 'Distribution of Expression Lengths',
    save_path: str = None
):
    """
    Plot distribution of expression lengths.

    Args:
        expressions: list of expressions
        block_size: model block size
        title: plot title
        save_path: path to save figure (optional)
    """
    lengths = [len(expr.strip()) for expr in expressions if expr.strip()]

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(block_size, color='red', linestyle='--', linewidth=2,
                label=f'Block size ({block_size})')
    plt.xlabel('Expression Length (characters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved as '{save_path}'")

    plt.show()

    # Print statistics
    print(f"\\nExpression length statistics:") 
    print(f"  Mean: {np.mean(lengths):.1f} characters")
    print(f"  Median: {np.median(lengths):.1f} characters")
    print(f"  Max: {np.max(lengths)} characters")
    print(f"  95th percentile: {np.percentile(lengths, 95):.1f} characters")
    coverage = (np.array(lengths) <= block_size).mean() * 100
    print(f"  Block size of {block_size} covers {coverage:.1f}% of expressions")


def analyze_error_patterns(results: List[Tuple[str, str, str, bool]]) -> Dict[str, int]:
    """
    Analyze error patterns in predictions.

    Args:
        results: list of (input, expected, predicted, correct) tuples

    Returns:
        Dictionary of error pattern counts
    """
    error_patterns = defaultdict(int)

    for inp, exp, pred, corr in results:
        if not corr:
            if pred == "":
                error_patterns['empty_prediction'] += 1
            elif len(pred) != len(exp):
                error_patterns['wrong_length'] += 1
            else:
                error_patterns['wrong_value'] += 1

    return dict(error_patterns)


def print_evaluation_summary(
    accuracy: float,
    digit_accuracy: float,
    operation_stats: Dict[str, Dict[str, int]],
    results: List[Tuple[str, str, str, bool]],
    model_params: int
):
    """
    Print comprehensive evaluation summary.

    Args:
        accuracy: exact match accuracy
        digit_accuracy: character-level accuracy
        operation_stats: operation statistics
        results: evaluation results
        model_params: number of model parameters
    """
    print("\\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)

    print(f"\\nOverall Metrics:")
    print(f"  Exact Match Accuracy: {accuracy:.2f}%")
    print(f"  Digit-Level Accuracy: {digit_accuracy:.2f}%")
    print(f"  Total Samples Evaluated: {len(results)}")
    print(f"  Model Parameters: {model_params/1e6:.2f}M")

    print(f"\\nAccuracy by Operation Type:")
    print("-" * 70)
    print(f"{'Operation':<20} {'Correct':<10} {'Total':<10} {'Accuracy'}")
    print("-" * 70)

    for op_type in sorted(operation_stats.keys()):
        stats = operation_stats[op_type]
        acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"{op_type:<20} {stats['correct']:<10} {stats['total']:<10} {acc:.2f}%")

    # Error analysis
    incorrect = [r for r in results if not r[3]]
    if incorrect:
        error_patterns = analyze_error_patterns(results)
        print(f"\\nError Analysis:")
        print(f"  Total Errors: {len(incorrect)}")
        for pattern, count in error_patterns.items():
            pct = (count / len(incorrect)) * 100
            print(f"  {pattern}: {count} ({pct:.1f}%)")


def save_results_to_file(
    results: List[Tuple[str, str, str, bool]],
    filepath: str,
    num_correct: int = 20,
    num_incorrect: int = 20
):
    """
    Save sample results to a text file for report appendix.

    Args:
        results: evaluation results
        filepath: path to save file
        num_correct: number of correct examples to save
        num_incorrect: number of incorrect examples to save
    """
    correct_examples = [r for r in results if r[3]]
    incorrect_examples = [r for r in results if not r[3]]

    with open(filepath, 'w') as f:
        f.write("="*70 + "\\n")
        f.write("SAMPLE PROMPT-OUTPUT PAIRS\\n")
        f.write("="*70 + "\\n\\n")

        f.write("Demonstrating Strengths (Correct Predictions):\\n")
        f.write("-" * 70 + "\\n")
        for i in range(min(num_correct, len(correct_examples))):
            inp, exp, pred, _ = correct_examples[i]
            f.write(f"Input:    {inp}\\n")
            f.write(f"Expected: {exp}\\n")
            f.write(f"Output:   {pred} ✓\\n")
            f.write("\\n")

        f.write("\\nDemonstrating Weaknesses (Incorrect Predictions):\\n")
        f.write("-" * 70 + "\\n")
        for i in range(min(num_incorrect, len(incorrect_examples))):
            inp, exp, pred, _ = incorrect_examples[i]
            f.write(f"Input:    {inp}\\n")
            f.write(f"Expected: {exp}\\n")
            f.write(f"Output:   {pred} ✗\\n")
            f.write("\\n")

    print(f"Results saved to '{filepath}'")
