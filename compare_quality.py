#!/usr/bin/env python3
"""
Compare quality metrics across multiple episodes and reference examples.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
from quality_analyzer import QualityAnalyzer


def analyze_all_episodes(file_paths: List[str]) -> List[Dict]:
    """Analyze all provided files"""
    results = []

    for filepath in file_paths:
        path = Path(filepath)
        if not path.exists():
            print(f"⚠️  File not found: {filepath}")
            continue

        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        analyzer = QualityAnalyzer(text, path.stem)
        result = analyzer.analyze()
        result['filepath'] = str(path)
        results.append(result)

    return results


def print_comparison_table(results: List[Dict]):
    """Print comparison table"""
    if not results:
        print("No results to compare")
        return

    print("\n" + "="*120)
    print("  QUALITY COMPARISON TABLE")
    print("="*120)

    # Header
    episodes = [r['episode'] for r in results]
    header = f"{'Metric':<25} | " + " | ".join(f"{ep[:15]:^15}" for ep in episodes)
    print(header)
    print("-"*120)

    # Metrics to display
    metrics = [
        ('overall_score', 'Overall Score', 'overall_score'),
        ('sentence_complexity', 'Sentence Avg', 'avg_length'),
        ('paragraph_density', 'Para Density', 'avg_sentences'),
        ('repetition_patterns', 'Repetitions', 'violations'),
        ('abstract_concrete_ratio', 'Abstract:Concrete', 'ratio'),
        ('dialogue_ratio', 'Dialogue %', 'percentage'),
        ('scene_progression', 'Scene/50 lines', 'rate_per_50'),
        ('information_novelty', 'Info Novelty %', 'novelty_percentage'),
        ('subordinate_clause_depth', 'Deep Nesting %', 'percentage'),
        ('opening_line_quality', 'Opening Words', 'word_count')
    ]

    for metric_key, label, detail_key in metrics:
        row = f"{label:<25} | "

        if metric_key == 'overall_score':
            values = [r['overall_score'] for r in results]
            row += " | ".join(f"{v:^15.3f}" for v in values)
        elif metric_key in results[0]:
            values = []
            for r in results:
                metric = r[metric_key]
                value = metric.get(detail_key, '')

                # Format value
                if isinstance(value, float):
                    value_str = f"{value:.2f}"
                elif isinstance(value, int):
                    value_str = f"{value}"
                else:
                    value_str = str(value)

                # Add pass/fail indicator
                if metric.get('pass'):
                    value_str = f"✓ {value_str}"
                else:
                    value_str = f"✗ {value_str}"

                values.append(value_str)

            row += " | ".join(f"{v:^15}" for v in values)
        else:
            row += " | ".join(f"{'N/A':^15}" for _ in results)

        print(row)

    print("="*120)

    # Summary
    print("\n📊 SUMMARY:")
    for r in results:
        score = r['overall_score']
        grade = 'A' if score >= 0.9 else 'B' if score >= 0.8 else 'C' if score >= 0.7 else 'D' if score >= 0.6 else 'F'
        status = '✅' if score >= 0.8 else '⚠️' if score >= 0.7 else '❌'
        print(f"{status} {r['episode'][:30]:<30} | Score: {score:.3f} ({grade})")

    print()


def print_improvement_recommendations(results: List[Dict]):
    """Print improvement recommendations"""
    print("\n" + "="*70)
    print("  IMPROVEMENT RECOMMENDATIONS")
    print("="*70)

    for r in results:
        score = r['overall_score']
        if score >= 0.8:
            continue  # Skip episodes that meet quality target

        print(f"\n📋 {r['episode']}")
        print(f"   Current Score: {score:.3f} | Target: 0.800")
        print("   Issues to fix:")

        # Identify failing metrics
        issues = []

        sent_comp = r['sentence_complexity']
        if not sent_comp['pass']:
            avg_len = sent_comp['avg_length']
            if avg_len < 12:
                issues.append(
                    f"   • Sentences too short ({avg_len:.1f} words) - "
                    "Combine actions with -고, -며"
                )
            elif avg_len > 20:
                issues.append(
                    f"   • Sentences too long ({avg_len:.1f} words) - "
                    "Break into shorter units"
                )

        dialogue = r['dialogue_ratio']
        if not dialogue['pass']:
            pct = dialogue['percentage']
            if pct < 15:
                issues.append(
                    f"   • Not enough dialogue ({pct:.1f}%) - "
                    "Add 3-5 character exchanges"
                )
            elif pct > 30:
                issues.append(
                    f"   • Too much dialogue ({pct:.1f}%) - "
                    "Balance with action"
                )

        para = r['paragraph_density']
        if not para['pass']:
            avg_sents = para['avg_sentences']
            if avg_sents < 2:
                issues.append(
                    f"   • Paragraphs too short ({avg_sents:.1f} sents) - "
                    "Combine related actions"
                )
            elif avg_sents > 4:
                issues.append(
                    f"   • Paragraphs too long ({avg_sents:.1f} sents) - "
                    "Split into 2-3 sentence units"
                )

        abstract = r['abstract_concrete_ratio']
        if not abstract['pass']:
            ratio = abstract['ratio']
            if ratio != 'inf':
                issues.append(
                    f"   • Too abstract (ratio={ratio:.2f}) - "
                    "Use more concrete details (3:1 target)"
                )

        repetition = r['repetition_patterns']
        if not repetition['pass']:
            violations = repetition['violations']
            issues.append(
                f"   • {violations} repetition patterns - "
                "Vary sentence structure"
            )

        scene_prog = r['scene_progression']
        if not scene_prog['pass']:
            rate = scene_prog['rate_per_50']
            if rate > 3:
                issues.append(
                    f"   • Too many transitions ({rate:.1f} per 50 lines) - "
                    "Stay in scenes longer"
                )

        if issues:
            for issue in issues:
                print(issue)
        else:
            print("   ✅ No major issues")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Compare quality across episodes')
    parser.add_argument('files', nargs='+', help='Chapter files to compare')
    parser.add_argument('--json', action='store_true', help='Output JSON')
    parser.add_argument('--output', help='Output JSON file')

    args = parser.parse_args()

    # Analyze all files
    results = analyze_all_episodes(args.files)

    if args.json:
        output_data = {
            'episodes': results,
            'summary': {
                'total': len(results),
                'passed': sum(1 for r in results if r['overall_score'] >= 0.8),
                'avg_score': sum(r['overall_score'] for r in results) / len(results) if results else 0
            }
        }

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Saved to {args.output}")
        else:
            print(json.dumps(output_data, ensure_ascii=False, indent=2))
    else:
        print_comparison_table(results)
        print_improvement_recommendations(results)


if __name__ == '__main__':
    main()
