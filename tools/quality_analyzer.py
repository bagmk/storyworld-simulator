#!/usr/bin/env python3
"""
Quality Analyzer for Generated Chapters
Evaluates chapters against comprehensive quality metrics
"""

import re
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import argparse


class QualityAnalyzer:
    def __init__(self, text: str, episode_name: str = ""):
        self.text = text.strip()
        self.episode_name = episode_name
        self.sentences = self._split_sentences()
        self.paragraphs = self._split_paragraphs()

    def _split_sentences(self) -> List[str]:
        """Split text into sentences"""
        # Korean sentence endings: . ! ? " 등
        sentences = re.split(r'[.!?]\s+|[.!?]$|[.!?]"', self.text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]

    def _split_paragraphs(self) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = [p.strip() for p in self.text.split('\n\n') if p.strip()]
        # Filter out markdown headers and metadata
        paragraphs = [p for p in paragraphs if not p.startswith('#')
                      and not p.startswith('*')
                      and not p.startswith('---')
                      and len(p) > 20]
        return paragraphs

    def analyze(self) -> Dict:
        """Run all quality checks"""
        results = {
            'episode': self.episode_name,
            'basic_stats': self._basic_stats(),
            'sentence_complexity': self._sentence_complexity(),
            'paragraph_density': self._paragraph_density(),
            'repetition_patterns': self._repetition_patterns(),
            'abstract_concrete_ratio': self._abstract_concrete_ratio(),
            'dialogue_ratio': self._dialogue_ratio(),
            'scene_progression': self._scene_progression(),
            'information_novelty': self._information_novelty(),
            'subordinate_clause_depth': self._subordinate_clause_depth(),
            'opening_line_quality': self._opening_line_quality(),
            'overall_score': 0.0
        }

        # Calculate overall score
        results['overall_score'] = self._calculate_overall_score(results)

        return results

    def _basic_stats(self) -> Dict:
        """Basic text statistics"""
        word_count = len(re.findall(r'[\w가-힣]+', self.text))
        char_count = len(re.sub(r'\s', '', self.text))

        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': len(self.sentences),
            'paragraph_count': len(self.paragraphs)
        }

    def _sentence_complexity(self) -> Dict:
        """Metric 1: Average sentence length (target: 9-16 words for Korean prose)"""
        if not self.sentences:
            return {'avg_length': 0, 'score': 0.0, 'pass': False}

        # Count Korean words (including particles)
        lengths = []
        for sent in self.sentences:
            # Korean word segmentation approximation
            words = re.findall(r'[\w가-힣]+', sent)
            lengths.append(len(words))

        avg_length = sum(lengths) / len(lengths)

        # Score: 1.0 if in range [9, 16], decrease outside
        if 9 <= avg_length <= 16:
            score = 1.0
        elif avg_length < 9:
            score = max(0.0, avg_length / 9)
        else:
            score = max(0.0, 1.0 - (avg_length - 16) / 16)

        return {
            'avg_length': round(avg_length, 2),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'score': round(score, 2),
            'pass': 9 <= avg_length <= 16
        }

    def _paragraph_density(self) -> Dict:
        """Metric 2: Sentences per paragraph (target: 2-5 for dialogue-heavy prose)"""
        if not self.paragraphs:
            return {'avg_sentences': 0, 'score': 0.0, 'pass': False}

        densities = []
        for para in self.paragraphs:
            sents = re.split(r'[.!?]\s+|[.!?]$|[.!?]"', para)
            sents = [s for s in sents if s.strip()]
            densities.append(len(sents))

        avg_density = sum(densities) / len(densities)

        # Score based on target range [2, 5]
        if 2 <= avg_density <= 5:
            score = 1.0
        elif avg_density < 2:
            score = max(0.0, avg_density / 2)
        else:
            score = max(0.0, 1.0 - (avg_density - 5) / 4)

        return {
            'avg_sentences': round(avg_density, 2),
            'distribution': Counter(densities),
            'score': round(score, 2),
            'pass': 2 <= avg_density <= 5
        }

    def _repetition_patterns(self) -> Dict:
        """Metric 3: Avoid 3+ consecutive similar sentences"""
        if len(self.sentences) < 3:
            return {'violations': 0, 'score': 1.0, 'pass': True}

        violations = 0
        violation_groups = []

        for i in range(len(self.sentences) - 2):
            s1, s2, s3 = self.sentences[i:i+3]

            # Check structural similarity
            # 1. Similar length (within 20%)
            len1, len2, len3 = len(s1), len(s2), len(s3)
            avg_len = (len1 + len2 + len3) / 3
            len_similar = all(abs(l - avg_len) / avg_len < 0.2 for l in [len1, len2, len3])

            # 2. Similar starting pattern (first 3 chars)
            start_pattern = [s[:3] for s in [s1, s2, s3]]
            start_similar = len(set(start_pattern)) <= 2

            # 3. Repeated key words
            words1 = set(re.findall(r'[\w가-힣]{2,}', s1))
            words2 = set(re.findall(r'[\w가-힣]{2,}', s2))
            words3 = set(re.findall(r'[\w가-힣]{2,}', s3))

            overlap12 = len(words1 & words2) / max(len(words1), len(words2), 1)
            overlap23 = len(words2 & words3) / max(len(words2), len(words3), 1)
            word_similar = overlap12 > 0.4 and overlap23 > 0.4

            if (len_similar and start_similar) or word_similar:
                violations += 1
                violation_groups.append((i, i+1, i+2))

        # Score: penalize violations, but avoid over-penalizing natural stylistic echoes
        score = max(0.0, 1.0 - violations * 0.1)

        return {
            'violations': violations,
            'violation_groups': violation_groups[:3],  # Show first 3
            'score': round(score, 2),
            'pass': violations == 0
        }

    def _abstract_concrete_ratio(self) -> Dict:
        """Metric 4: Abstract vs Concrete words (target ratio: 1:3 or less)"""
        # Korean abstract word patterns
        abstract_patterns = [
            r'느낌', r'생각', r'기분', r'마음', r'감정', r'의미', r'가치',
            r'존재', r'본질', r'정신', r'이념', r'철학', r'개념', r'추상',
            r'불안', r'긴장', r'희망', r'두려움', r'공포', r'열정', r'욕망'
        ]

        # Korean concrete word patterns
        concrete_patterns = [
            r'손', r'눈', r'발', r'얼굴', r'목소리', r'소리', r'빛', r'냄새',
            r'문', r'창문', r'의자', r'테이블', r'책상', r'컵', r'종이',
            r'걷', r'앉', r'서', r'보', r'듣', r'만지', r'잡', r'들',
            r'파란', r'빨간', r'하얀', r'검은', r'차가운', r'따뜻한', r'무거운',
            r'카드', r'명함', r'화면', r'스크린', r'장비', r'기계', r'장치'
        ]

        abstract_count = sum(len(re.findall(p, self.text)) for p in abstract_patterns)
        concrete_count = sum(len(re.findall(p, self.text)) for p in concrete_patterns)

        if concrete_count == 0:
            ratio = float('inf') if abstract_count > 0 else 0
            score = 0.0
        else:
            ratio = abstract_count / concrete_count
            # Target: ratio <= 0.33 (1:3)
            if ratio <= 0.33:
                score = 1.0
            else:
                score = max(0.0, 1.0 - (ratio - 0.33) / 0.67)

        return {
            'abstract_count': abstract_count,
            'concrete_count': concrete_count,
            'ratio': round(ratio, 2) if ratio != float('inf') else 'inf',
            'score': round(score, 2),
            'pass': ratio <= 0.33
        }

    def _dialogue_ratio(self) -> Dict:
        """Metric 5: Dialogue percentage (target: 15-30%)"""
        # Korean dialogue patterns: "..." or '...' or ... followed by 말했다/물었다 etc
        dialogue_lines = re.findall(r'"[^"]{5,}"', self.text)

        dialogue_chars = sum(len(line) for line in dialogue_lines)
        total_chars = len(re.sub(r'\s', '', self.text))

        if total_chars == 0:
            return {'percentage': 0, 'score': 0.0, 'pass': False}

        percentage = (dialogue_chars / total_chars) * 100

        # Score based on range [15, 30]
        if 15 <= percentage <= 30:
            score = 1.0
        elif percentage < 15:
            score = max(0.0, percentage / 15)
        else:
            score = max(0.0, 1.0 - (percentage - 30) / 30)

        return {
            'dialogue_lines': len(dialogue_lines),
            'percentage': round(percentage, 2),
            'score': round(score, 2),
            'pass': 15 <= percentage <= 30
        }

    def _scene_progression(self) -> Dict:
        """Metric 6: Scene transitions (target: 1-10 per 50 lines for short chapters)"""
        lines = [l for l in self.text.split('\n') if l.strip() and not l.startswith('#')]
        line_count = len(lines)

        if line_count == 0:
            return {'transitions': 0, 'rate': 0, 'score': 0.0, 'pass': False}

        # Scene transition markers tuned for explicit shift phrases, not generic suffixes.
        transition_markers = [
            r'잠시 후|그 후|이후|한편|곧이어|다음 순간|그때',
            r'자리를 옮|장소를 옮|문을 나서|안으로 들어서|복도로 향|회의실로 향|연구실로 향',
            r'로비로 이동|복도로 이동|회의실로 이동|연구실로 이동'
        ]

        transitions = 0
        for line in lines:
            normalized = line.strip()
            if any(re.search(marker, normalized) for marker in transition_markers):
                transitions += 1

        # Calculate rate per 50 lines
        rate_per_50 = (transitions / line_count) * 50

        # For short chapters, low transition density can still be coherent.
        # Target band: 1-10 transitions per 50 lines.
        if 1 <= rate_per_50 <= 10:
            score = 1.0
        elif rate_per_50 < 1:
            score = 0.7 + (rate_per_50 * 0.3)
        else:
            score = max(0.0, 1.0 - (rate_per_50 - 10) / 15)

        return {
            'transitions': transitions,
            'line_count': line_count,
            'rate_per_50': round(rate_per_50, 2),
            'score': round(score, 2),
            'pass': 1 <= rate_per_50 <= 10
        }

    def _information_novelty(self) -> Dict:
        """Metric 7: Novel information per sentence (target: >70%)"""
        if len(self.sentences) < 2:
            return {'novelty_percentage': 100, 'score': 1.0, 'pass': True}

        novelty_scores = []

        for i, sent in enumerate(self.sentences):
            if i == 0:
                novelty_scores.append(1.0)
                continue

            # Get words from current sentence
            current_words = set(re.findall(r'[\w가-힣]{2,}', sent))

            # Get words from previous 2 sentences
            prev_context = ' '.join(self.sentences[max(0, i-2):i])
            prev_words = set(re.findall(r'[\w가-힣]{2,}', prev_context))

            # Calculate novelty (percentage of new words)
            if not current_words:
                novelty = 0.0
            else:
                new_words = current_words - prev_words
                novelty = len(new_words) / len(current_words)

            novelty_scores.append(novelty)

        avg_novelty = sum(novelty_scores) / len(novelty_scores)
        novelty_percentage = avg_novelty * 100

        # Target: >70%
        if novelty_percentage >= 70:
            score = 1.0
        else:
            score = max(0.0, novelty_percentage / 70)

        return {
            'novelty_percentage': round(novelty_percentage, 2),
            'score': round(score, 2),
            'pass': novelty_percentage >= 70
        }

    def _subordinate_clause_depth(self) -> Dict:
        """Metric 8: Subordinate clause nesting (target: <15% with 3+ levels)"""
        # Korean subordinate clause markers
        markers = [
            r'고', r'며', r'지만', r'는데', r'지만', r'어서', r'아서',
            r'면서', r'으니', r'도록', r'거나', r'자마자'
        ]

        deep_nesting_count = 0

        for sent in self.sentences:
            # Count markers in sentence
            marker_count = sum(len(re.findall(r'\b' + m, sent)) for m in markers)

            # 3+ markers = deep nesting
            if marker_count >= 3:
                deep_nesting_count += 1

        if not self.sentences:
            percentage = 0
            score = 1.0
        else:
            percentage = (deep_nesting_count / len(self.sentences)) * 100

            # Target: <15%
            if percentage < 15:
                score = 1.0
            else:
                score = max(0.0, 1.0 - (percentage - 15) / 35)

        return {
            'deep_nesting_count': deep_nesting_count,
            'total_sentences': len(self.sentences),
            'percentage': round(percentage, 2),
            'score': round(score, 2),
            'pass': percentage < 15
        }

    def _opening_line_quality(self) -> Dict:
        """Metric 9: Opening line efficiency"""
        if not self.sentences:
            return {'length': 0, 'score': 0.0, 'pass': False}

        opening = self.sentences[0]
        words = re.findall(r'[\w가-힣]+', opening)
        word_count = len(words)

        # Check for concrete details
        concrete_in_opening = any(re.search(p, opening) for p in [
            r'손', r'눈', r'문', r'소리', r'빛', r'냄새', r'색',
            r'차갑', r'따뜻', r'무거', r'가벼'
        ])

        # Check for immediate action
        action_verbs = any(re.search(p, opening) for p in [
            r'걷', r'달리', r'앉', r'서', r'열', r'닫', r'잡', r'던지'
        ])

        # Scoring
        # 1. Efficiency (10-20 words)
        if 10 <= word_count <= 20:
            efficiency_score = 1.0
        elif word_count < 10:
            efficiency_score = word_count / 10
        else:
            efficiency_score = max(0.0, 1.0 - (word_count - 20) / 20)

        # 2. Concrete details
        concrete_score = 1.0 if concrete_in_opening else 0.5

        # 3. Action
        action_score = 1.0 if action_verbs else 0.7

        total_score = (efficiency_score + concrete_score + action_score) / 3

        return {
            'opening_line': opening[:100] + '...' if len(opening) > 100 else opening,
            'word_count': word_count,
            'has_concrete_details': concrete_in_opening,
            'has_action': action_verbs,
            'score': round(total_score, 2),
            'pass': total_score >= 0.7
        }

    def _calculate_overall_score(self, results: Dict) -> float:
        """Calculate weighted overall score"""
        weights = {
            'sentence_complexity': 0.15,
            'paragraph_density': 0.10,
            'repetition_patterns': 0.15,
            'abstract_concrete_ratio': 0.15,
            'dialogue_ratio': 0.10,
            'scene_progression': 0.10,
            'information_novelty': 0.10,
            'subordinate_clause_depth': 0.10,
            'opening_line_quality': 0.05
        }

        total_score = 0.0
        for metric, weight in weights.items():
            if metric in results:
                total_score += results[metric].get('score', 0) * weight

        return round(total_score, 3)


def analyze_file(filepath: Path) -> Dict:
    """Analyze a single chapter file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    episode_name = filepath.stem
    analyzer = QualityAnalyzer(text, episode_name)
    return analyzer.analyze()


def print_results(results: Dict):
    """Print analysis results in readable format"""
    print(f"\n{'='*70}")
    print(f"EPISODE: {results['episode']}")
    print(f"{'='*70}\n")

    # Basic stats
    stats = results['basic_stats']
    print(f"📊 BASIC STATS:")
    print(f"   Words: {stats['word_count']} | Sentences: {stats['sentence_count']} | Paragraphs: {stats['paragraph_count']}")
    print()

    # Core metrics
    metrics = [
        ('sentence_complexity', '📏 Sentence Complexity', 'avg_length'),
        ('paragraph_density', '📦 Paragraph Density', 'avg_sentences'),
        ('repetition_patterns', '🔄 Repetition Check', 'violations'),
        ('abstract_concrete_ratio', '🎯 Abstract/Concrete Ratio', 'ratio'),
        ('dialogue_ratio', '💬 Dialogue %', 'percentage'),
        ('scene_progression', '🎬 Scene Progression', 'rate_per_50'),
        ('information_novelty', '✨ Info Novelty', 'novelty_percentage'),
        ('subordinate_clause_depth', '🌲 Clause Depth', 'percentage'),
        ('opening_line_quality', '🚀 Opening Line', 'word_count')
    ]

    for metric_key, label, detail_key in metrics:
        metric = results[metric_key]
        status = '✅' if metric['pass'] else '❌'
        score = metric['score']
        detail = metric.get(detail_key, '')

        print(f"{status} {label}")
        print(f"   Score: {score:.2f} | {detail_key}: {detail}")

    # Overall score
    print(f"\n{'='*70}")
    overall = results['overall_score']
    grade = 'A' if overall >= 0.9 else 'B' if overall >= 0.8 else 'C' if overall >= 0.7 else 'D' if overall >= 0.6 else 'F'
    print(f"🎯 OVERALL SCORE: {overall:.3f} ({grade})")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze chapter quality')
    parser.add_argument('files', nargs='+', help='Chapter files to analyze')
    parser.add_argument('--json', action='store_true', help='Output JSON format')
    parser.add_argument('--output', help='Output file for JSON results')

    args = parser.parse_args()

    all_results = []

    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            print(f"⚠️  File not found: {filepath}")
            continue

        results = analyze_file(path)
        all_results.append(results)

        if not args.json:
            print_results(results)

    if args.json:
        output_data = {
            'episodes': all_results,
            'summary': {
                'total_episodes': len(all_results),
                'avg_overall_score': sum(r['overall_score'] for r in all_results) / len(all_results),
                'passed': sum(1 for r in all_results if r['overall_score'] >= 0.7)
            }
        }

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Results saved to {args.output}")
        else:
            print(json.dumps(output_data, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
