#!/usr/bin/env python3
"""
Quality-Adaptive Chapter Generator

Automatically adjusts prose generation parameters based on quality metrics.
Iteratively regenerates scenes until quality targets are met.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import sys
import re

from quality_analyzer import QualityAnalyzer

logger = logging.getLogger(__name__)


class QualityAdaptiveGenerator:
    """
    Generates chapters with automatic quality improvement iterations.

    Process:
    1. Generate chapter with current settings
    2. Analyze quality metrics
    3. If quality < threshold, adjust prompts and regenerate
    4. Repeat until quality targets met or max iterations reached
    """

    def __init__(
        self,
        target_overall_score: float = 0.8,
        max_iterations: int = 3,
        quality_thresholds: Dict[str, float] = None
    ):
        self.target_overall_score = target_overall_score
        self.max_iterations = max_iterations

        # Default quality thresholds for each metric
        self.quality_thresholds = quality_thresholds or {
            'sentence_complexity': 0.7,
            'paragraph_density': 0.7,
            'repetition_patterns': 0.9,
            'abstract_concrete_ratio': 0.7,
            'dialogue_ratio': 0.6,
            'scene_progression': 0.5,
            'information_novelty': 0.8,
            'subordinate_clause_depth': 0.8,
            'opening_line_quality': 0.7
        }

    def generate_with_quality_control(
        self,
        episode_id: str,
        episode_config_path: str,
        protagonist: str,
        target_words: int = 800,
        num_scenes: int = 3
    ) -> Tuple[str, Dict]:
        """
        Generate chapter with iterative quality improvement.

        Returns:
            (chapter_path, final_quality_results)
        """
        iteration = 0
        best_score = 0.0
        best_chapter_path = None
        best_results = None
        safe_episode_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", episode_id or "unknown")
        feedback_path = Path(f"output/quality_feedback_{safe_episode_id}.json")
        if feedback_path.exists():
            feedback_path.unlink()

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"  Quality Iteration {iteration}/{self.max_iterations}")
            logger.info(f"{'='*70}\n")

            # Generate chapter
            chapter_path = self._generate_chapter(
                episode_id,
                episode_config_path,
                protagonist,
                target_words,
                num_scenes,
                iteration
            )

            if not chapter_path:
                logger.error("Chapter generation failed")
                continue

            # Analyze quality
            results = self._analyze_quality(chapter_path)
            overall_score = results['overall_score']

            logger.info(f"📊 Quality Score: {overall_score:.3f}")

            # Track best result
            if overall_score > best_score:
                best_score = overall_score
                best_chapter_path = chapter_path
                best_results = results

            # Identify weak metrics and adjust
            weak_metrics = self._identify_weak_metrics(results)

            # Check if target met with hard constraints
            if overall_score >= self.target_overall_score and not weak_metrics:
                logger.info(f"✅ Target quality {self.target_overall_score} achieved!")
                break

            if not weak_metrics:
                logger.info("No weak metrics to improve")
                break

            logger.info(f"⚠️  Weak metrics: {', '.join(weak_metrics)}")

            # Write reinforcement feedback for next iteration
            self._write_reinforcement_feedback(
                episode_id=episode_id,
                weak_metrics=weak_metrics,
                results=results,
                iteration=iteration
            )

        # Summary
        logger.info(f"\n{'='*70}")
        logger.info(f"  Quality Control Complete")
        logger.info(f"{'='*70}")
        logger.info(f"Iterations: {iteration}")
        logger.info(f"Best Score: {best_score:.3f}")
        logger.info(f"Best Chapter: {best_chapter_path}")

        if best_results:
            self._print_quality_summary(best_results)
        else:
            logger.error("No successful result to summarize")

        return best_chapter_path, best_results

    def _generate_chapter(
        self,
        episode_id: str,
        episode_config_path: str,
        protagonist: str,
        target_words: int,
        num_scenes: int,
        iteration: int
    ) -> str:
        """Run generate_chapter.py subprocess"""
        try:
            output_file = f"output/{episode_id}_iter{iteration}_chapter.md"

            cmd = [
                sys.executable,
                "generate_chapter.py",
                "--episode", episode_id,
                "--episode-config", episode_config_path,
                "--protagonist", protagonist,
                "--words", str(target_words),
                "--scenes", str(num_scenes),
                "--output", "output"
            ]

            logger.info(f"Generating: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                logger.error(f"Generation failed: {result.stderr}")
                return None

            # Find generated chapter
            output_dir = Path("output")
            candidates = list(output_dir.glob(f"{episode_id}*chapter.md"))

            if not candidates:
                logger.error("No chapter file found after generation")
                return None

            # Get most recent
            latest = max(candidates, key=lambda p: p.stat().st_mtime)

            # Rename to include iteration
            new_path = output_dir / f"{episode_id}_iter{iteration}_chapter.md"
            if latest != new_path:
                latest.rename(new_path)

            logger.info(f"✅ Generated: {new_path}")
            return str(new_path)

        except subprocess.TimeoutExpired:
            logger.error("Generation timeout")
            return None
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return None

    def _analyze_quality(self, chapter_path: str) -> Dict:
        """Analyze chapter quality"""
        with open(chapter_path, 'r', encoding='utf-8') as f:
            text = f.read()

        analyzer = QualityAnalyzer(text, Path(chapter_path).stem)
        return analyzer.analyze()

    def _identify_weak_metrics(self, results: Dict) -> List[str]:
        """Identify metrics below threshold"""
        weak = []

        for metric_name, threshold in self.quality_thresholds.items():
            if metric_name not in results:
                continue

            metric_score = results[metric_name].get('score', 0)

            if metric_score < threshold:
                weak.append(metric_name)

        # Hard constraint override for dialogue balance
        dialogue = results.get('dialogue_ratio', {})
        dialogue_pct = dialogue.get('percentage')
        if isinstance(dialogue_pct, (int, float)):
            if dialogue_pct < 20 or dialogue_pct > 23:
                if 'dialogue_ratio' not in weak:
                    weak.append('dialogue_ratio')

        return weak

    def _build_reinforcement_directives(self, weak_metrics: List[str], results: Dict) -> List[str]:
        """
        Build prompt directives based on weak metrics.
        These directives are persisted and consumed by prose_generator.py.
        """
        directives = []

        if 'sentence_complexity' in weak_metrics:
            avg_len = results['sentence_complexity'].get('avg_length', 0)
            if avg_len < 9:
                directives.append(
                    "CRITICAL: Sentences TOO SHORT (avg={:.1f}). "
                    "MUST combine actions with conjunctions: -고, -며, -면서. "
                    "Target 10-14 words/sentence.".format(avg_len)
                )
            elif avg_len > 16:
                directives.append(
                    "CRITICAL: Sentences TOO LONG (avg={:.1f}). "
                    "Break into shorter units. Target 10-14 words/sentence.".format(avg_len)
                )

        if 'dialogue_ratio' in weak_metrics:
            pct = results['dialogue_ratio'].get('percentage', 0)
            if pct < 20:
                directives.append(
                    "CRITICAL: NOT ENOUGH DIALOGUE ({:.1f}%). "
                    "MUST include at least 5 quoted exchanges. "
                    "Characters must speak directly in quotes. Target 20-23%.".format(pct)
                )
            elif pct > 23:
                directives.append(
                    "CRITICAL: TOO MUCH DIALOGUE ({:.1f}%). "
                    "Balance with action and description. Target 20-23%.".format(pct)
                )

        if 'paragraph_density' in weak_metrics:
            avg_sents = results['paragraph_density'].get('avg_sentences', 0)
            if avg_sents < 2:
                directives.append(
                    "CRITICAL: Paragraphs TOO SHORT (avg={:.1f} sentences). "
                    "Combine related actions into 2-3 sentence paragraphs.".format(avg_sents)
                )
            elif avg_sents > 5:
                directives.append(
                    "CRITICAL: Paragraphs TOO LONG (avg={:.1f} sentences). "
                    "Split into shorter units. Keep around 2-5 sentences/paragraph.".format(avg_sents)
                )

        if 'abstract_concrete_ratio' in weak_metrics:
            ratio = results['abstract_concrete_ratio'].get('ratio', 0)
            if ratio == 'inf' or (isinstance(ratio, (int, float)) and ratio > 0.33):
                directives.append(
                    "CRITICAL: TOO MANY ABSTRACT WORDS (ratio={}). "
                    "SHOW concrete actions: 손이 떨렸다, 목소리가 낮았다. "
                    "AVOID: 불안했다, 긴장됐다. Use 3:1 concrete:abstract.".format(ratio)
                )

        if 'repetition_patterns' in weak_metrics:
            violations = results['repetition_patterns'].get('violations', 0)
            directives.append(
                f"CRITICAL: {violations} REPETITION violations. "
                "Vary sentence structure. No 3+ similar sentences in a row."
            )

        if 'information_novelty' in weak_metrics:
            novelty = results['information_novelty'].get('novelty_percentage', 0)
            directives.append(
                f"CRITICAL: Low information novelty ({novelty:.1f}%). "
                "Each sentence must introduce new details. Avoid repetition."
            )

        if 'scene_progression' in weak_metrics:
            rate = results['scene_progression'].get('rate_per_50', 0)
            if rate > 3:
                directives.append(
                    f"CRITICAL: TOO MANY scene transitions ({rate:.1f} per 50 lines). "
                    "Stay in scenes longer. Develop moments. Target 2-3 per 50 lines."
                )

        return directives

    def _write_reinforcement_feedback(
        self,
        episode_id: str,
        weak_metrics: List[str],
        results: Dict,
        iteration: int
    ) -> None:
        """
        Persist reinforcement directives for the next generation iteration.
        """
        directives = self._build_reinforcement_directives(weak_metrics, results)
        payload = {
            "episode_id": episode_id,
            "iteration": iteration,
            "weak_metrics": weak_metrics,
            "directives": directives
        }

        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_episode_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", episode_id or "unknown")
        feedback_path = output_dir / f"quality_feedback_{safe_episode_id}.json"
        feedback_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        logger.info(f"✅ Reinforcement feedback written: {feedback_path}")

    def _print_quality_summary(self, results: Dict):
        """Print quality summary"""
        print("\n" + "="*70)
        print("  QUALITY SUMMARY")
        print("="*70)

        metrics = [
            ('sentence_complexity', '📏 Sentence Complexity'),
            ('paragraph_density', '📦 Paragraph Density'),
            ('repetition_patterns', '🔄 Repetition Check'),
            ('abstract_concrete_ratio', '🎯 Abstract/Concrete'),
            ('dialogue_ratio', '💬 Dialogue %'),
            ('scene_progression', '🎬 Scene Progression'),
            ('information_novelty', '✨ Info Novelty'),
            ('subordinate_clause_depth', '🌲 Clause Depth'),
            ('opening_line_quality', '🚀 Opening Line')
        ]

        for metric_key, label in metrics:
            if metric_key not in results:
                continue

            metric = results[metric_key]
            status = '✅' if metric['pass'] else '❌'
            score = metric['score']

            print(f"{status} {label}: {score:.2f}")

        overall = results['overall_score']
        grade = 'A' if overall >= 0.9 else 'B' if overall >= 0.8 else 'C' if overall >= 0.7 else 'D' if overall >= 0.6 else 'F'
        print(f"\n🎯 OVERALL: {overall:.3f} ({grade})")
        print("="*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate chapter with automatic quality improvement'
    )
    parser.add_argument('--episode-id', required=True, help='Episode ID')
    parser.add_argument('--episode-config', required=True, help='Episode YAML config')
    parser.add_argument('--protagonist', required=True, help='Protagonist agent ID')
    parser.add_argument('--target-words', type=int, default=800, help='Target word count')
    parser.add_argument('--scenes', type=int, default=3, help='Number of scenes')
    parser.add_argument('--target-score', type=float, default=0.8, help='Target quality score')
    parser.add_argument('--max-iterations', type=int, default=3, help='Max improvement iterations')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    generator = QualityAdaptiveGenerator(
        target_overall_score=args.target_score,
        max_iterations=args.max_iterations
    )

    chapter_path, results = generator.generate_with_quality_control(
        episode_id=args.episode_id,
        episode_config_path=args.episode_config,
        protagonist=args.protagonist,
        target_words=args.target_words,
        num_scenes=args.scenes
    )

    if chapter_path:
        print(f"\n✅ Best chapter: {chapter_path}")
        print(f"📊 Final score: {results['overall_score']:.3f}")
    else:
        print("\n❌ Chapter generation failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
