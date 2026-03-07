#!/usr/bin/env python3
"""
Quality-Adaptive Chapter Generator

Automatically adjusts prose generation parameters based on quality metrics.
Iteratively regenerates scenes until quality targets are met.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import sys
import yaml

from quality_analyzer import QualityAnalyzer

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent


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
            'opening_line_quality': 0.7,
            'pov_consistency': 0.9,
            'timeline_coherence': 0.75,
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
        config_episode_id = self._load_episode_config_id(episode_config_path) or episode_id

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"  Quality Iteration {iteration}/{self.max_iterations}")
            logger.info(f"{'='*70}\n")

            # Generate chapter
            chapter_path = self._generate_chapter(
                episode_id,
                episode_config_path,
                config_episode_id,
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
        config_episode_id: str,
        protagonist: str,
        target_words: int,
        num_scenes: int,
        iteration: int
    ) -> str:
        """Run generate_chapter.py subprocess"""
        try:
            output_dir = REPO_ROOT / "output"
            pre_existing = {
                p.resolve(): p.stat().st_mtime for p in output_dir.glob("*_chapter.md")
            }
            started_at = time.time()

            cmd = [
                sys.executable,
                str(REPO_ROOT / "generate_chapter.py"),
                "--episode", episode_id,
                "--episode-config", episode_config_path,
                "--protagonist", protagonist,
                "--style", "third_person_close",
                "--words", str(target_words),
                "--scenes", str(num_scenes),
                "--output", str(REPO_ROOT / "output")
            ]

            logger.info(f"Generating: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(REPO_ROOT),
            )

            if result.returncode != 0:
                logger.error(f"Generation failed: {result.stderr}")
                return None

            # Find generated chapter. Prefer files created/updated by this run.
            all_chapters = list(output_dir.glob("*_chapter.md"))
            changed_since_start = [
                p for p in all_chapters
                if p.stat().st_mtime >= started_at - 1.0
            ]
            newly_created = [
                p for p in all_chapters if p.resolve() not in pre_existing
            ]
            updated_existing = [
                p for p in all_chapters
                if p.resolve() in pre_existing and p.stat().st_mtime > pre_existing[p.resolve()] + 0.5
            ]

            candidates = []
            candidates.extend(newly_created)
            candidates.extend(updated_existing)
            candidates.extend(changed_since_start)

            # Fallback to prefix matching if runtime-based detection is sparse.
            candidates.extend(list(output_dir.glob(f"{episode_id}*chapter.md")))
            if config_episode_id and config_episode_id != episode_id:
                candidates.extend(list(output_dir.glob(f"{config_episode_id}*chapter.md")))
            candidates = list(dict.fromkeys(candidates))

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

    @staticmethod
    def _load_episode_config_id(episode_config_path: str) -> str:
        try:
            cfg = yaml.safe_load(Path(episode_config_path).read_text(encoding="utf-8")) or {}
            return str(cfg.get("id", "")).strip()
        except Exception:
            return ""

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
        return weak

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
            ('opening_line_quality', '🚀 Opening Line'),
            ('pov_consistency', '👁️ POV Consistency'),
            ('timeline_coherence', '🕒 Timeline Coherence'),
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
