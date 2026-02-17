#!/usr/bin/env python3
"""
Test script to verify story arc integration with Director AI.
"""

import yaml
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.novel_writer.director import DirectorAI
from src.novel_writer.llm_client import LLMClient
from src.novel_writer.models import ClueManager


def load_yaml(path: str) -> dict:
    """Load YAML file"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_story_arc_loading():
    """Test that story arcs are properly loaded and contextualized"""

    print("="*70)
    print("  STORY ARC INTEGRATION TEST")
    print("="*70)

    # Load configs
    storyline = load_yaml("config/storyline.yaml")
    world_facts = load_yaml("config/world_facts.yaml")

    # Test episodes from different arcs
    test_episodes = [
        ("ep01_academic_presentation", "SETUP"),
        ("ep02_nsa_funding", "SETUP"),
        ("ep09_opening", "DISCOVERY"),
        ("ep24_elena_past", "TECHNICAL"),
        ("ep31_kidnap_moreno", "CRISIS"),
    ]

    llm = LLMClient(
        model="gpt-4o-mini",
        premium_model="gpt-4o",
    )

    for episode_id, expected_arc in test_episodes:
        print(f"\n{'─'*70}")
        print(f"Testing: {episode_id}")
        print(f"Expected Arc: {expected_arc}")
        print(f"{'─'*70}")

        # Try to load episode config
        episode_config_path = Path(f"config/episodes/{episode_id}.yaml")
        if not episode_config_path.exists():
            print(f"⚠️  Episode config not found: {episode_config_path}")
            print(f"   Creating minimal config for testing...")

            # Create minimal config
            episode_config = {
                "id": episode_id,
                "summary": f"Test episode for {episode_id}",
                "max_turns": 12,
                "characters": [],
                "must_resolve": [],
            }
        else:
            episode_config = load_yaml(str(episode_config_path))

        # Create Director
        clue_manager = ClueManager([])
        director = DirectorAI(
            episode_config=episode_config,
            world_facts=world_facts,
            clue_manager=clue_manager,
            llm=llm,
            storyline=storyline,
        )

        # Get storyline guidance
        guidance = director.get_storyline_guidance()

        if guidance:
            print(f"\n📋 Storyline Guidance:\n")
            print(guidance)

            # Check arc info
            arc_info = director.storyline_context.get("story_arc", {})
            if arc_info:
                actual_arc = arc_info.get("name", "UNKNOWN")
                match = "✅" if actual_arc == expected_arc else "❌"
                print(f"\n{match} Arc Match: {actual_arc} (expected: {expected_arc})")

                print(f"\n📊 Arc Details:")
                print(f"   Position: {arc_info.get('act_position')}")
                print(f"   Episode: {arc_info.get('episode_in_arc')}/{arc_info.get('total_in_arc')}")
                print(f"   Progress: {arc_info.get('progress_percentage')}%")
                print(f"   Emotional: {arc_info.get('emotional_trajectory')}")

                if arc_info.get('is_arc_opening'):
                    print(f"   🌟 Arc Opening Episode")
                if arc_info.get('is_arc_climax'):
                    print(f"   🔥 Arc Climax Episode")
            else:
                print(f"❌ No arc info found")
        else:
            print(f"❌ No storyline guidance generated")

    print(f"\n{'='*70}")
    print("  TEST COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    test_story_arc_loading()
