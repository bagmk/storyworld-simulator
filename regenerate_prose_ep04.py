#!/usr/bin/env python3
"""
Regenerate prose for EP04 using existing simulation data
"""

import json
import logging
from pathlib import Path
from scene_distiller import SceneDistiller
from prose_generator import ProseGenerator
from config_loader import load_episode, load_storyline
from llm_client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def main():
    # Paths
    simulation_file = Path("output/ep04_lab_whispers_trial1_simulation.json")
    ep_config_file = Path("config/episodes/ep04_lab_whispers.yaml")
    storyline_file = Path("config/storyline.yaml")
    output_file = Path("output/ep04_lab_whispers_chapter.md")

    # Load simulation data
    print(f"Loading simulation: {simulation_file}")
    with open(simulation_file, "r", encoding="utf-8") as f:
        simulation_data = json.load(f)

    # Load episode config
    print(f"Loading episode config: {ep_config_file}")
    ep_config = load_episode(ep_config_file)

    # Load storyline for context
    print(f"Loading storyline: {storyline_file}")
    storyline = load_storyline(storyline_file)

    # Initialize LLM client
    llm = LLMClient(model="gpt-4o-mini")

    # Distill scenes
    print("Distilling scenes...")
    distiller = SceneDistiller(llm=llm, protagonist="Kim Sumin")
    scenes = distiller.distill(
        interactions=simulation_data.get("interactions", []),
        emotions=simulation_data.get("emotions", []),
        num_scenes=3
    )

    print(f"Distilled {len(scenes)} scenes")
    for i, scene in enumerate(scenes, 1):
        print(f"  Scene {i}: {scene.title} ({len(scene.key_dialogue)} dialogue, {len(scene.key_actions)} actions)")

    # Generate prose
    print("Generating prose...")
    generator = ProseGenerator(llm=llm)

    # Build episode metadata
    episode_meta = {
        "episode_number": 4,
        "total_episodes": 49,
        "location": ep_config.get("location", "Unknown"),
        "pacing": ep_config.get("pacing", "Medium"),
        "pacing_tone": ep_config.get("pacing_tone", ""),
        "protagonist": ep_config.get("protagonist", "Kim Sumin"),
        "title": ep_config.get("title", "Lab Whispers"),
    }

    # Generate chapter
    chapter_text = generator.generate_chapter(
        scenes=scenes,
        episode=episode_meta,
        target_words=800,
        pov="1st person"
    )

    # Write output
    print(f"Writing to {output_file}")
    output_file.write_text(chapter_text, encoding="utf-8")

    # Count words for verification
    word_count = len(chapter_text.split())
    print(f"✓ Generated {word_count} words")

    # Check for dialogue (quick check for quotation marks)
    quote_count = chapter_text.count('"')
    print(f"✓ Found {quote_count} quotation marks (dialogue indicator)")

    print("\n✓ Done! Run quality_analyzer.py to check metrics.")

if __name__ == "__main__":
    main()
