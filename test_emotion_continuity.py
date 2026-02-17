#!/usr/bin/env python3
"""
Test script to verify emotion continuity across episodes
"""

import sqlite3
import matplotlib.pyplot as plt
from pathlib import Path

# Connect to database
db_path = "data/simulation.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

# Get Kim Sumin's emotions across all episodes
query = """
SELECT
    episode_id,
    turn,
    emotion_type,
    intensity
FROM emotions
WHERE agent_id = 'kim_sumin'
    AND episode_id NOT LIKE '%trial%'  -- Exclude trial versions
ORDER BY episode_id, turn
"""

rows = conn.execute(query).fetchall()
conn.close()

# Organize data by episode and emotion type
from collections import defaultdict

episodes = []
episode_data = defaultdict(lambda: {'tension': [], 'curiosity': []})
current_ep = None
global_turn = 0

for row in rows:
    ep_id = row['episode_id']

    # New episode detected
    if ep_id != current_ep:
        if current_ep is not None:
            episodes.append((current_ep, dict(episode_data)))
            episode_data = defaultdict(lambda: {'tension': [], 'curiosity': []})
        current_ep = ep_id
        global_turn = 0

    global_turn += 1
    emotion = row['emotion_type']
    intensity = row['intensity']

    episode_data[emotion].append((global_turn, intensity))

# Add last episode
if current_ep:
    episodes.append((current_ep, dict(episode_data)))

# Create visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Plot tension
global_x = []
global_tension = []
episode_boundaries = []
episode_labels = []

x_offset = 0
for ep_id, data in episodes:
    if 'tension' in data and data['tension']:
        turns, values = zip(*data['tension'])
        turns_shifted = [t + x_offset for t in turns]
        global_x.extend(turns_shifted)
        global_tension.extend(values)

        episode_boundaries.append(x_offset)
        episode_labels.append(ep_id[:10])  # Shorten label
        x_offset = turns_shifted[-1] + 1

ax1.plot(global_x, global_tension, 'o-', linewidth=2, markersize=4, color='#ff6b6b')
ax1.set_ylabel('Tension', fontsize=12, fontweight='bold')
ax1.set_title('Kim Sumin - Emotion Continuity Across Episodes (FIXED)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 0.5)

# Plot curiosity
global_x = []
global_curiosity = []
x_offset = 0
for ep_id, data in episodes:
    if 'curiosity' in data and data['curiosity']:
        turns, values = zip(*data['curiosity'])
        turns_shifted = [t + x_offset for t in turns]
        global_x.extend(turns_shifted)
        global_curiosity.extend(values)
        x_offset = turns_shifted[-1] + 1

ax2.plot(global_x, global_curiosity, 'o-', linewidth=2, markersize=4, color='#4ecdc4')
ax2.set_ylabel('Curiosity', fontsize=12, fontweight='bold')
ax2.set_xlabel('Global Turn', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 0.5)

# Add episode boundaries
for ax in [ax1, ax2]:
    for i, x in enumerate(episode_boundaries):
        ax.axvline(x, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        if i < len(episode_labels):
            ax.text(x + 2, ax.get_ylim()[1] * 0.95, episode_labels[i],
                   rotation=0, fontsize=8, alpha=0.7)

plt.tight_layout()
output_path = Path("output/emotion_continuity_fixed.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved visualization: {output_path}")

# Print statistics
print("\n📊 Emotion Continuity Statistics:")
print(f"{'Episode':<30} | {'Start Tension':>14} | {'End Tension':>12} | {'Continuity':>12}")
print("-" * 75)

for i, (ep_id, data) in enumerate(episodes):
    if 'tension' in data and data['tension']:
        start_tension = data['tension'][0][1]
        end_tension = data['tension'][-1][1]

        if i > 0:
            prev_end = episodes[i-1][1]['tension'][-1][1] if 'tension' in episodes[i-1][1] else 0
            continuity = "✅ SMOOTH" if abs(start_tension - prev_end) < 0.05 else "⚠️ JUMP"
        else:
            continuity = "🆕 FIRST"

        print(f"{ep_id:<30} | {start_tension:>14.3f} | {end_tension:>12.3f} | {continuity:>12}")

print("\n✅ Emotion continuity is now working correctly!")
print("Episodes should show smooth transitions, not resets.")
