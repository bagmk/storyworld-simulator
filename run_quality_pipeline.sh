#!/bin/bash
# Quality-Controlled Chapter Generation Pipeline
#
# Usage: ./run_quality_pipeline.sh EPISODE_ID
#
# This script:
# 1. Runs simulation if needed
# 2. Generates chapter with quality control
# 3. Analyzes and reports quality metrics
# 4. Iteratively improves until target quality is met

set -e

EPISODE_ID="$1"

if [ -z "$EPISODE_ID" ]; then
    echo "Usage: $0 EPISODE_ID"
    echo "Example: $0 ep02_nsa_funding"
    exit 1
fi

# Configuration
CONFIG_DIR="config/episodes"
EPISODE_CONFIG="${CONFIG_DIR}/${EPISODE_ID}.yaml"
OUTPUT_DIR="output"
TARGET_WORDS=800
NUM_SCENES=3
TARGET_SCORE=0.80
MAX_ITERATIONS=3

echo "=========================================="
echo "  Quality-Controlled Pipeline"
echo "=========================================="
echo "Episode: $EPISODE_ID"
echo "Target: $TARGET_WORDS words, $NUM_SCENES scenes"
echo "Quality Target: $TARGET_SCORE"
echo "Max Iterations: $MAX_ITERATIONS"
echo ""

# Check if episode config exists
if [ ! -f "$EPISODE_CONFIG" ]; then
    echo "❌ Episode config not found: $EPISODE_CONFIG"
    exit 1
fi

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not set"
    echo "   Set with: export OPENAI_API_KEY=your_key_here"
    exit 1
fi

# Step 1: Check if simulation data exists
echo "📊 Step 1: Checking simulation data..."
DB_FILE="data/simulation.db"

if [ ! -f "$DB_FILE" ]; then
    echo "❌ Database not found: $DB_FILE"
    echo "   Run simulation first: python3 trial_simulate.py --episode $EPISODE_ID"
    exit 1
fi

# Check if interactions/emotions exist
INTERACTION_COUNT=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM interactions WHERE episode_id='$EPISODE_ID'")
EMOTION_COUNT=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM emotions WHERE episode_id='$EPISODE_ID'")

if [ "$INTERACTION_COUNT" -eq 0 ] || [ "$EMOTION_COUNT" -eq 0 ]; then
    if [ "$INTERACTION_COUNT" -eq 0 ]; then
        echo "⚠️  No simulation data found for $EPISODE_ID"
    fi
    if [ "$EMOTION_COUNT" -eq 0 ]; then
        echo "⚠️  No emotion data found for $EPISODE_ID"
    fi

    SUMMARY_JSON="${OUTPUT_DIR}/${EPISODE_ID}_trial_summary.json"
    SIMULATION_JSON=""

    # Prefer existing trial outputs first (faster than rerunning).
    if [ -f "$SUMMARY_JSON" ]; then
        WINNING_TRIAL=$(python3 -c "
import json
from pathlib import Path
p = Path('$SUMMARY_JSON')
if p.exists():
    data = json.loads(p.read_text(encoding='utf-8'))
    win = data.get('winning_trial')
    print(win if isinstance(win, int) else '')
")
        if [ -n "$WINNING_TRIAL" ] && [ -f "${OUTPUT_DIR}/${EPISODE_ID}_trial${WINNING_TRIAL}_simulation.json" ]; then
            SIMULATION_JSON="${OUTPUT_DIR}/${EPISODE_ID}_trial${WINNING_TRIAL}_simulation.json"
        fi
    fi

    if [ -z "$SIMULATION_JSON" ]; then
        FALLBACK_SIM="${OUTPUT_DIR}/${EPISODE_ID}_trial1_simulation.json"
        if [ -f "$FALLBACK_SIM" ]; then
            SIMULATION_JSON="$FALLBACK_SIM"
        fi
    fi

    if [ -z "$SIMULATION_JSON" ]; then
        echo "   Running simulation..."
        python3 trial_simulate.py \
            --episode "$EPISODE_CONFIG" \
            --characters "config/characters.yaml" \
            --world "config/world_facts.yaml" \
            --storyline "config/storyline.yaml" \
            --max-trials 3 \
            --output "$OUTPUT_DIR"

        SUMMARY_JSON="${OUTPUT_DIR}/${EPISODE_ID}_trial_summary.json"
        WINNING_TRIAL=$(python3 -c "
import json
from pathlib import Path
p = Path('$SUMMARY_JSON')
if p.exists():
    data = json.loads(p.read_text(encoding='utf-8'))
    win = data.get('winning_trial')
    print(win if isinstance(win, int) else '')
")
        if [ -n "$WINNING_TRIAL" ] && [ -f "${OUTPUT_DIR}/${EPISODE_ID}_trial${WINNING_TRIAL}_simulation.json" ]; then
            SIMULATION_JSON="${OUTPUT_DIR}/${EPISODE_ID}_trial${WINNING_TRIAL}_simulation.json"
        elif [ -f "${OUTPUT_DIR}/${EPISODE_ID}_trial1_simulation.json" ]; then
            SIMULATION_JSON="${OUTPUT_DIR}/${EPISODE_ID}_trial1_simulation.json"
        fi
    fi

    if [ -f "$SIMULATION_JSON" ]; then
        echo "   Importing simulation data from $SIMULATION_JSON ..."
        python3 -c "
import json
import sqlite3
import uuid
from datetime import datetime

with open('$SIMULATION_JSON', 'r', encoding='utf-8') as f:
    data = json.load(f)

conn = sqlite3.connect('$DB_FILE')
cursor = conn.cursor()

# Rebuild episode rows to keep interactions/emotions consistent.
cursor.execute('DELETE FROM emotions WHERE episode_id = ?', ('$EPISODE_ID',))
cursor.execute('DELETE FROM interactions WHERE episode_id = ?', ('$EPISODE_ID',))

inserted_interactions = 0
inserted_emotions = 0

for i, interaction in enumerate(data.get('interactions', [])):
    ix_id = str(interaction.get('id') or uuid.uuid4())
    turn = int(interaction.get('turn') or (i + 1))
    speaker_id = interaction.get('speaker_id') or interaction.get('agent_id') or 'unknown'
    speaker_name = interaction.get('speaker_name') or interaction.get('agent_name') or speaker_id
    content = interaction.get('content') or interaction.get('response') or ''
    action_type = interaction.get('action_type') or 'dialogue'
    timestamp = interaction.get('timestamp') or datetime.utcnow().isoformat()
    metadata = interaction.get('metadata') or {}
    if not isinstance(metadata, dict):
        metadata = {}

    cursor.execute('''
        INSERT INTO interactions
        (id, episode_id, turn, speaker_id, speaker_name, content, action_type, timestamp, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        ix_id,
        '$EPISODE_ID',
        turn,
        speaker_id,
        speaker_name,
        content,
        action_type,
        timestamp,
        json.dumps(metadata, ensure_ascii=False)
    ))
    inserted_interactions += 1

    emotions = metadata.get('emotions', {}) if isinstance(metadata, dict) else {}
    if isinstance(emotions, dict):
        for emotion_type, intensity in emotions.items():
            if not isinstance(emotion_type, str):
                continue
            try:
                val = float(intensity)
            except (TypeError, ValueError):
                continue
            if val <= 0:
                continue
            val = max(0.0, min(1.0, val))
            cursor.execute('''
                INSERT INTO emotions
                (agent_id, interaction_id, episode_id, turn, emotion_type, intensity, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                speaker_id,
                ix_id,
                '$EPISODE_ID',
                turn,
                emotion_type,
                val,
                timestamp,
            ))
            inserted_emotions += 1

conn.commit()
conn.close()
print('✅ Imported', inserted_interactions, 'interactions and', inserted_emotions, 'emotions')
"
    else
        echo "❌ Simulation JSON not found after run/import attempt"
        exit 1
    fi
else
    echo "✅ Found $INTERACTION_COUNT interactions and $EMOTION_COUNT emotions for $EPISODE_ID"
fi

# Step 2: Generate chapter with quality control
echo ""
echo "📝 Step 2: Generating chapter with quality control..."

python3 quality_adaptive_generator.py \
    --episode-id "$EPISODE_ID" \
    --episode-config "$EPISODE_CONFIG" \
    --protagonist "kim_sumin" \
    --target-words "$TARGET_WORDS" \
    --scenes "$NUM_SCENES" \
    --target-score "$TARGET_SCORE" \
    --max-iterations "$MAX_ITERATIONS"

# Step 3: Analyze final quality
echo ""
echo "📊 Step 3: Final quality analysis..."

LATEST_CHAPTER=$(ls -t "${OUTPUT_DIR}/${EPISODE_ID}"*chapter.md 2>/dev/null | head -1)

if [ -n "$LATEST_CHAPTER" ]; then
    echo "   Analyzing: $LATEST_CHAPTER"
    python3 quality_analyzer.py "$LATEST_CHAPTER"

    # Save quality report
    QUALITY_REPORT="${OUTPUT_DIR}/${EPISODE_ID}_quality_report.json"
    python3 quality_analyzer.py "$LATEST_CHAPTER" --json --output "$QUALITY_REPORT"
    echo "   Quality report saved: $QUALITY_REPORT"
else
    echo "❌ No chapter found"
    exit 1
fi

echo ""
echo "=========================================="
echo "  Pipeline Complete!"
echo "=========================================="
echo "Chapter: $LATEST_CHAPTER"
echo "Quality Report: $QUALITY_REPORT"
