"""Analyze CSV to find patterns of missed blinks."""
import csv

csv_path = r'E:\Projects\Graduation Project AI\manual_tests\liveness_detection\ear_diagnostic_data.csv'

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# Filter face-detected frames
face_data = [row for row in data if row['face_detected'] == 'True']

print("=" * 70)
print("MISSED BLINK ANALYSIS")
print("=" * 70)

# Find all transitions from OPEN to below close threshold (0.21)
# These are potential blinks that might be missed

transitions = []
prev_ear = None
prev_state = None

for i, row in enumerate(face_data):
    ear = float(row['ear'])
    state = row['eye_state']

    if prev_ear is not None:
        # Detect when EAR drops below close threshold
        if prev_ear >= 0.21 and ear < 0.21:
            transitions.append({
                'frame': i,
                'type': 'potential_blink_start',
                'ear_from': prev_ear,
                'ear_to': ear,
                'state': state
            })

        # Detect state changes
        if prev_state != state:
            transitions.append({
                'frame': i,
                'type': f'state_change_{prev_state}_to_{state}',
                'ear': ear,
                'state': state
            })

    prev_ear = ear
    prev_state = state

print(f"\nTotal EAR drops below 0.21: {sum(1 for t in transitions if t['type'] == 'potential_blink_start')}")
print(f"Total state changes: {sum(1 for t in transitions if 'state_change' in t['type'])}")

# Look for rapid EAR fluctuations (potential missed blinks)
rapid_drops = []
for i in range(len(face_data) - 1):
    ear1 = float(face_data[i]['ear'])
    ear2 = float(face_data[i+1]['ear'])

    # Check for rapid drop and recovery (possible missed blink)
    if ear1 > 0.25 and ear2 < 0.22 and i+2 < len(face_data):
        ear3 = float(face_data[i+2]['ear'])
        if ear3 > 0.25:
            # Rapid drop and recovery in 2 frames - possible missed blink
            rapid_drops.append({
                'frame': i,
                'ear_before': ear1,
                'ear_drop': ear2,
                'ear_after': ear3,
                'state': face_data[i+1]['eye_state']
            })

print(f"\nRapid EAR fluctuations (2-frame patterns): {len(rapid_drops)}")

if rapid_drops:
    print("\nFirst 5 rapid fluctuations:")
    for i, drop in enumerate(rapid_drops[:5], 1):
        print(f"  {i}. Frame {drop['frame']}: {drop['ear_before']:.3f} -> {drop['ear_drop']:.3f} -> {drop['ear_after']:.3f} (state: {drop['state']})")

# Check smoothing effect
print("\n" + "=" * 70)
print("SMOOTHING ANALYSIS (current window = 3)")
print("=" * 70)

# Check if smoothing is masking blinks
short_dips = []
for i in range(len(face_data) - 3):
    ears = [float(face_data[i+j]['ear']) for j in range(4)]

    # Pattern: open -> dip -> open (1 frame dip)
    if ears[0] > 0.25 and ears[1] < 0.22 and ears[2] > 0.25:
        short_dips.append({
            'frame': i,
            'pattern': f'{ears[0]:.3f} -> {ears[1]:.3f} -> {ears[2]:.3f}',
            'detected': face_data[i+1]['eye_state'] == 'closed'
        })

print(f"\nShort 1-frame EAR dips found: {len(short_dips)}")
if short_dips:
    detected = sum(1 for d in short_dips if d['detected'])
    missed = len(short_dips) - detected
    print(f"  Detected as CLOSED: {detected}")
    print(f"  Missed (stayed OPEN): {missed}")

    if missed > 0:
        print(f"\n  PROBLEM: Smoothing may be masking {missed} short blinks!")
        print("\n  Examples of missed short dips:")
        missed_examples = [d for d in short_dips if not d['detected']][:3]
        for i, ex in enumerate(missed_examples, 1):
            print(f"    {i}. Frame {ex['frame']}: {ex['pattern']}")

print("\n" + "=" * 70)
