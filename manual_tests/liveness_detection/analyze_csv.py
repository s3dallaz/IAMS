"""Quick CSV analysis to diagnose blink detection issues."""
import csv
import statistics

csv_path = r'E:\Projects\Graduation Project AI\manual_tests\liveness_detection\ear_diagnostic_data.csv'

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# Filter face-detected frames
face_data = [row for row in data if row['face_detected'] == 'True']

# Track blink events
prev_blink_count = 0
blink_events = []
prev_state = None

for i, row in enumerate(face_data):
    state = row['eye_state']
    blink_count = int(row['blink_count'])
    ear = float(row['ear'])

    if blink_count > prev_blink_count:
        blink_events.append({
            'frame': i,
            'ear': ear,
            'from': prev_state,
            'to': state
        })
        prev_blink_count = blink_count

    prev_state = state

# Analyze closed eye sequences
closed_sequences = []
current_seq = []

for row in face_data:
    if row['eye_state'] == 'closed':
        current_seq.append(float(row['ear']))
    elif current_seq:
        closed_sequences.append(current_seq)
        current_seq = []

print("=" * 60)
print("BLINK DETECTION ANALYSIS")
print("=" * 60)
print(f"\nTotal frames with face: {len(face_data)}")
print(f"Total blinks detected: {len(blink_events)}")
print(f"Closed eye sequences: {len(closed_sequences)}")

if closed_sequences:
    seq_lengths = [len(seq) for seq in closed_sequences]
    print(f"\nClosed sequence stats:")
    print(f"  Min length: {min(seq_lengths)} frames")
    print(f"  Max length: {max(seq_lengths)} frames")
    print(f"  Avg length: {statistics.mean(seq_lengths):.1f} frames")

    print(f"\n⚠ ISSUE FOUND: {len(closed_sequences)} closed sequences but only {len(blink_events)} blinks counted")
    print(f"  Missing blinks: {len(closed_sequences) - len(blink_events)}")

print("\nEAR ranges:")
ear_values = [float(row['ear']) for row in face_data]
print(f"  Open eyes (typical): {statistics.median([e for e in ear_values if e > 0.25]):.3f}")
print(f"  Closed eyes (typical): {statistics.median([float(row['ear']) for row in face_data if row['eye_state'] == 'closed']):.3f}")

print("\nFirst 5 blink events:")
for i, event in enumerate(blink_events[:5], 1):
    print(f"  {i}. Frame {event['frame']}: EAR={event['ear']:.3f}, {event['from']} -> {event['to']}")

print("\n" + "=" * 60)
