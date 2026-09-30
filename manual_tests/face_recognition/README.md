# Manual Face Recognition Testing

This directory contains a standalone testing setup for the face recognition system. It uses real images from the filesystem and is completely separate from the production database.

## Directory Structure

```
manual_tests/face_recognition/
├── README.md                    # This file
├── test_recognition.py          # Testing script
├── known/                       # Known faces (training data)
│   ├── student_001/            # Student 1 images
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── student_002/            # Student 2 images
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
└── unknown/                     # Unknown faces (test data)
    ├── test1.jpg
    ├── test2.jpg
    └── ...
```

## How to Use

### 1. Prepare Known Faces (Training Data)

Place images of known students in subdirectories under `known/`:

- Each subdirectory represents one student
- Name subdirectories with student IDs (e.g., `student_001`, `student_002`, `alice`, `bob`)
- Place one or more face images per student in their subdirectory
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`, `.webp`

**Example:**
```
known/
├── student_001/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
└── student_002/
    ├── face1.jpg
    └── face2.jpg
```

**Tips for best results:**
- Use clear, well-lit photos with visible faces
- Multiple images per student improve accuracy
- Images should contain only one face (or have the target face prominent)
- Front-facing photos work best

### 2. Prepare Unknown Faces (Test Data)

Place test images directly in the `unknown/` directory (not in subdirectories):

```
unknown/
├── test_person1.jpg
├── classroom_photo.jpg
└── unknown_face.png
```

These images will be tested against all known faces to see if they match.

### 3. Run the Test

From the project root directory, run:

```bash
.venv\Scripts\python.exe manual_tests\face_recognition\test_recognition.py
```

Or on Unix/macOS:
```bash
.venv/bin/python manual_tests/face_recognition/test_recognition.py
```

### 4. Interpret Results

The script will:

1. **Load Known Faces**: Process all images in `known/` subdirectories and generate embeddings
2. **Test Unknown Faces**: Process each image in `unknown/` and attempt recognition

**Output for each test image:**
- `✅ Recognized as: student_001` - Face matched a known student
- `❓ Unknown (no match found)` - Face did not match any known student
- `📏 Distance: 0.4523` - Recognition confidence (lower = better match)

**Distance Threshold:** 0.6 (configurable in the script)
- Distance ≤ 0.6: Recognized
- Distance > 0.6: Unknown

## Example Output

```
======================================================================
LOADING KNOWN FACES
======================================================================

📁 Processing student_001...
   ✅ photo1.jpg: Embedding generated (conf: 0.952)
   ✅ photo2.jpg: Embedding generated (conf: 0.891)
   ✅ Loaded 2 embedding(s) for student_001

📁 Processing student_002...
   ✅ face1.jpg: Embedding generated (conf: 0.924)
   ✅ Loaded 1 embedding(s) for student_002

📊 Summary: Loaded 2 student(s) with embeddings

======================================================================
TESTING UNKNOWN FACES
======================================================================

Testing 3 image(s)...

──────────────────────────────────────────────────────────────────────
🖼️  Testing: test1.jpg
──────────────────────────────────────────────────────────────────────
👤 Detected 1 face(s)

   Face #1:
   └─ Bounding box: (120, 45, 180, 220)
   └─ Confidence: 0.956
   └─ ✅ Recognized as: student_001
   └─ 📏 Distance: 0.4523

──────────────────────────────────────────────────────────────────────
🖼️  Testing: unknown_person.jpg
──────────────────────────────────────────────────────────────────────
👤 Detected 1 face(s)

   Face #1:
   └─ Bounding box: (95, 30, 150, 185)
   └─ Confidence: 0.912
   └─ ❓ Unknown (no match found)
   └─ 📏 Best distance: 0.7834
```

## Error Handling

The script handles various error conditions gracefully:

- **Invalid image formats**: Skips and reports
- **No faces detected**: Reports and continues
- **Multiple faces**: Uses the first detected face and warns
- **Corrupted images**: Skips and reports error
- **Missing faces in crops**: Reports embedding failure

## Adjusting Recognition Threshold

To change the recognition sensitivity, edit the `threshold` parameter in `test_recognition.py`:

```python
def _find_best_match(
    self, 
    embedding: np.ndarray, 
    threshold: float = 0.6  # Change this value
) -> Tuple[Optional[str], float]:
```

- **Lower threshold (e.g., 0.4)**: Stricter matching, fewer false positives
- **Higher threshold (e.g., 0.8)**: More lenient matching, more false positives

## Notes

- This testing setup is completely separate from the production database
- No data is written to the database
- Test images are not stored or processed by the main application
- The script uses the same `FaceDetector` and `FaceEmbedder` classes as the main application
- Results may vary based on image quality, lighting, and pose

## Troubleshooting

**"No faces detected"**
- Ensure images contain visible faces
- Try images with better lighting
- Use front-facing photos

**"Failed to load image"**
- Check image file is not corrupted
- Verify file extension is supported
- Try re-saving the image

**"Could not generate embedding"**
- Face region may be too small
- Try using higher resolution images
- Ensure face is clearly visible

**Import errors**
- Make sure you're running from the project root
- Activate the virtual environment first
- Verify all dependencies are installed
