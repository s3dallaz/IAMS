# Manual Face Embedding Testing

This directory contains a standalone testing setup for the face embedding system. It generates 128-dimensional face embeddings from images using the FaceDetector and FaceEmbedder.

## Directory Structure

```
manual_tests/face_embedding/
├── README.md                    # This file
├── test_embedding.py            # Testing script
└── images/                      # Test images (must contain single faces)
    ├── person1_photo1.jpg
    ├── person1_photo2.jpg
    ├── person2_photo1.jpg
    └── ...
```

## How to Use

### 1. Prepare Test Images

Place test images in the `images/` folder:

- Each image should contain **exactly one face**
- Images with no faces or multiple faces will be skipped
- Use clear, well-lit photos for best results
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`, `.webp`

**Tip:** To test embedding similarity, add multiple photos of the same person with different names (e.g., `alice1.jpg`, `alice2.jpg`).

### 2. Run the Test

From the project root directory:

```bash
.venv\Scripts\python.exe manual_tests\face_embedding\test_embedding.py
```

Or on Unix/macOS:
```bash
.venv/bin/python manual_tests/face_embedding/test_embedding.py
```

### 3. Interpret Results

The script will:

1. **Process each image** and attempt to generate an embedding
2. **Report success or failure** with detailed status messages
3. **Verify embedding dimensions** (must be 128)
4. **Compare all embeddings** and show Euclidean distances

## Example Output

```
======================================================================
FACE EMBEDDING TEST
======================================================================

This script generates 128-dimensional face embeddings from images.
Images must contain exactly one face.

Images directory: E:\Projects\Graduation Project AI\manual_tests\face_embedding\images

[INFO] Found 4 image(s)

[INFO] Initializing FaceDetector and FaceEmbedder...
[OK] Components initialized

======================================================================
PROCESSING IMAGES
======================================================================

alice1.jpg
----------
[OK] Success (confidence: 0.952)
     Embedding shape: (128,)
     Embedding dtype: float64

alice2.jpg
----------
[OK] Success (confidence: 0.918)
     Embedding shape: (128,)
     Embedding dtype: float64

bob1.jpg
--------
[OK] Success (confidence: 0.934)
     Embedding shape: (128,)
     Embedding dtype: float64

group_photo.jpg
---------------
[FAILED] Multiple faces detected (3)

======================================================================
SUMMARY
======================================================================

Total images: 4
Successful embeddings: 3
Failed: 1

======================================================================
EMBEDDING COMPARISONS
======================================================================

Euclidean distances between embeddings:
(Lower distance = more similar faces)

alice1.jpg <-> alice2.jpg
  Distance: 0.3842

alice1.jpg <-> bob1.jpg
  Distance: 0.8921

alice2.jpg <-> bob1.jpg
  Distance: 0.9134

======================================================================
TESTING COMPLETE
======================================================================

[OK] Resources released
```

## Understanding the Output

### Success Messages

**For each image:**
- `[OK] Success` - Embedding generated successfully
- `Embedding shape: (128,)` - Confirms 128-dimensional vector
- `Embedding dtype: float64` - Data type of the embedding
- `confidence: 0.952` - Face detection confidence (0.0 to 1.0)

### Failure Messages

**Common failures:**
- `No face detected` - Image doesn't contain a visible face
- `Multiple faces detected (N)` - Image has more than one face
- `Failed to load image` - Invalid or corrupted file
- `Invalid face crop` - Face region extraction failed
- `Could not generate embedding` - Embedding generation failed

### Embedding Comparisons

**Euclidean distance interpretation:**
- **< 0.6**: Same person (typical threshold for recognition)
- **0.6 - 0.8**: Possibly same person or similar appearance
- **> 0.8**: Different people

Lower distance means faces are more similar. Photos of the same person should have distance < 0.6.

## What the Script Does

1. **Loads all images** from `images/` folder
2. **Detects faces** using MediaPipe FaceDetector
3. **Validates** exactly one face per image
4. **Extracts face region** using bounding box
5. **Generates embedding** using face_recognition library (dlib)
6. **Verifies dimensions** are exactly 128
7. **Compares all embeddings** pairwise using Euclidean distance

## Technical Details

**Face Detection:**
- Uses MediaPipe BlazeFace model
- Default confidence threshold: 0.7 (from `config.py`)

**Face Embedding:**
- Uses face_recognition library (dlib's ResNet model)
- Generates 128-dimensional vectors
- Encoding represents unique facial features

**Distance Metric:**
- Euclidean distance: `sqrt(sum((a - b)^2))`
- Standard metric for face recognition
- Recognition threshold typically 0.6

## Error Handling

The script handles various error conditions:

- **Invalid images**: Skips and reports
- **No faces**: Reports and continues
- **Multiple faces**: Reports count and skips
- **Corrupted files**: Handles gracefully
- **Embedding failures**: Reports specific error

## Testing Recommendations

**Good test cases:**
1. Multiple photos of the same person (should have low distance)
2. Photos of different people (should have high distance)
3. Images with no faces (should fail gracefully)
4. Images with multiple faces (should report and skip)
5. Various image qualities and lighting conditions

**Expected results:**
- Same person: distance 0.3 - 0.5
- Different people: distance 0.8 - 1.2
- Similar appearance: distance 0.6 - 0.8

## Notes

- This test is completely separate from the main application
- No data is written to the database
- Uses the same components as the production system
- Embeddings are generated in memory and not saved
- All comparisons are for observation only

## Troubleshooting

**"No face detected" on valid face images**
- Face may be too small in the image
- Try closer/larger face photos
- Ensure good lighting
- Check face is front-facing

**"Could not generate embedding"**
- Face region may be too small
- Try higher resolution images
- Ensure face is clearly visible
- Check face detection confidence

**"Failed to load image"**
- Check file format is supported
- Verify file is not corrupted
- Try re-saving the image
- Check file permissions

**All embeddings have high distance**
- This is normal for different people
- Same person photos should have distance < 0.6
- Similar-looking people may have distance 0.6 - 0.8

**Components fail to initialize**
- Check MediaPipe model file exists (`blaze_face_short_range.tflite`)
- Verify face_recognition library is installed
- Check virtual environment is activated
