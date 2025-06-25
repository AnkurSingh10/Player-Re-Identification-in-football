# ⚽Football Player Re-Identification (Cross-Camera Mapping)

This project implements a cross-camera player tracking system using a fine-tuned YOLOv8+ model, ResNet50 embeddings, and DeepSORT.

## 📁 Files
- `main.py`: Runs the entire pipeline

## 🔧 Setup Instructions

### 1. Clone the repository or copy files into a working directory

### 2. Install Dependencies
```bash
pip install -r Requirements.txt
```

Contents of `Requirements.txt`:
```
torch
opencv-python
tensorflow
keras
ultralytics
scikit-learn
deep_sort_realtime
pickle
```

### 3. Place Videos and YOLO Model
- Place the following in your working directory:
  - `broadcast.mp4`
  - `tacticam.mp4`
  - `best.pt` (YOLOv8 model file)

### 4. Run the Code
```bash
python main.py
```

This will:
- Detect and track players in both videos
- Extract ResNet50 embeddings
- Match players across cameras using cosine similarity
- Save final outputs to `/output/`

### 5. Output Files
- `realtime_embeddings_broadcast.pkl`
- `realtime_embeddings_tacticam.pkl`
- `final_mapped_players.pkl`

These contain per-player metadata and matched `global_id`s.
