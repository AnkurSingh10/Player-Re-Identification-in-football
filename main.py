## Importing libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU for TensorFlow as kaggle shows  version cuDNN version mismatch
#can use PyTorch for resnet and embeddings but i am more comfortable with tf .
from ultralytics import YOLO
import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image as keras_image
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pickle


## Loading Model 

model = YOLO("best.pt")  

results_broadcast = model("broadcast.mp4", stream=True)
results_tacticam = model("tacticam.mp4", stream=True)

## Embeddings

print("Class names:", model.names)

BASE_DIR = "/output"
os.makedirs(BASE_DIR, exist_ok=True)

# Load ResNet50 
resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

def extract_embeddings_from_video(video_path, source_name):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    save_crop_dir = f"{BASE_DIR}/crops_{source_name}"
    save_vid_path = f"{BASE_DIR}/{source_name}_tracked.mp4"
    os.makedirs(save_crop_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_vid = cv2.VideoWriter(save_vid_path, fourcc, fps, (width, height))

    all_embeddings = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        boxes = results[0].boxes

        detections = []
        player_data_list = []

        for i, box in enumerate(boxes):
            cls = int(box.cls.item())
            if cls not in [1, 2]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1
            player_crop_bgr = frame[y1:y2, x1:x2]
            if player_crop_bgr.size == 0:
                continue

            img = cv2.resize(player_crop_bgr, (224, 224))
            img = keras_image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            emb = resnet.predict(img)[0]
            emb = tf.linalg.l2_normalize(emb, axis=0).numpy()

            crop_filename = f"{save_crop_dir}/frame_{frame_idx:04d}_player_{i:02d}.jpg"
            cv2.imwrite(crop_filename, player_crop_bgr)

            detections.append(([x1, y1, w, h], 0.99, emb))
            player_data_list.append({
                "frame": frame_idx,
                "bbox": [x1, y1, x2, y2],
                "embedding": emb,
                "source": source_name,
                "crop_path": crop_filename
            })

        tracks = tracker.update_tracks(detections, frame=frame)
        for track, pdata in zip(tracks, player_data_list):
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = pdata["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            pdata["player_id"] = int(track_id)
            all_embeddings.append(pdata)

        out_vid.write(frame)
        frame_idx += 1

    cap.release()
    out_vid.release()

    out_frame_path = f"{BASE_DIR}/realtime_embeddings_{source_name}.pkl"
    with open(out_frame_path, "wb") as f:
        pickle.dump(all_embeddings, f)

    print(f"✅ Saved DeepSORT tracked embeddings and video for {source_name}")

## loading embeddings
def load_all_embeddings():
    broadcast_path = os.path.join(BASE_DIR, "realtime_embeddings_broadcast.pkl")
    tacticam_path = os.path.join(BASE_DIR, "realtime_embeddings_tacticam.pkl")
    all_data = []
    for path in [broadcast_path, tacticam_path]:
        with open(path, "rb") as f:
            all_data.extend(pickle.load(f))
    return all_data

## merging similar ids
def merge_similar_ids(embeddings, threshold=0.95):
    id_to_embs = defaultdict(list)
    for e in embeddings:
        id_to_embs[e["player_id"]].append(e["embedding"])

    id_to_avg = {pid: np.mean(vecs, axis=0) for pid, vecs in id_to_embs.items()}

    ids = list(id_to_avg.keys())
    id_vecs = np.stack([id_to_avg[pid] for pid in ids])
    sim_matrix = cosine_similarity(id_vecs)

    merged = {}
    cluster = {}
    current_cluster = 0
    for i in range(len(ids)):
        if ids[i] in merged:
            continue
        cluster[current_cluster] = [ids[i]]
        merged[ids[i]] = current_cluster
        for j in range(i+1, len(ids)):
            if sim_matrix[i, j] > threshold:
                merged[ids[j]] = current_cluster
                cluster[current_cluster].append(ids[j])
        current_cluster += 1

    new_data = []
    for e in embeddings:
        old_id = e["player_id"]
        if old_id in merged:
            e["player_id"] = merged[old_id]
        new_data.append(e)

    print(f"Merged into {len(set(merged.values()))} unique player IDs.")
    return new_data


def save_cluster_results(embeddings):
    out_path = os.path.join(BASE_DIR, "player_mapping_with_ids.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f" Saved final player mapping with IDs at: {out_path}")


def overlay_ids_on_video(source_name, player_data):
    input_path = f"{source_name}.mp4"
    output_path = os.path.join(BASE_DIR, f"{source_name}_with_ids.mp4")

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_vid = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_map = {}
    for e in player_data:
        if e["source"] == source_name:
            frame_map.setdefault(e["frame"], []).append(e)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for e in frame_map.get(frame_idx, []):
            x1, y1, x2, y2 = e["bbox"]
            pid = e["player_id"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        out_vid.write(frame)
        frame_idx += 1

    cap.release()
    out_vid.release()
    print(f"🎥 Visualization saved: {output_path}")


extract_embeddings_from_video("broadcast.mp4", "broadcast")
extract_embeddings_from_video("tacticam.mp4", "tacticam")

## Load and save combined player embeddings
combined = load_all_embeddings()
merged_combined = merge_similar_ids(combined)
save_cluster_results(merged_combined)

## Visualize tracked IDs on video
overlay_ids_on_video("broadcast", merged_combined)
overlay_ids_on_video("tacticam", merged_combined)

def get_similarity_score(embed1, embed2):
    return cosine_similarity(embed1.reshape(1, -1), embed2.reshape(1, -1))[0][0]



with open("/output/realtime_embeddings_broadcast.pkl", "rb") as f:
    broadcast_embeds = pickle.load(f)

with open("/output/realtime_embeddings_tacticam.pkl", "rb") as f:
    tacticam_embeds = pickle.load(f)

def cross_camera_id_assignment(broadcast, tacticam, threshold=0.95):
    id_counter = 0
    id_map = {}

    for entry in broadcast:
        emb = entry["embedding"]
        entry["global_id"] = id_counter
        id_map[id_counter] = emb
        id_counter += 1

    for entry in tacticam:
        emb = entry["embedding"]
        best_id = None
        best_sim = -1
        for gid, b_emb in id_map.items():
            sim = cosine_similarity(emb.reshape(1, -1), b_emb.reshape(1, -1))[0][0]
            if sim > best_sim:
                best_sim = sim
                best_id = gid

        if best_sim >= threshold:
            entry["global_id"] = best_id
        else:
            entry["global_id"] = id_counter
            id_map[id_counter] = emb
            id_counter += 1

    return broadcast + tacticam

mapped_detections = cross_camera_id_assignment(broadcast_embeds, tacticam_embeds)

with open(f"{BASE_DIR}/final_mapped_players.pkl", "wb") as f:
    pickle.dump(mapped_detections, f)

print("\nCross-camera player mapping complete. Output saved in:")
print(f"- {BASE_DIR}/final_mapped_players.pkl")
