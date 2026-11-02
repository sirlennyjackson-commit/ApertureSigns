import os, zipfile
import numpy as np

FEATURES_PER_FRAME = 225
DEFAULT_SEQUENCE_LENGTH = 30

def _xyz(p):
    if isinstance(p, dict):
        return float(p["x"]), float(p["y"]), float(p["z"])
    return float(p[0]), float(p[1]), float(p[2])

def hand_features(hand):
    if not hand:
        return np.zeros(63, dtype=np.float32)
    wx, wy, wz = _xyz(hand[0])
    out = []
    for p in hand[:21]:
        x,y,z = _xyz(p)
        out.extend((x-wx, y-wy, z-wz))
    return np.asarray((out + [0.0]*63)[:63], dtype=np.float32)

def pose_features(pose):
    if not pose:
        return np.zeros(99, dtype=np.float32)
    out=[]
    for p in pose[:33]:
        out.extend(_xyz(p))
    return np.asarray((out + [0.0]*99)[:99], dtype=np.float32)

def frame_features(frame):
    return np.concatenate([
        hand_features(frame.get("left_hand", [])),
        hand_features(frame.get("right_hand", [])),
        pose_features(frame.get("pose", []))
    ]).astype(np.float32)

def sequence_features(frames, sequence_length=30):
    if not frames:
        return np.zeros((sequence_length,225), dtype=np.float32)
    x=np.asarray([frame_features(f) for f in frames], dtype=np.float32)
    if len(x)==sequence_length:
        return x
    old=np.linspace(0,1,len(x))
    new=np.linspace(0,1,sequence_length)
    out=np.empty((sequence_length,225),dtype=np.float32)
    for j in range(225):
        out[:,j]=np.interp(new,old,x[:,j])
    return out

def load_zip_dataset(zip_path,min_examples=5,sequence_length=30):
    records=[]; counts={}
    with zipfile.ZipFile(zip_path) as z:
        names=[n for n in z.namelist() if "/Landmarks/" in n and n.lower().endswith(".npy")]
        for name in names:
            label=os.path.basename(name)[:-4].strip().upper()
            if not label: continue
            try:
                arr=np.load(z.open(name),allow_pickle=True)
            except Exception:
                continue
            if not isinstance(arr,np.ndarray) or len(arr)<5:
                continue
            counts[label]=counts.get(label,0)+1
            parts=name.split("/")
            group=f"{parts[1]}_{parts[2]}" if len(parts)>=3 else "UNKNOWN"
            records.append({
                "label":label,
                "group":group,
                "sequence":sequence_features(arr.tolist(),sequence_length),
                "source":name,
                "frames":len(arr)
            })
    eligible={k for k,v in counts.items() if v>=min_examples}
    return [r for r in records if r["label"] in eligible], counts
