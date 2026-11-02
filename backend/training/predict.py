import argparse,json
from pathlib import Path
import joblib,numpy as np
from dataset_loader import sequence_features

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="models/sign/classifier/aperture_sign_classifier.joblib")
    p.add_argument("--metadata",default="models/sign/classifier/model_metadata.json")
    p.add_argument("--sequence",required=True)
    p.add_argument("--threshold",type=float,default=None)
    a=p.parse_args()
    model=joblib.load(a.model); meta=json.loads(Path(a.metadata).read_text(encoding="utf-8"))
    arr=np.load(a.sequence,allow_pickle=True)
    x=sequence_features(arr.tolist(),meta["sequence_length"]).reshape(1,-1)
    prob=model.predict_proba(x)[0]; order=np.argsort(prob)[::-1][:5]
    threshold=meta.get("confidence_threshold_recommended",.65) if a.threshold is None else a.threshold
    for i in order: print(f"{meta['classes'][i]:20s} {prob[i]*100:6.2f}%")
    best=int(order[0])
    print("PREDICTION:",meta["classes"][best] if prob[best]>=threshold else "UNKNOWN / LOW CONFIDENCE")

if __name__=="__main__": main()
