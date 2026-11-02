import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.metrics import accuracy_score,classification_report,top_k_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/processed/sign_dataset.npz")
    p.add_argument("--out-dir",default="models/sign/classifier")
    a=p.parse_args()
    d=np.load(a.data,allow_pickle=True)
    X=d["X"].reshape(len(d["X"]),-1); y=d["y"].astype(int); groups=d["groups"]; labels=d["labels"].tolist()
    splitter=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=42)
    tr,te=next(splitter.split(X,y,groups=groups))
    train_classes=set(y[tr]); te=np.asarray([i for i in te if y[i] in train_classes])
    if len(te)==0: raise SystemExit("No evaluable test samples after group split.")
    model=Pipeline([("scale",StandardScaler()),("svc",SVC(kernel="rbf",C=8,gamma="scale",
        probability=True,class_weight="balanced",random_state=42))])
    print(f"Training sequences: {len(tr)}")
    print(f"Test sequences: {len(te)}")
    print(f"Classes: {len(labels)}")
    print("Training SVM...")
    model.fit(X[tr],y[tr])
    pred=model.predict(X[te]); proba=model.predict_proba(X[te])
    acc=accuracy_score(y[te],pred)
    top3=top_k_accuracy_score(y[te],proba,k=min(3,len(labels)),labels=np.arange(len(labels)))
    print(f"TOP-1 ACCURACY: {acc:.4f}")
    print(f"TOP-3 ACCURACY: {top3:.4f}")
    print(classification_report(y[te],pred,labels=sorted(train_classes),
          target_names=[labels[i] for i in sorted(train_classes)],zero_division=0))
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,out/"aperture_sign_classifier.joblib")
    meta={"model_type":"SVC-RBF","sequence_length":int(d["X"].shape[1]),
          "features_per_frame":int(d["X"].shape[2]),"flattened_features":int(X.shape[1]),
          "classes":labels,"train_sequences":int(len(tr)),"test_sequences":int(len(te)),
          "top1_accuracy":float(acc),"top3_accuracy":float(top3),
          "confidence_threshold_recommended":.65,"random_state":42}
    (out/"model_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(f"MODEL SAVED: {out/'aperture_sign_classifier.joblib'}")

if __name__=="__main__": main()
