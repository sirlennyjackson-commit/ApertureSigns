import argparse,json
from pathlib import Path
import numpy as np
from dataset_loader import load_zip_dataset

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--zip",required=True)
    p.add_argument("--out",default="data/processed/sign_dataset.npz")
    p.add_argument("--min-examples",type=int,default=5)
    p.add_argument("--sequence-length",type=int,default=30)
    a=p.parse_args()
    records,counts=load_zip_dataset(a.zip,a.min_examples,a.sequence_length)
    if not records: raise SystemExit("No eligible sequences found.")
    labels=sorted({r["label"] for r in records})
    ids={v:i for i,v in enumerate(labels)}
    X=np.stack([r["sequence"] for r in records]).astype(np.float32)
    y=np.asarray([ids[r["label"]] for r in records],dtype=np.int64)
    groups=np.asarray([r["group"] for r in records])
    sources=np.asarray([r["source"] for r in records])
    frames=np.asarray([r["frames"] for r in records],dtype=np.int32)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,X=X,y=y,groups=groups,sources=sources,frames=frames,labels=np.asarray(labels))
    meta={"source_zip":a.zip,"sequence_length":a.sequence_length,"features_per_frame":225,
          "min_examples":a.min_examples,"total_sequences":len(records),"classes":len(labels),
          "labels":labels,"all_label_counts":counts}
    out.with_suffix(".json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(f"Prepared: {out}")
    print(f"Sequences: {len(records)}")
    print(f"Classes: {len(labels)}")
    print(f"Shape: {X.shape}")

if __name__=="__main__": main()
