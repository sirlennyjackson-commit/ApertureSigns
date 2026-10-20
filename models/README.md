# models/

`signs.npz` is produced by `python -m backend.training.train` (git-ignored).
It stores plain arrays (scaler + linear weights + labels + metadata), no pickle,
so loading a model file cannot execute code. Override the location with
`APERTURE_MODEL_PATH`.
