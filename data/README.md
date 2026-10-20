# data/

`data/sequences/<label>/<signer>__<timestamp>.npz` - landmark clips written by
`python -m backend.training.collect`. Git-ignored on purpose: it is personal
movement data. Only landmarks are stored, never video. Get consent first.

Each `.npz` holds `left (T,21,3)`, `right (T,21,3)`, `pose (T,33,4)`, `face (T,3)`;
undetected body parts are NaN.
