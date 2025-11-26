# Image Localization Pipeline

This repository provides a simplified demo for the seq + wire retrieval workflow used in indoor localization.

## Getting the Latest Changes
If you cloned this repository earlier, run the following to sync with the latest updates (including the streamlined `demo.py`):

```bash
git pull origin work
```

If you are on a different branch, switch to `work` first:

```bash
git checkout work
git pull
```

## Demo Command
The demo expects precomputed graphs and wireframe outputs. Replace the placeholder paths with your own data roots:

```bash
python demo.py \
  --query_json /path/to/real_graph.json \
  --candidate_json /path/to/synthetic_graphs.json \
  --query_root /path/to/real/rgb/ \
  --candidate_root /path/to/synthetic/rgb/ \
  --wire_query_json /path/to/real_wire.json \
  --wire_candidate_json /path/to/synthetic_wire.json \
  --keywords Room_1 \
  --output_json results_rank.json \
  --coarse_output results_coarse.json
```

Key outputs:
- `output_json`: fused ranking with seq + wire scores.
- `coarse_output`: MI-style pairs for downstream PnP/ICP localization.
- Visualization collages are stored under the `--out_root` directory (defaults to `./results`).
