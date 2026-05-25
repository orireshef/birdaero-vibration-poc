import json
import numpy as np
import urllib.request
from vibration_poc.engine import load_fno_model, evaluate_design
from vibration_poc.dataset.config import GridNormStats
from vibration_poc.dataset.preprocess import load_meta, _iter_graphs

model = load_fno_model(best_path, hidden_dim=FNO_HIDDEN_DIM, num_layers=FNO_NUM_LAYERS, modes=FNO_MODES, device=device)

stats_path = dataset_config.processed_dir / "grid" / "grid_norm_stats.json"
with open(stats_path) as f:
    norm_stats = GridNormStats(**json.load(f))

test_tfrecord = dataset_config.raw_dir / "test.tfrecord"
if not test_tfrecord.exists():
    urllib.request.urlretrieve(dataset_config.base_url + "/test.tfrecord", test_tfrecord)

meta = load_meta(dataset_config.raw_dir / "meta.json")
graphs = list(_iter_graphs(test_tfrecord, meta, max_trajectories=1))
test_graph = graphs[min(100, len(graphs) - 1)]

metrics, results = evaluate_design(test_graph, model, norm_stats, num_steps=50, bc_node_types=[1, 2, 3], grid_config=grid_config)

gt_disp = test_graph["y"].numpy()
gt_mag = np.linalg.norm(gt_disp, axis=1)
gt_max = gt_mag.max() * 1000
gt_mean = gt_mag.mean() * 1000
pred_max = metrics.max_displacement * 1000
pred_mean = metrics.mean_displacement * 1000
ratio = metrics.max_displacement / max(gt_mag.max(), 1e-10)

print("Ground truth:  max=%.3f mm, mean=%.3f mm" % (gt_max, gt_mean))
print("Prediction:    max=%.3f mm, mean=%.3f mm" % (pred_max, pred_mean))
print("Pred/GT ratio: %.2fx" % ratio)
