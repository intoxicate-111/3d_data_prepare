from pathlib import Path
import numpy as np
import trimesh
from mlr.synthetic import SyntheticRenderConfig, generate_synthetic_dataset
from thingi10k50_prep.io_utils import ensure_dir

out_dir = Path('/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/data_prepare/debug_render_output')
ensure_dir(out_dir)

# Use a simple synthetic mesh to confirm the downstream renderer writes files.
mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
render_cfg = SyntheticRenderConfig(num_views=4, width=128, height=128, trajectory='sphere', backend='cpu', normalize_mesh=False)
result = generate_synthetic_dataset(mesh=mesh, out_dir=out_dir, config=render_cfg, source_mesh_path=out_dir / 'source.obj')
print('dataset', result.dataset_path)
print('mesh', result.mesh_path)
print('images', len(result.image_paths))
for p in [out_dir / 'dataset.json', out_dir / 'cameras.json', out_dir / 'mesh.obj', out_dir / 'images', out_dir / 'masks', out_dir / 'depth']:
    print(p, 'exists', p.exists())
