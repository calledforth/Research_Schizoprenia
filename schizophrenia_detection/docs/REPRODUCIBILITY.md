# Reproducibility Guide

This guide covers practices for ensuring reproducible results in schizophrenia detection analysis.

## Random Seeds

Set seeds for all frameworks:

```python
import numpy as np
import tensorflow as tf
import random
import os

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'
```

## Deterministic Operations

Enable deterministic TensorFlow operations:

```python
# TensorFlow deterministic
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# For CUDA
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
```

## Environment Management

### Exact Version Pinning

Create `requirements-exact.txt`:

```bash
pip freeze > requirements-exact.txt
```

### Docker Container

```dockerfile
FROM tensorflow/tensorflow:2.12.0-gpu

COPY requirements-exact.txt .
RUN pip install -r requirements-exact.txt

COPY . /app
WORKDIR /app
```

## Configuration Management

Save exact configuration:

```python
from schizophrenia_detection.config import Config

config = Config()
config.save('experiment_config.json')
```

Load configuration:

```python
config = Config.load('experiment_config.json')
```

## Cross-Validation Protocol

Document CV strategy:

```python
# Stratified 5-fold CV with fixed splits
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Process fold
    pass
```

## Data Versioning

Track data versions:

```python
# Log data hash
import hashlib

def get_data_hash(data_path):
    with open(data_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

data_hash = get_data_hash('data/fmri/sub-001_func.nii.gz')
print(f"Data hash: {data_hash}")
```

## Results Archiving

Archive complete experiment:

```python
import json
import shutil
from datetime import datetime

experiment_dir = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(experiment_dir)

# Save configuration
config.save(f"{experiment_dir}/config.json")

# Save results
with open(f"{experiment_dir}/results.json", 'w') as f:
    json.dump(results, f, indent=2)

# Save model
model.save(f"{experiment_dir}/model.h5")
```

## Logging

Comprehensive logging:

```python
from schizophrenia_detection.utils.logging_utils import setup_logger

logger = setup_logger('experiment', 'experiment.log')

logger.info(f"Configuration: {config.to_dict()}")
logger.info(f"Data hash: {data_hash}")
logger.info(f"Results: {results}")
```

## Best Practices

- Fix random seeds before any random operations
- Use deterministic operations when possible
- Save exact configuration and environment details
- Document data preprocessing steps
- Archive complete experiments
- Use version control for all code
- Share exact reproduction instructions

## Verification

Verify reproducibility:

```python
# Run same experiment twice
results1 = run_experiment(config)
results2 = run_experiment(config)

# Check if results match
assert np.allclose(results1['accuracy'], results2['accuracy'], rtol=1e-5)
```

## Next Steps

1. Implement reproducibility practices in your workflow
2. Archive experiments systematically
3. Share reproduction instructions with collaborators