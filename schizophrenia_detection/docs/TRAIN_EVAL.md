# Training and Evaluation Guide

This guide covers training and evaluation workflows for the schizophrenia detection model using notebooks and programmatic interfaces.

## Training

### Notebook Training

Use [model_training.ipynb](../notebooks/model_training.ipynb:1) for interactive training:

1. Load preprocessed data
2. Configure model parameters
3. Train with cross-validation
4. Monitor progress with TensorBoard

### Programmatic Training

```python
from schizophrenia_detection.training import trainer
from schizophrenia_detection.models.sspnet_3d_cnn import SSPNet3DCNN
from schizophrenia_detection.config import Config

# Setup
config = Config()
model = SSPNet3DCNN(config.model)

# Train
history = trainer.train_model(
    model=model,
    train_data=train_generator,
    val_data=val_generator,
    config=config
)
```

### Cross-Validation

```python
from schizophrenia_detection.training.cross_validation import cross_validate

# Perform 5-fold cross-validation
cv_results = cross_validate(
    model_class=SSPNet3DCNN,
    data=data,
    labels=labels,
    cv_folds=5,
    config=config
)
```

## Evaluation

### Metrics

- **Accuracy**: Overall classification accuracy
- **Sensitivity**: True positive rate (schizophrenia detection)
- **Specificity**: True negative rate (control detection)
- **ROC/AUC**: Receiver operating characteristic
- **Confusion Matrix**: Detailed classification results

### Evaluation Code

```python
from schizophrenia_detection.training.evaluator import evaluate_model

# Evaluate model
results = evaluate_model(
    model=model,
    test_data=test_generator,
    config=config
)

print(f"Accuracy: {results['accuracy']:.3f}")
print(f"Sensitivity: {results['sensitivity']:.3f}")
print(f"Specificity: {results['specificity']:.3f}")
print(f"AUC: {results['auc']:.3f}")
```

### Results Analysis

Use [results_analysis.ipynb](../notebooks/results_analysis.ipynb:1) for:
- Performance visualization
- Confusion matrix plots
- ROC curves
- Error analysis

## Configuration

### Training Parameters

```python
# In config.py
config.training.epochs = 100
config.training.learning_rate = 0.001
config.training.batch_size = 4
config.training.early_stopping = True
config.training.patience = 20
```

### Evaluation Parameters

```python
config.evaluation.cv_folds = 5
config.evaluation.classification_threshold = 0.5
config.evaluation.results_dir = "./results"
```

## Best Practices

- Use cross-validation for robust performance estimates
- Monitor training/validation loss to detect overfitting
- Save checkpoints regularly
- Use mixed precision training for GPU efficiency
- Evaluate on held-out test set only once

## Outputs

Training and evaluation generate:
- Model checkpoints in `./checkpoints/`
- Metrics in `./results/`
- Training plots in `./visualizations/`
- Logs in `./logs/`

## Next Steps

1. Run [model_training.ipynb](../notebooks/model_training.ipynb:1)
2. Analyze results with [results_analysis.ipynb](../notebooks/results_analysis.ipynb:1)
3. Visualize using [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md:1)