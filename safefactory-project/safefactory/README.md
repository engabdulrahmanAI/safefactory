# SafeFactory

An AI project that predicts equipment failure from the data the equipment produces (sensor readings such as temperature, rotational speed, torque, tool wear, vibration and pressure). It estimates whether a machine will fail, predicts the failure type, scores equipment risk, prioritizes maintenance, and explains the result in plain language through a retrieval-based assistant (TF-IDF retrieval + GPT-4o-mini).

## Components

| Component | Task | Model | Data |
|---|---|---|---|
| `01_binary_failure` | Machine failure (binary) | XGBoost (RandomizedSearchCV) | AI4I 2020 |
| `02_multiclass_failure` | Failure type: HDF / OSF / PWF / TWF | XGBoost (`multi:softprob`) | AI4I 2020 failure rows |
| `03_risk_score` | Equipment risk score (regression) | RandomForestRegressor | Distribution equipment risk dataset |
| `04_maintenance_priority` | Maintenance priority 1/2/3 | RandomForestClassifier (`class_weight='balanced'`) | Smart manufacturing maintenance dataset |
| `rag.py` | Conversational assistant | Binary + multiclass models, TF-IDF retriever, GPT-4o-mini | `data/failure_knowledge_base.csv` |

## Results

All figures come from the outputs saved in the notebooks (single 80/20 split, `random_state=42`).

### Binary failure (AI4I 2020, 10,000 rows, 339 failures; test set 2,000 rows / 68 failures)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| XGBoost (default) | 0.9865 | 0.887 | 0.691 | 0.777 |
| RandomForest | 0.984 | 0.875 | 0.618 | 0.724 |
| BalancedRandomForest | 0.9205 | 0.288 | 0.912 | 0.438 |
| XGBoost tuned (saved model, threshold 0.5) | 0.984 | 0.750 | 0.794 | 0.771 |

Tuned hyperparameters: `n_estimators=873`, `max_depth=5`, `learning_rate=0.07`, `subsample=0.8`, `colsample_bytree=0.9`, `min_child_weight=3`, `gamma=0`, `scale_pos_weight≈35.65`. Best cross-validation F1: 0.7397.

### Failure type (338 failure rows, RNF dropped, test set 66 rows)

XGBoost: accuracy 0.894, macro F1 0.88. Weakest class: OSF (recall 0.69).

### Risk score (1,000 rows, test set 200 rows)

RandomForestRegressor: MAE 0.0220, RMSE 0.0284, R² 0.9585.

### Maintenance priority (1,430 rows, test set 286 rows)

RandomForestClassifier: accuracy 0.99, macro F1 0.97. Class 3 (74 samples overall): recall 0.87.

## Limitations

- Test sets are small (66 to 286 rows), so results have high variance.
- In the binary notebook the threshold sweep was evaluated on the test set; there is no separate validation set.
- The risk and maintenance-priority datasets look synthetic; their high scores should not be read as real-world performance.
- `Model/maintenance_priority_model.pkl` includes a `Risk_Score` feature that is not in the feature list of the notebook cell, so the saved model was trained with a different version of that code.
- The notebooks do not contain the code that saved the `.pkl` files.
- `rag.py` requires an OpenAI API key to extract sensor readings from free text.

## Data sources

- AI4I 2020: `kagglehub.dataset_download("stephanmatzka/predictive-maintenance-dataset-ai4i-2020")`
- Smart manufacturing maintenance: `kagglehub.dataset_download("ziya07/smart-manufacturing-maintenance-dataset")`
- Equipment risk dataset: source not recorded; the notebook expects `distribution_equipment_risk_dataset.csv` to be uploaded.
- Failure costs in the knowledge base follow Rayarao & Rayarao (2024), as cited in the notebooks.

Raw datasets are not committed. Download them with the commands above.

## Run the assistant

```
pip install -r requirements.txt
cp .env.example .env
python rag.py
```

Set `OPENAI_API_KEY` in `.env`. Type a question with sensor readings (machine type, air temperature, process temperature, rotational speed, torque, tool wear). Type `r` to clear the stored readings and `q` to quit.

## Structure

```
rag.py
Model/                  saved models (.pkl)
data/                   failure knowledge base
notebooks/              training and evaluation notebooks
experiments/            early failure-type experiment (no saved outputs)
```

Notebook origin: `01_binary_failure` = `binary_Faluir.ipynb`, `02_multiclass_failure` = `MClass_Faliur.ipynb`, `03_risk_score` = `Risk_Level.ipynb`, `04_maintenance_priority` = `Smart_Manufacturing_Maintenance.ipynb`.
