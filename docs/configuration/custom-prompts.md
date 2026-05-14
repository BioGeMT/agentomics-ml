# Custom Prompts

Customize the agent's optimization goal with custom user prompts.

## Default Prompt

Without customization, the agent uses:

> Develop a machine learning model that generalizes well to new unseen data.

## Using Custom Prompts

### Command Line

```bash
./run.sh --user-prompt "Your custom instructions here"
```

### Examples

**Simple models only:**
```bash
./run.sh --user-prompt "Only create simple ML models like logistic regression and shallow decision trees"
```

**Focus on interpretability:**
```bash
./run.sh --user-prompt "Prioritize model interpretability over performance. Use models where feature importance can be easily explained."
```

**Specific model type:**
```bash
./run.sh --user-prompt "Use gradient boosting models like XGBoost or LightGBM"
```

**Handle imbalanced data:**
```bash
./run.sh --user-prompt "The dataset is highly imbalanced. Use appropriate techniques like SMOTE, class weights, or focal loss."
```

**Neural networks:**
```bash
./run.sh --user-prompt "Focus on deep learning approaches. Design custom neural network architectures."
```

**Quick iterations:**
```bash
./run.sh --user-prompt "Keep models simple and training fast. Avoid complex architectures that take long to train."
```

## What Custom Prompts Affect

The user prompt influences the agentic steps:

| Step | How It's Used |
|------|---------------|
| Iteration Planning | Cross-step guidance for the current iteration |
| Data Exploration | What to look for in the data |
| Data Split | Split strategy considerations |
| Data Representation | Feature encoding choices |
| Model Architecture | Model selection and design |
| Model Training | Training approach and hyperparameters |
| Model Inference | Prediction pipeline design |

Validation evaluation itself is deterministic: it runs the generated inference script on train/validation data and scores the configured `--val-metric`.

## Prompt Tips

### Be Specific

Instead of:
> Make a good model

Use:
> Create a random forest model with feature selection. Focus on the top 50 most important features.

### Include Constraints

> Maximum training time should be 30 minutes. Model size should be under 100MB for deployment.

### Mention Domain Knowledge

> This is gene expression data. Consider using models that handle high-dimensional sparse data well.

### Specify Metrics

> Optimize for AUROC rather than accuracy, as the classes are imbalanced.

## Combining with Other Options

Custom prompts work with all other options:

```bash
./run.sh \
  --user-prompt "Use only sklearn models, no neural networks" \
  --model openai/gpt-4 \
  --dataset my_data \
  --iterations 15 \
  --val-metric AUROC
```

## Limitations

Custom prompts guide the agent but don't guarantee specific outcomes:

- The agent may still try different approaches
- Very restrictive prompts may limit performance
- Some requests may not be feasible for certain datasets

## Dataset Description vs User Prompt

| Dataset Description | User Prompt |
|---------------------|-------------|
| Domain information about the data | Instructions for the agent |
| Goes in `dataset_description.md` | Passed via `--user-prompt` |
| Describes what the data is | Describes what to do |

**Example dataset_description.md:**
> This dataset contains RNA-seq expression levels from tumor samples. Features are gene expression values.

**Example user prompt:**
> Focus on gene signature discovery. Use feature selection to identify the most predictive genes.

Both can be used together - they complement each other.

## Advanced: Prompt Engineering

For complex requirements, structure your prompt:

```bash
./run.sh --user-prompt "
GOAL: Create an interpretable model for clinical use.
CONSTRAINTS:
- Must use logistic regression or decision tree
- Maximum 10 features
- Training time under 5 minutes
PRIORITIES:
1. Interpretability
2. Precision (minimize false positives)
3. Overall accuracy
"
```
