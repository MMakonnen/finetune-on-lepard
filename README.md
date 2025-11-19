# **finetune-on-lepard**

This repository contains code for **Legal Passage Retrieval** — the task of retrieving relevant legal passages of precedent given a legal context — using the **LePaRD dataset** ([Mahari et al., 2024](https://aclanthology.org/2024.acl-long.532.pdf)).

The codebase includes multiple approaches:
- **LLaMA 3.1 8B fine-tuning** (language modeling approach with special tokens)
- **DistilBERT baseline** (classification approach)
- **Embedding-based retrieval** (e5-Mistral-7B and general embedding methods)
- **Metadata integration** (judge names, court information, and other contextual features)

The pipeline consists of three key steps:

1. **Dataset Creation**
2. **Model Training**
3. **Evaluation**

The entire workflow is set up to run efficiently on Euler.

---

## **Setup & Usage**

### **1. Recommended Environment Setup**

To ensure compatibility, it is recommended to use a **Conda environment** with the required dependencies. The `requirements.txt` file is available **directly in the root of the GitHub repository**. Follow these standard steps to set up the environment:

```bash
conda create --name finetune-lpr-env python=3.12.9
conda activate finetune-lpr-env
pip install -r requirements.txt
```

The `requirements.txt` file contains the **exact package versions** used in the Conda environment when running the fine-tuning pipeline on the Euler cluster. However, it may include **some additional packages** that are not strictly necessary for running the pipeline.

---

### **2. Configure Parameters**

Modify `src/config.py` to define key settings for dataset selection, data splits, model parameters, training configurations, and evaluation settings. The configuration file provides extensive flexibility, including options for dataset size, training mode (epoch-based or step-based), LoRA settings, batch sizes, and learning rates.

#### **Update Checkpoint Path**

Before running the fine-tuning pipeline, **update the checkpoint path** in `src/config.py`.

Locate the following line:

```python
"checkpoint_path": "/cluster/scratch/your_euler_username/checkpoints_model",
```

Replace `your_euler_username` with your **Euler cluster account name** to ensure that model checkpoints are correctly saved during fine-tuning. If this is not updated, the pipeline may fail due to missing storage permissions.

#### **Set Dataset and Model Paths**

In `src/config.py`, the following fields must be correctly set after running the dataset creation step:

- **`train_data` and `test_data`**: Once the dataset creation job has run successfully, navigate to the `finetune_datasets` folder in the project directory. Copy the exact filenames of the generated train and test data and update these fields accordingly.
- **`finetuned_model`**: After running both dataset creation and model fine-tuning, update this field with the trained model path before running the evaluation script.

Failure to set these fields correctly will result in missing file errors during training or evaluation.

---

### **3. Update Email for Job Notifications**

Before running the scripts, open the following job scripts:

- `euler_scripts/jobscript_create_data.sh`
- `euler_scripts/jobscript_finetune_model.sh`
- `euler_scripts/jobscript_eval_model.sh`

Replace **ENTER_YOUR_MAIL_HERE** with your email address. This ensures that you receive notifications when your job starts, completes, or fails.

---

### **4. Adjust Compute Resource Allocation**

Currently, the following resource allocation settings in the dataset creation, fine-tuning, and evaluation job scripts serve as **placeholders**:

#### **Dataset Creation (`jobscript_create_data.sh`)**

```bash
#SBATCH --mem-per-cpu=30G
#SBATCH --time=00:05:00
```

- Preparing the **full 10k dataset** with **30G CPU memory** took **about 2 minutes**.
- Adjust CPU allocation depending on dataset size and processing speed requirements.

#### **Fine-tuning (`jobscript_finetune_model.sh`)**

```bash
#SBATCH --mem-per-cpu=80G
#SBATCH --gres=gpumem:70G
#SBATCH --time=18:00:00
```

- Fine-tuning the **8B LLaMA 3 model** on **20% of the top 10k LePaRD dataset** took **~14 hours** with **80G CPU memory** and **70G GPU memory**.

#### **Evaluation (`jobscript_eval_model.sh`)**

```bash
#SBATCH --mem-per-cpu=40G
#SBATCH --gres=gpumem:38G
```

- Evaluating the **test set (~41k samples)** took **~2.5 hours** using **40G CPU memory** and **38G GPU memory**.

Ensure that these values are updated according to the dataset size, model type, and available cluster resources.

---

### **5. Run the Full Pipeline on Euler**

Make sure to run all `sbatch` commands from the **root directory** of the project folder (`finetune-on-lepard`):

```bash
sbatch euler_scripts/jobscript_create_data.sh
sbatch euler_scripts/jobscript_finetune_model.sh
sbatch euler_scripts/jobscript_eval_model.sh
```

**Logs for each step** (dataset creation, fine-tuning, and evaluation) will be created in the root directory. Check these logs to monitor progress or debug issues.

---

### **6. DistilBERT Baseline Setup**

In addition to LLaMA 3 fine-tuning, the repository now includes code to fine-tune a **DistilBERT model** as a **baseline** for the Legal Passage Retrieval task.

- All related code is located in the `src_distilbert` folder.
- Configuration for DistilBERT fine-tuning is defined in `src_distilbert/distilbert_config.py`.
- Job scripts for running the DistilBERT pipeline on Euler are available in the `euler_scripts/distilbert` folder.
- The structure and logic follow the same three steps as for LLaMA 3:
  1. **Data creation**
  2. **Fine-tuning**
  3. **Evaluation**

📌 _Extensive documentation for the DistilBERT setup will follow soon, but the process mirrors the LLaMA 3 pipeline._

---

## **7. Embedding-Based Retrieval (SBERT & e5-Mistral)**

The repository includes code for **embedding-based retrieval approaches** using Sentence-BERT (SBERT) and the **e5-Mistral-7B model**.

### **SBERT Implementation**

- Code is located in the `src_sbert` folder
- Configuration files:
  - `src_sbert/sbert_config.py` — Standard SBERT configuration
  - `src_sbert/sbert_config_e5.py` — e5-Mistral-7B specific configuration
- Job scripts for running SBERT experiments are in `euler_scripts/sbert` and `euler_scripts/sbert_e5`
- The pipeline follows the same three-step structure: data creation, fine-tuning, and evaluation

### **e5-Mistral-7B Embedding Model**

The e5-Mistral-7B model provides an alternative embedding-based approach to passage retrieval:

- **Configuration**: `src_sbert/sbert_config_e5.py`
- **Training**: `src_sbert/sbert_finetune_e5.py`
- **Evaluation**: `src_sbert/sbert_eval_e5.py`
- **Data creation**: `src_sbert/sbert_create_data.py`

**Note**: Embedding-based approaches (including e5-Mistral) have shown substantially worse accuracy compared to classification-based methods. Results suggest that approaches leveraging target-passage classification outperform those relying purely on embeddings.

---

## **8. Metadata Scraping and Integration**

### **Metadata Scraping Pipeline**

The repository includes scripts for **scraping and enriching legal case metadata** from external sources:

- **Location**: `meta_data_scraping/` folder
- **Main scripts**:
  - `scrape_meta_data.py` — Core scraping functionality
  - `scrape_meta_data_hybrid.py` — Hybrid scraping approach
  - `create_data_to_enrich.py` — Prepares data for enrichment
- **Data storage**: Scraped metadata is stored in `meta_data_scraping/meta_judge_data/`

The scraping pipeline enriches case entries with:
- **Judge names** (scraped from CourtListener based on formatted citation strings)
- **Court information** (destination court, source court)
- **Temporal metadata** (source dates)

### **Judge Metadata Integration**

**Judge names** have been identified as the most promising metadata feature, consistently improving retrieval accuracy when integrated into the pipeline.

**Current Implementation**:
- Judge names are scraped from CourtListener for all cases in the dataset
- The scraping ensures **complete coverage** across train/validation/test splits to avoid evaluation bias
- Judge metadata is integrated into both training and evaluation pipelines
- Data files with judge metadata are stored in `finetuning_data_judge/`

**Important Considerations**:
- ⚠️ **Label Leakage Prevention**: Only metadata available at retrieval time (e.g., destination court, judges) should be included. Source-related information (source court, source date) directly encodes information about the target passage and constitutes label leakage.
- ⚠️ **Complete Coverage**: When adding external metadata, ensure it is consistently available for **all samples** in all splits (train/val/test), not just the training set. Partial coverage can silently bias evaluation results.

**Job Scripts**:
- `euler_scripts/meta_data_judges/jobscript_JUDGES_create_data.sh` — Creates enriched datasets with judge metadata
- `euler_scripts/meta_data_judges/jobscript_JUDGES_scrape_data.sh` — Runs the scraping pipeline

---

## **9. Project Background & Key Insights**

### **Research Context**

This project investigates Legal Passage Retrieval using the LePaRD dataset. The baseline established in prior work showed that a **simple classification approach using DistilBERT** performed best. Building on this:

1. **LLaMA 3.1 8B Approach**: We replaced DistilBERT with a larger Llama 3.1 8B model, reframing the task as a **language modeling problem**. The model's vocabulary was extended with 10k special tokens (representing the 10k precedent passages), and the model was fine-tuned to predict the correct special token for each context. This improved retrieval accuracy modestly (a few percentage points).

2. **Embedding Approaches**: We tested embedding-based retrieval (e.g., using the e5-Mistral-7B model), but this yielded substantially worse accuracy compared to classification methods.

3. **Metadata Integration**: Subsequent experiments focused on including metadata during training and retrieval (e.g., court of origin, judge names). The most successful extension was the **inclusion of judge names**, scraped from CourtListener, which consistently improved retrieval accuracy.

### **Key Findings**

- **Classification approaches** outperform embedding-based retrieval methods
- **Larger models** (e.g., DistilBERT → Llama 3) improve performance slightly, but **architectural or data innovations** (e.g., metadata, judge embeddings) yield greater gains
- Merely expanding existing dataset metadata provides minor improvements; **adding new, meaningful information** (e.g., judges, court hierarchy) is more impactful
- Incorporating target passage representations offered no meaningful accuracy gain
- **Top-10 retrieval accuracy** is a useful metric: improvements matter most in difficult cases where baseline models already perform well on easy ones

### **Reference Results**

| Model / Setting                           | Data              | Top-1  | Top-5  | Top-10 | Notes                                  |
| ----------------------------------------- | ----------------- | ------ | ------ | ------ | -------------------------------------- |
| DistilBERT (Classification, context only) | 10 k              | 0.3817 | 0.7159 | 0.8100 | baseline                               |
| DistilBERT (+ DEST court metadata only)   | 10 k              | 0.3833 | 0.7189 | 0.8134 | minor gain                             |
| Llama 3 (Classification, context only)    | 10 k              | 0.4061 | 0.7578 | 0.8490 | took ~130 h (80 GB GPU)                |
| Llama 3 (20% subset, context only)        | 41 k test samples | 0.2884 | 0.5932 | 0.6969 | ~13 h train + 2 h eval                 |
| Llama 3 (+ Judge metadata)                | same subset       | 0.3122 | 0.6398 | 0.7501 | judge info helps                       |
| e5-Mistral-7B (embedding approach)        | 25% subset        | 0.1008 | 0.2864 | 0.3941 | direct embedding ranking; 30 h / epoch |

### **Dataset Information**

- Always use the **10k-passage version** of the LePaRD dataset: [https://huggingface.co/datasets/rmahari/LePaRD](https://huggingface.co/datasets/rmahari/LePaRD)
- Note: The file labeled "train" actually contains the full dataset — it must be manually split into train, validation, and test sets
- Standard splits: initially 80/10/10, later switched to **90/5/5** (seed 42)
- Training: **1 epoch** for all runs
- Always evaluate **Top-1, Top-5, Top-10 retrieval accuracy**

### **Efficient Experimentation Tips**

- Use **subsets (≈ 20–25%)** of the dataset for quick iteration — relative model performance trends are consistent with full-dataset results
- To test new metadata ideas, start with **DistilBERT** (fast to train). Larger models like Llama 3 may extract richer signals, but DistilBERT suffices to test whether a concept is worth scaling up

---

## **10. Next Steps**

The most promising direction for future work is **integrating judge embeddings into the retrieval pipeline in a more sophisticated way**:

1. **Obtain judge embeddings** from Robert Mahari (available via coordination)
2. **Explore modeling strategies** to integrate judge embeddings into the retrieval architecture beyond simple metadata inclusion
3. **Train and evaluate** new approaches on the 20% subset for comparison

The current implementation includes judge names as metadata, but there is significant potential to leverage **judge embeddings** (dense representations of judges) to further improve retrieval accuracy.

---

## **Important Notes for Future Development**

### **Avoiding Label Leakage**

Be meticulous about avoiding information leakage:
- Only include features that are **actually available at retrieval time** (e.g., destination court, judges)
- Treat any feature tied to the ground-truth passage (source-related information) as a red flag for potential leakage
- Source court and source date directly encode information about the origin of the passage and reveal something about the label being predicted

### **Consistent Data Coverage**

When adding external metadata (judges, embeddings, etc.):
- Double-check that it is consistently available for **all samples** in all splits, not just the training set
- Partial coverage can quietly undermine conclusions, even if everything "looks fine" at first glance
- Ensure that any enrichment (scraping, embeddings, metadata) is **aligned with the final train/val/test splits**

### **Reproducibility**

- Always record **seeds and hyperparameters** for reproducibility
- Maintain consistent **train/val/test splits** across experiments
- Document any changes to data preprocessing or enrichment pipelines

---

Let me know if you'd like a Markdown table of contents or internal links added.
