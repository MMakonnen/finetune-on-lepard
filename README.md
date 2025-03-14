# **finetune-on-lepard**

This repository contains code to **fine-tune a Llama3 model** on the ETHZ Euler cluster for the **Legal Passage Retrieval Task** using the **LePaRD dataset**. The pipeline consists of three key steps:

1. **Dataset Creation**
2. **Model Training**
3. **Evaluation**

The entire workflow is set up to run efficiently on Euler.

---

## **Setup & Usage**

### **1. Configure Parameters**

Modify `src/config.py` to define key settings for dataset selection, data splits, model parameters, training configurations, and evaluation settings. The configuration file provides extensive flexibility, including options for dataset size, training mode (epoch-based or step-based), LoRA settings, batch sizes, and learning rates.

#### **Update Checkpoint Path**

Before running the fine-tuning pipeline, **update the checkpoint path** in `src/config.py`.

Locate the following line:

```python
"checkpoint_path": "/cluster/scratch/your_euler_username/checkpoints_model",
```

Replace `your_euler_username` with your **Euler cluster account name** to ensure that model checkpoints are correctly saved during fine-tuning. If this is not updated, the pipeline may fail due to missing storage permissions.

---

### **2. Update Email for Job Notifications**

Before running the scripts, open the following job scripts:

- `euler_scripts/jobscript_create_data.sh`
- `euler_scripts/jobscript_finetune_model.sh`
- `euler_scripts/jobscript_eval_model.sh`

Replace **ENTER_YOUR_MAIL_HERE** with your email address. This ensures that you receive notifications when your job starts, completes, or fails.

---

### **3. Adjust Compute Resource Allocation**

Currently, the following resource allocation settings in the dataset creation, fine-tuning, and evaluation job scripts serve as **placeholders**:

#### **Dataset Creation (`jobscript_create_data.sh`)**

```bash
#SBATCH --mem-per-cpu=30G
#SBATCH --time=00:05:00
```

- Preparing the **full 10k dataset** with **30G CPU memory** took **about 3 minutes**.
- Adjust CPU allocation depending on dataset size and processing speed requirements.

#### **Fine-tuning (`jobscript_finetune_model.sh`)**

```bash
#SBATCH --mem-per-cpu=80G
#SBATCH --gres=gpumem:70G
#SBATCH --time=18:00:00
```

- Fine-tuning the **8B Llama3 model** on **20% of the top 10k LePaRD dataset** took **~14 hours** with **80G CPU memory** and **70G GPU memory**.

#### **Evaluation (`jobscript_eval_model.sh`)**

```bash
#SBATCH --mem-per-cpu=40G
#SBATCH --gres=gpumem:38G
```

- Evaluating the **test set (~41k samples)** took **~2.5 hours** using **40G CPU memory** and **38G GPU memory**.

Ensure that these values are updated according to the dataset size, model type, and available cluster resources.

---

### **4. Run the Full Pipeline on Euler**

Execute the following scripts to launch each step of the pipeline:

```bash
sbatch euler_scripts/jobscript_create_data.sh
sbatch euler_scripts/jobscript_finetune_model.sh
sbatch euler_scripts/jobscript_eval_model.sh
```

---

### **5. Recommended Environment Setup**

To ensure compatibility, it is recommended to use a **Conda environment** with the required dependencies. The `requirements.txt` file is available **directly in the root of the GitHub repository**. Follow these standard steps to set up the environment:

```bash
conda create --name finetune-lpr-env python=3.12.9
conda activate finetune-lpr-env
pip install -r requirements.txt
```

The `requirements.txt` file contains the **exact package versions** used in the Conda environment when running the fine-tuning pipeline on the Euler cluster. However, it may include **some additional packages** that are not strictly necessary for running the pipeline.
