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

### **2. Update Email for Job Notifications**

Before running the scripts, open `euler_scripts/jobscript_finetune_model.sh` and `euler_scripts/jobscript_eval_model.sh`, then replace **ENTER_YOUR_MAIL_HERE** with your email address. This ensures that you receive updates when your job is queued and when it completes.

### **3. Adjust Compute Resource Allocation**

Currently, the following resource allocation settings in the fine-tuning and evaluation job scripts serve as **placeholders**:

#### Fine-tuning (`jobscript_finetune_model.sh`):

```bash
#SBATCH --mem-per-cpu=80G
#SBATCH --gres=gpumem:70G
#SBATCH --time=18:00:00
```

These values should be adjusted depending on the compute requirements of the task.

- As a guideline, fine-tuning the **8B Llama3 model** on **20% of the top 10k LePaRD dataset** required **~14 hours** with **80G CPU memory** and **70G GPU memory**.

#### Evaluation (`jobscript_eval_model.sh`):

```bash
#SBATCH --mem-per-cpu=40G
#SBATCH --gres=gpumem:38G
```

- Evaluating the corresponding **test set (~41k samples)** took **~2.5 hours** using **40G CPU memory** and **38G GPU memory**.

Ensure that these values are updated according to the model size, dataset fraction, and available cluster resources.

### **4. Run the Full Pipeline on Euler**

Execute the following scripts to launch each step of the pipeline:

```bash
sbatch euler_scripts/jobscript_create_data.sh
sbatch euler_scripts/jobscript_finetune_model.sh
sbatch euler_scripts/jobscript_eval_model.sh
```

### **5. Recommended Environment Setup**

To ensure compatibility, it is recommended to use a **Conda environment** with the required dependencies. The `requirements.txt` file is available **directly in the root of the GitHub repository**. Follow these standard steps to set up the environment:

```bash
conda create --name finetune-lpr-env python=3.12.9
conda activate finetune-lpr-env
pip install -r requirements.txt
```

The `requirements.txt` file contains the **exact package versions** used in the Conda environment when running the fine-tuning pipeline on the Euler cluster. However, it may include **some additional packages** that are not strictly necessary for running the pipeline.
