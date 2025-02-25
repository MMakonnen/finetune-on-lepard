# **finetune-on-lepard**

This repository contains code to **fine-tune a Llama3 model** on the ETHZ Euler cluster for the **Legal Passage Retrieval Task** using the **LePaRD dataset**.

---

## **Current State**: Full Pipeline Running on Euler ✅

**Recent Additions**:

- The entire pipeline (**dataset creation → training → evaluation**) is now set up on the Euler cluster.

---

## **Next Step**: Training & Iteration on Euler 🚀

- Running full training runs based on available compute.
- Iterating on hyperparameters and model performance.
- Further optimizations and cleanup.

---

## **Setup & Usage**

1. **Configure Parameters**

   - Edit `src/config.py` to specify dataset settings, train/validation/test splits, and other parameters.

2. **Run the Full Pipeline on Euler**
   - **Submit jobs using the provided shell scripts:**
     ```bash
     sbatch euler_scripts/jobscript_create_data.sh
     sbatch euler_scripts/jobscript_finetune_model.sh
     sbatch euler_scripts/jobscript_eval_model.sh
     ```
