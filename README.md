# **finetune-on-lepard**

This repository contains code to **fine-tune a Llama3 model** on the ETHZ Euler cluster for the **Legal Passage Retrieval Task** using the **LePaRD dataset**.

---

## **Current State**: Fine-Tuning and Evaluation Implemented ✅

**Recent Additions**:

- Fine-tuning of the model is now implemented using `finetune_model.py`.
- Model evaluation is also implemented in `eval_finetuned_model.py`.
- To create the **finetuning dataset**, run `create_finetuning_dataset.py` after setting the desired **configuration** in `config.py`.

---

## **Next Step**: Deploying the Model on the ETHZ Euler Cluster 🚀

- Code for deploying the model on the Euler cluster will be added next.
- Minor fixes and cleanups are planned.

---

## **Setup & Usage**

1. **Configure Parameters**

   - Edit `config.py` to specify dataset settings, train/validation/test splits, and other parameters.

2. **Create Finetuning Dataset**

   ```bash
   python create_finetuning_dataset.py
   ```

   - Generates processed data in `finetuning_data/`.

3. **Fine-Tune the Model**

   ```bash
   python finetune_model.py
   ```

   - Trains the model using the prepared dataset.

4. **Evaluate the Fine-Tuned Model**

   ```bash
   python eval_finetuned_model.py
   ```

   - Runs evaluation metrics to assess model performance.

---
