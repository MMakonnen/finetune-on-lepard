# **finetune-on-lepard**

This repository contains code to **fine-tune a Llama3 model** on the ETHZ Euler cluster for the **Legal Passage Retrieval Task** using the **LePaRD dataset**.

---

## **Current State**: Finetuning Dataset Creation Complete ✅

**Recent Additions**:

- To create the **finetuning dataset**, run `create_finetuning_dataset.py` after setting the desired **configuration** in `config.py`.

---

## **Next Step**: Clean up and upload Model Training Code 🚀

---

## **Setup & Usage**

1. **Configure Parameters**

   - Edit `config.py` to specify dataset settings, train/validation/test splits, and other parameters.

2. **Create Finetuning Dataset**

   ```bash
   python create_finetuning_dataset.py
   ```

   - Generates processed data in `finetuning_data/`.

3. **Train the Model** (Coming Soon)
