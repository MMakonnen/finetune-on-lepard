from datasets import Dataset
from sentence_transformers.evaluation import InformationRetrievalEvaluator

def get_ir_evaluator(dataset: Dataset, name: str) -> InformationRetrievalEvaluator:
    df = dataset.to_pandas()

    # Build dicts: str IDs → texts
    queries = { f"q{i}": txt
                for i, txt in enumerate(df['question'].tolist()) }
    corpus  = { f"p{i}": txt
                for i, txt in enumerate(df['answer'].tolist()) }

    # Ground-truth: map each query ID to a list of relevant passage IDs
    # Here we assume one-to-one alignment: q0→p0, q1→p1, etc.
    relevant_docs = { qid: [qid.replace("q", "p")]
                      for qid in queries.keys() }

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        show_progress_bar=True,
        batch_size=128,
        corpus_chunk_size=1024
    )