
# Evaluation Framework

This folder contains the evaluation framework for measuring the performance of the Enterprise RAG pipeline in `backend/app/` using FinanceBench 10-K financial filings.

No files in `backend/app/` were modified for these experiments. Evaluation scripts import the existing pipeline and apply experimental changes, such as alternative prompts and retrieval settings, at runtime.

The experimental findings, analysis, limitations, and recommendations are documented in [REPORT.md](REPORT.md).

## Requirements

- Python 3.12
- Repository virtual environment
- Ollama running locally
- FinanceBench open-source dataset
- Required Python packages:
  - pandas
  - scipy
  - chromadb
  - ollama
  - sentence-transformers

Install dependencies:

```powershell
pip install -r requirements.txt
```

Pull the Ollama model:

```powershell
ollama pull qwen2.5:3b
```

Make sure Ollama is running before executing the evaluation scripts.

## FinanceBench Dataset

Download the FinanceBench open-source dataset from:

https://github.com/patronus-ai/financebench

The evaluation uses:

- `data/financebench_open_source.jsonl`
- `pdfs/` containing the corresponding financial filings

Do not commit downloaded PDF files or the `chroma_db/` directory.

## Paths to Configure

Some scripts contain hard-coded Windows paths. Update these paths near the top of the relevant scripts to match your local FinanceBench download location.

| Variable | Scripts |
|---|---|
| `PDF_DIR` | `index_financebench.py`, `chunk_size_experiment.py`, `add_unsupported.py` |
| `SRC` | `build_dataset.py`, `run_oracle.py`, `llm_probe.py` |

## Reproducing the Experiments

Run all commands from the repository root with the virtual environment activated and Ollama running.

### 1. Index the FinanceBench Corpus

```powershell
python evaluation\index_financebench.py
```

This indexes the financial filings into the `policy_documents` ChromaDB collection using the default 200-character chunk size.

Already indexed filings are skipped. The complete corpus contains approximately 38,000 chunks.

### 2. Build the Evaluation Dataset

```powershell
python evaluation\build_dataset.py
python evaluation\add_unsupported.py
```

The dataset builder creates:

`evaluation/dataset_financebench.json`

It contains supported questions and starter unsupported and contradictory cases.

Important: Run `add_unsupported.py` after `build_dataset.py`, because the dataset builder overwrites the output file.

Unsupported questions are initially screened using keyword searches. Confirm their absence manually against the relevant PDF filings.

### 3. Run the Baseline Evaluation

```powershell
python evaluation\run_evaluation.py --out evaluation\results\run_baseline.json

python evaluation\score_results.py evaluation\results\run_baseline.json
```

Optional arguments:

- `--limit N`: Evaluate only N questions.
- `--category supported`: Evaluate supported questions.
- `--category unsupported`: Evaluate unsupported questions.
- `--category contradictory`: Evaluate contradictory questions.

Reusing an existing output file allows an interrupted run to resume.

### 4. Retrieval, Reranker and Chunk-Size Experiments

Run:

```powershell
python evaluation\retrieval_experiment.py

python evaluation\retrieval_experiment.py --scoped

python evaluation\significance.py

python evaluation\chunk_size_experiment.py
```

These experiments evaluate:

- Retrieval depth and evidence-page recall.
- Vector search versus reranking.
- Retrieval restricted to the correct filing.
- Chunk sizes of 200, 600 and 1,000 characters.

The `--scoped` option restricts retrieval to the correct filing.

The chunk-size experiment builds separate ChromaDB collections named `eval_c600` and `eval_c1000` for four filings. It does not modify the default `policy_documents` collection.

### 5. Diagnose Generation and Verification

Run:

```powershell
python evaluation\diagnose_hits.py

python evaluation\run_oracle.py

python evaluation\llm_probe.py

python evaluation\verifier_test.py
```

Purpose of each experiment:

- `diagnose_hits.py`: Identifies retrieval failures.
- `run_oracle.py`: Supplies gold evidence passages directly to the pipeline to isolate generation performance.
- `llm_probe.py`: Compares the original pipeline prompt with an alternative financial-document prompt.
- `verifier_test.py`: Evaluates the hallucination verifier using verbatim, altered-number and unrelated claims.

### 6. Run Alternative Prompt Experiments

Run:

```powershell
python evaluation\run_best_config.py

python evaluation\run_prompt_c.py

python evaluation\run_prompt_d.py

python evaluation\probe_empty_claims.py
```

These experiments use:

- 1,000-character chunks.
- A retrieval pool of 20 candidates.
- Reranking to the top 3 candidates.
- Retrieval restricted to the correct filing.
- Alternative prompt instructions.

The prompt configurations are:

| Run | Description |
|---|---|
| Pipeline prompt (A) | Original pipeline prompt |
| Prompt B | Financial-document-specific prompt |
| Prompt C | Prompt B with sentence-level citation requirements |
| Prompt D | Prompt C with an explicit rule against citation-only answers |

Output files are saved under `evaluation/results/`.

### 7. Score and Analyse Results

Run:

```powershell
python evaluation\score_results.py evaluation\results\run_prompt_d.json

python evaluation\error_analysis.py evaluation\results\run_prompt_d.json

python evaluation\compare_runs.py
```

These scripts:

- Calculate retrieval and answer-generation metrics.
- Analyse claim-level verification failures.
- Identify where questions fail in the evaluation pipeline.
- Compare the baseline and alternative prompt configurations.
- Calculate confidence intervals for reported proportions.

The comparison script generates:

`evaluation/results/comparison.md`

The error-analysis script generates:

`evaluation/results/error_analysis.csv`

Note that `error_analysis.csv` is overwritten each time the error-analysis script runs. Copy it before adding manual labels if you need to preserve previous annotations.

## Project Files

| File | Purpose |
|---|---|
| `dataset_financebench.json` | Evaluation questions, reference answers, evidence pages and categories |
| `build_dataset.py` | Builds the supported-question dataset |
| `add_unsupported.py` | Adds unsupported questions |
| `index_financebench.py` | Indexes FinanceBench PDFs into ChromaDB |
| `run_evaluation.py` | Runs the original pipeline against the dataset |
| `score_results.py` | Calculates evaluation metrics |
| `retrieval_experiment.py` | Evaluates retrieval depth and reranking |
| `significance.py` | Performs paired statistical comparisons |
| `chunk_size_experiment.py` | Compares different chunk sizes |
| `run_oracle.py` | Tests generation with gold evidence |
| `llm_probe.py` | Compares prompt configurations |
| `probe_empty_claims.py` | Investigates citation-only responses |
| `verifier_test.py` | Tests the hallucination verifier independently |
| `run_best_config.py` | Runs Prompt B |
| `run_prompt_c.py` | Runs Prompt C |
| `run_prompt_d.py` | Runs Prompt D |
| `diagnose_hits.py` | Diagnoses retrieval performance |
| `error_analysis.py` | Analyses answer and verification failures |
| `compare_runs.py` | Generates experiment comparison tables |
| `results/` | Stores raw experiment outputs, CSV files and comparison results |

The earlier prototype, `backend/app/evaluate_retrieval.py`, covers retrieval only. The evaluation scripts in this folder provide the expanded evaluation framework.

## Using the Results

The `results/per_case_*.csv` files contain per-question evaluation information, including:

- Question ID and category.
- Pipeline status.
- Evidence-page retrieval results.
- Claim and citation counts.
- Response latency.

These CSV files can be loaded into external analysis tools or dashboards.

## Limitations and Notes

- Evaluation runs use a local Qwen 2.5 3B model through Ollama.
- Each question typically takes approximately 10–20 seconds to evaluate.
- FinanceBench evidence page numbers are zero-based, while the document index uses one-based page numbers.
- `build_dataset.py` detects and applies the required page-number offset.
- Chunk sizes are measured in characters, not tokens.
- Prompt B, C and D were developed while examining the same evaluation questions, introducing a risk of prompt-tuning bias.
- Results should be interpreted cautiously because of the relatively small evaluation sample.
- The experimental retrieval configurations differ from the original shipped pipeline, so improvements cannot be attributed to a single change.

For detailed experimental results, statistical analysis, failure cases and recommendations, see [REPORT.md](REPORT.md).
