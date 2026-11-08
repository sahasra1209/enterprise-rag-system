Evaluation framework



Evaluates the Enterprise RAG pipeline in backend/app/ on FinanceBench 10-K filings. Nothing in backend/app/ is modified: scripts import the pipeline and, where an experiment needs different settings (prompt, retrieval pool), patch the relevant function at run time inside the script.



Findings are in REPORT.md.



Requirements

Python 3.12 and the repository's virtual environment (pip install -r requirements.txt). The evaluation scripts also need pandas, scipy, chromadb, ollama and sentence-transformers; make sure these are in requirements.txt.

Ollama running locally, with the model the pipeline uses:

powershell

&#x20; ollama pull qwen2.5:3b

The FinanceBench open-source sample (github.com/patronus-ai/financebench): the data/financebench\_open\_source.jsonl file and the pdfs/ folder. Do not commit the PDFs or the chroma\_db/ folder.

Paths to edit



Several scripts contain hard-coded Windows paths near the top. Change them to where you downloaded FinanceBench:



Variable	Used in

PDF\_DIR (folder of PDFs)	index\_financebench.py, chunk\_size\_experiment.py, add\_unsupported.py

SRC (path to financebench\_open\_source.jsonl)	build\_dataset.py, run\_oracle.py, llm\_probe.py

Reproducing the results



Run from the repository root, with the virtual environment active and Ollama running. Order matters.



1\. Index the corpus (200-character chunks, the pipeline's default)

powershell

python evaluation\\index\_financebench.py



Writes to the policy\_documents collection in ./chroma\_db. Already-indexed filings are skipped. Takes several minutes (about 38,000 chunks).



2\. Build the test set

powershell

python evaluation\\build\_dataset.py

python evaluation\\add\_unsupported.py



build\_dataset.py writes evaluation/dataset\_financebench.json (39 supported questions plus starter unsupported and contradictory cases) and prints the page-number offset it detected. It overwrites the file, so run add\_unsupported.py after it, never before. add\_unsupported.py adds unsupported questions only if their keywords do not appear together on any page of the filing; confirm each added question by hand.



3\. Baseline: pipeline as shipped

powershell

python evaluation\\run\_evaluation.py --out evaluation\\results\\run\_baseline.json

python evaluation\\score\_results.py evaluation\\results\\run\_baseline.json



Options for run\_evaluation.py: --limit N, --category supported|unsupported|contradictory. Re-using an --out file resumes an interrupted run. The scorer also writes evaluation/results/per\_case\_<run>.csv.



4\. Retrieval, reranker, chunk size

powershell

python evaluation\\retrieval\_experiment.py

python evaluation\\retrieval\_experiment.py --scoped

python evaluation\\significance.py

python evaluation\\chunk\_size\_experiment.py

\--scoped restricts retrieval to the correct filing.

significance.py reads retrieval\_experiment\_scoped.json (paired tests, reranker versus vector).

chunk\_size\_experiment.py builds two extra collections, eval\_c600 and eval\_c1000, for four filings (AMD, American Express, Boeing, Amcor). It does not touch policy\_documents. The prompt runs below need eval\_c1000.

5\. Where the pipeline loses answers

powershell

python evaluation\\diagnose\_hits.py

python evaluation\\run\_oracle.py

python evaluation\\llm\_probe.py

python evaluation\\verifier\_test.py

run\_oracle.py gives the pipeline the gold evidence passages instead of its own retrieval.

llm\_probe.py compares the pipeline's prompt with a reworded prompt on gold evidence.

verifier\_test.py tests the hallucination detector alone on verbatim, number-altered, and unrelated claims.

6\. End-to-end runs with larger chunks and alternative prompts

powershell

python evaluation\\run\_best\_config.py

python evaluation\\run\_prompt\_c.py

python evaluation\\run\_prompt\_d.py

python evaluation\\probe\_empty\_claims.py



Each uses 1,000-character chunks (eval\_c1000), retrieves 20 candidates inside the correct filing, reranks to 3, and runs the pipeline with a modified prompt. run\_best\_config.py is prompt B and writes run\_best\_config.json. Prompts C and D are defined at the top of their scripts. probe\_empty\_claims.py re-runs four questions and prints the raw model output.



7\. Analysis and comparison

powershell

python evaluation\\score\_results.py evaluation\\results\\run\_prompt\_d.json

python evaluation\\error\_analysis.py evaluation\\results\\run\_prompt\_d.json

python evaluation\\compare\_runs.py

error\_analysis.py prints the failure funnel and writes evaluation/results/error\_analysis.csv (overwritten on every run; copy it first if you add hand labels, for example to error\_analysis\_promptD.csv).

compare\_runs.py builds the comparison tables and 95% confidence intervals from the run files and writes evaluation/results/comparison.md.

Files

File	Purpose

dataset\_financebench.json	Test set: questions, reference answers, evidence pages, categories

build\_dataset.py, add\_unsupported.py	Build and extend the test set

index\_financebench.py	Index FinanceBench PDFs into ChromaDB in batches (ChromaDB limits a single call to 5,461 records)

run\_evaluation.py	Run the pipeline as shipped over the dataset and save raw outputs

score\_results.py	Score a run: retrieval hit, answer status, verification, citations

retrieval\_experiment.py, significance.py	Page-level retrieval and reranker comparison with paired tests

chunk\_size\_experiment.py	Compare 200, 600 and 1,000-character chunks

run\_oracle.py, llm\_probe.py, probe\_empty\_claims.py	Diagnose the generation step

verifier\_test.py	Test the hallucination detector in isolation

run\_best\_config.py, run\_prompt\_c.py, run\_prompt\_d.py	End-to-end runs with larger chunks and alternative prompts

error\_analysis.py, diagnose\_hits.py, compare\_runs.py	Failure analysis and run comparison

results/	Raw run files (run\_\*.json), per-case CSVs, experiment outputs, comparison.md

dataset.json	Early 4-question prototype on the 3M 2015 filing, kept for reference



The earlier prototype script backend/app/evaluate\_retrieval.py was added in the first pull request and covers retrieval only; the scripts above supersede it.



Using the results elsewhere



results/per\_case\_\*.csv has one row per question (category, tier, status, page hit, claims and citations counts, latency) and can be loaded directly by the dashboard.



Notes

Evaluation runs call a local 3B model and take roughly 10 to 20 seconds per question.

Page numbers: FinanceBench evidence pages are 0-based; the index is 1-based. build\_dataset.py detects the offset (+1) from the data.

Chunk sizes are in characters, not tokens.

