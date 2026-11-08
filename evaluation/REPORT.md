Evaluation of the Enterprise RAG Pipeline on FinanceBench

Author: Geethika Kodela
Scope: Measure how the existing pipeline (backend/app/) behaves on real financial filings.

No file in backend/app/ was modified. Scripts that change pipeline behaviour (prompt, retrieval settings) do so at runtime, inside the evaluation scripts only.

Note: Items marked TODO must be completed or confirmed by the author before this report is final.
1. Summary

As shipped, the pipeline refused every FinanceBench question (44 of 44, including the 39 answerable ones). It uses 200-character chunks, retrieves 3 chunks, and reranks those same 3.

Retrieval is a bottleneck, but not the only one. The page containing the evidence was in the 3 chunks given to the LLM for 16% of supported questions (4 of 25) in the shipped configuration. With 1,000-character chunks and a 20-candidate pool it was 36% (9 of 25).

Even with the correct evidence page supplied, the pipeline's own prompt refused 37 of 39 questions (oracle run). A prompt written for financial filings produced answers for about half of the questions.

No tested configuration produced a meaningful number of verified answers. At most 1 of 25 supported questions passed verification (prompt C), and prompt D produced 13 answers with real text and 0 verified.

The hallucination verifier behaved well on controlled claims: 16 of 16 verbatim claims were accepted, and none of 32 altered or unrelated claims was accepted. On real LLM answers it accepted almost nothing, so it appears sensitive to paraphrase. This is a hypothesis, not a tested conclusion.

The reranker showed no statistically significant effect at this sample size (n = 39).

Larger chunks look better but the evidence is suggestive: at 1,000 characters, hit@3 gained 5 questions and lost 0 versus 200 characters (sign test p = 0.062, n = 25).

All sample sizes are small. Confidence intervals are given where proportions are reported, and differences of one or two questions should be treated as noise.

2. Setup
2.1 Pipeline Under Test

backend/app/answer.py, function generate_answer(question).

It retrieves with search_documents(top_k=3), reranks those 3, generates an answer with Ollama (qwen2.5:3b, temperature 0), splits the answer into claims, and verifies each claim against its cited passage with an NLI model.

Returned fields used by the evaluation:

answer
overall_status
claims
citations
retrieved_chunks
latency_ms
error
2.2 Corpus

Eight FinanceBench 10-K filings, selected because they have the most open-source questions:

AMD 2022
American Express 2022
Boeing 2022
PepsiCo 2022
Amcor 2023
3M 2022
Best Buy 2023
CVS Health 2022

Indexed at 200-character chunks (the pipeline's default, chunk_size=200, counted in characters) into the policy_documents collection: 38,230 chunks in total, including an earlier 3M 2015 filing that has no FinanceBench questions.

2.3 Test Set (evaluation/dataset_financebench.json)

39 supported questions from the FinanceBench open-source sample (12 information extraction, 27 reasoning), with reference answers and evidence pages.

FinanceBench page numbers are 0-based; the index is 1-based. The offset (+1) was verified against all 39 evidence items.

Unsupported questions: 3 in the first run, 9 in the final runs. Each was screened by keyword search of the filing's pages and should be confirmed by hand.

TODO: Confirm each unsupported question against its PDF.

Contradictory cases: Two cases (CONTRA-1/2) put a false number inside the question. The pipeline verifies its own generated claims, not the question, so these do not test the verifier and were dropped from scoring. The verifier is evaluated directly in Section 3.7.

The later end-to-end runs use 25 supported questions from the four filings for which a larger-chunk index was built (AMD, American Express, Boeing, Amcor).

2.4 Metrics
Gold page in context: The filing's evidence page appears among the chunks passed to the LLM. This is a strict proxy, because the same fact can appear on other pages.
Refused: overall_status = insufficient_evidence.
Empty reply: The model's reply was only a citation tag, so the claim text is empty.
Answer with real text: The model produced an answer containing actual text.
Verified: overall_status = supported.
Real text, rejected: Answer text was produced but not verified.

Proportions carry 95% Wilson confidence intervals. Rank comparisons use the Wilcoxon signed-rank test; paired hit/miss comparisons use an exact sign test.

3. Experiments and Results
3.1 Baseline: Pipeline as Shipped

44 cases (39 supported, 3 unsupported, 2 contradictory), all returned insufficient_evidence.

The right filing was among the retrieved chunks for:

92% of extraction questions (11/12)
81% of reasoning questions (22/27)

However, the right page was retrieved for only:

2/12 extraction questions
3/27 reasoning questions

Refusing all unsupported questions (3/3) is not evidence of good discrimination, because the system also refused all answerable ones.

3.2 Retrieval Depth (n = 39, Right Page Among Top K)
Setting	@3	@5	@10	@20	@50	MRR
Vector search, all filings (as the pipeline does)	13%	18%	23%	31%	44%	0.11
Vector search, correct filing only	13%	23%	31%	44%	59%	0.14

Restricting to the correct filing helps modestly. Even with 50 candidates inside the correct filing, 41% of questions never surface their evidence page.

Locating the right page inside a long filing is the main retrieval difficulty.

3.3 Reranker (Cross-Encoder), Correct Filing Only (n = 39)
Setting	@3	@5	@10	MRR
Vector only	13%	23%	31%	0.14
Retrieve 20, rerank	21%	28%	31%	0.20
Retrieve 50, rerank	21%	26%	31%	0.19

Paired tests for "retrieve 20, rerank" versus vector only:

Reciprocal rank: better on 8 questions, worse on 12, tied on 19 (Wilcoxon p = 0.723).
Hit@3: gained on 4, lost on 1 (exact sign test p = 0.375).

Conclusion: No significant benefit detected at n = 39. The reranker improves some questions substantially and degrades others.

Note that reranking only 3 retrieved chunks (as the shipped pipeline does) can reorder them but cannot recover missing evidence.

3.4 Chunk Size (4 Filings, 25 Supported Questions, Correct Filing Only)
Chunk size	Right page in top 3	Gold-number coverage in top 3 (n = 15)	Hit@3 vs 200 chars
200 chars (shipped)	16%	11%	—
600 chars	28%	28%	Gained 4, lost 1 (p = 0.375)
1,000 chars	36%	28%	Gained 5, lost 0 (p = 0.062)

The embedding model reads at most 256 word-pieces (max_seq_length = 256), roughly 1,000 characters of prose, so larger chunks would be partly ignored.

Page recall favours larger chunks by construction (fewer chunks per page); gold-number coverage, the share of key figures from the gold evidence present in the top 3 chunks, is the fairer measure.

Larger chunks also put more text in the prompt (about 2,800 characters for 1,000-character chunks), and the effect of that on a 3B model was not isolated.

3.5 Oracle Run: Correct Evidence Supplied (n = 39)

The pipeline was given the gold evidence passages (at most 2, truncated to 1,800 characters) instead of its own retrieval.

Gold page in context: 39/39.
Refused (insufficient_evidence): 37/39.
Answers produced: 2/39.
Verified: 0/39.

With perfect retrieval the shipped pipeline still refused 95%. This isolates a problem after retrieval: the prompt, the model, or both.

3.6 Prompt Comparison
Prompt	Description
A	The pipeline's own prompt ("policy-document question-answering assistant")
B	Reworded for 10-K filings, citations requested, numbers may be computed
C	B plus: every sentence must end with an evidence ID, at most three sentences, no commentary
D	C plus: never reply with only an evidence ID

Exact text:

Prompt B: run_best_config.py
Prompt C: run_prompt_c.py
Prompt D: run_prompt_d.py
Prompts A and B: llm_probe.py

Probe on gold evidence, 15 questions (12 extraction, 3 reasoning):

Prompt A refused 15/15.
Prompt B refused 10/15 and answered 5.
Of the answered questions with numbers, 1 of 3 matched the reference.

Prompts B, C and D were written and adjusted while looking at the same 25 questions, so later prompts have an advantage on them.

3.7 Verifier Tested in Isolation (verifier_test.py)

Claims were built from FinanceBench evidence sentences (16 usable cases of 39) and checked against their passage plus 150 characters of context.

Claim type	n	Result
Verbatim sentence	16	16 supported
Same sentence, one number altered	16	13 contradicted, 3 not established, 0 supported
Sentence from a different company's filing	16	13 not established, 3 contradicted, 0 supported

This is a best-case test: a verbatim claim is trivially entailed, several alterations were obvious (for example December 71, 5022), and subtle numeric errors are likely harder.

The verifier never accepted a false claim in this test.

3.8 End-to-End Comparison (25 Supported Questions Common to All Runs)
Run (n=25 supported questions)	Gold page in context	Refused	Empty reply (tag only)	Answer with real text	Verified	Real text, rejected
Pipeline as shipped (200-char chunks, top-3)	4/25 (16%, CI 6-35%)	25/25 (100%, CI 87-100%)	0/25 (0%, CI 0-13%)	0/25 (0%, CI 0-13%)	0/25 (0%, CI 0-13%)	0/25 (0%, CI 0-13%)
Prompt B (1,000-char chunks, pool 20, rerank 3)	9/25 (36%, CI 20-55%)	18/25 (72%, CI 52-86%)	0/25 (0%, CI 0-13%)	7/25 (28%, CI 14-48%)	0/25 (0%, CI 0-13%)	7/25 (28%, CI 14-48%)
Prompt C	9/25 (36%, CI 20-55%)	16/25 (64%, CI 45-80%)	4/25 (16%, CI 6-35%)	5/25 (20%, CI 9-39%)	1/25 (4%, CI 1-20%)	4/25 (16%, CI 6-35%)
Prompt D	9/25 (36%, CI 20-55%)	12/25 (48%, CI 30-67%)	0/25 (0%, CI 0-13%)	13/25 (52%, CI 33-70%)	0/25 (0%, CI 0-13%)	13/25 (52%, CI 33-70%)

Unsupported questions: All 9 were refused under prompts C and D (CI 70-100%); the 2 questions common to all runs were refused in every run.

Under prompt D the system answered 52% of supported questions and still refused all 9 unsupported ones, which is weak evidence that refusals are not purely indiscriminate (small n).

Repeatability: Prompt C run twice gave the same overall_status on 34/34 questions. An earlier prompt B configuration run twice differed on one question (20 versus 21 refusals), so one-question differences between runs are within noise. Only the status was compared, not the answer text.

What changed between the shipped run and the later runs: chunk size, retrieval pool and reranking, source filtering to the correct filing, and the prompt all changed together, so the improvement cannot be attributed to any single change.

3.9 Where Answers Are Lost (Failure Funnel, Prompt D, 25 Supported Questions)
9 refused with the evidence page not in context, and 3 refused with it in context.
13 answered, then rejected (7 without the gold page in context, 6 with it).
0 verified.

Claim-level:

12 uncertain (all not_established by the verifier).
1 invalid_citation.
No claim was contradicted and none supported.

Under prompt C, four of the nine answered questions contained only a citation tag (for example [E1]) with no answer text. The pipeline reports these as needs_review, which hides that no answer was produced.

Under prompt B, 14 of 21 claims were missing_citation; under prompt C this fell to 2 of 11.

4. Error analysis (hand labels)

All 13 answered questions from prompt D were manually reviewed and labeled against the FinanceBench reference answers.

Classification	Count
Correct	5
Partly correct	4
Wrong	3
Off-topic	1
Total	13

The manual review shows that although prompt D generated real answer text for 13 of 25 supported questions, only 5 were judged fully correct. Four were partly correct, three were wrong, and one was off-topic.

This highlights that generating an answer does not necessarily mean generating a correct answer. The verifier rejected all 13 answers, but the manual review suggests that some answers were factually correct or partly correct despite their rejection.

The labels are recorded in evaluation/results/error_analysis_promptD.csv in the your_label column.

5. Limitations
Small samples: 25 supported and 9 unsupported questions in the end-to-end runs (39 supported in the retrieval tests). Most confidence intervals are wide.
The end-to-end, chunk-size and prompt experiments use four filings, and retrieval was restricted to the correct filing, which the shipped pipeline does not do. Those retrieval numbers are upper bounds.
One embedding model (all-MiniLM-L6-v2), one LLM (qwen2.5:3b), one NLI verifier.
Prompts B, C and D were tuned on the same questions they were evaluated on.
Several factors changed at once between the shipped run and the later runs.
"Gold page in context" undercounts retrieval success when the same fact appears on other pages.
Numeric matching against reference answers is a rough heuristic; correctness judgements rely on hand labels by one person.
The verifier test is a best-case setting and does not cover subtle errors or paraphrases.
Only overall_status was compared for repeatability.
RAGAS metrics (faithfulness, answer relevance, context precision) were not computed. A local 3B model would be an unreliable judge, and a hosted judge was not used. This is a gap against the original plan.
Chunk-size results cover three sizes only, with one overlap setting (20% of the chunk size).
6. Recommendations for the Pipeline

Each recommendation lists its supporting evidence and what remains untested.

Use larger chunks (about 1,000 characters). Hit@3 gained 5 and lost 0 versus 200 characters (p = 0.062). Do not go much larger without changing the embedding model, since it truncates at about 1,000 characters.
Revise the answer prompt for financial documents. The shipped prompt refused 95% of questions even with the correct evidence. Include a rule that a reply must contain answer text, not only a citation tag.
Handle empty or tag-only replies explicitly. Report them as a generation failure instead of needs_review.
Examine the verifier on paraphrased claims. It accepted all verbatim claims but nearly no paraphrased ones in real answers. Test it against gold evidence with human-written paraphrases before relaxing any threshold.
Do not adopt the reranker as an improvement yet. No significant effect was found at n = 39, and reranking 3 chunks out of 3 cannot recover missing evidence. If kept, rerank a larger pool.
Expose retrieval parameters. Expose top_k, chunk size and an optional source filter as parameters of generate_answer, so evaluation can vary them without patching.
Remove debug output. Remove the NLI DEBUG print statements in the hallucination detector.
Review the citation gate. A correct answer (AmEx effective tax rate) was rejected because the model cited an evidence ID that was not in the context.
7. Reproducing the Results

See evaluation/README.md.

Scripts are in evaluation/; raw results are in evaluation/results/, and evaluation/results/comparison.md is generated by compare_runs.py.