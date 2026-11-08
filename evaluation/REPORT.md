Evaluation of the Enterprise RAG pipeline on FinanceBench



Author: Geethika Kodela (evaluation framework) Scope: measure how the existing pipeline (backend/app/) behaves on real financial filings. No file in backend/app/ was modified. Scripts that change pipeline behaviour (prompt, retrieval settings) do so at run time, inside the evaluation scripts only.



Items marked TODO need to be filled in or confirmed by the author before this report is final.



1\. Summary

As shipped, the pipeline refused every FinanceBench question (44 of 44, including the 39 answerable ones). It uses 200-character chunks, retrieves 3 chunks, and reranks those same 3.

Retrieval is a bottleneck, but not the only one. The page containing the evidence was in the 3 chunks given to the LLM for 16% of supported questions (4 of 25) in the shipped configuration. With 1,000-character chunks and a 20-candidate pool it was 36% (9 of 25).

Even with the correct evidence page supplied, the pipeline's own prompt refused 37 of 39 questions (oracle run). A prompt written for financial filings produced answers for about half of the questions.

No tested configuration produced a meaningful number of verified answers. At most 1 of 25 supported questions passed verification (prompt C), and prompt D produced 13 answers with real text and 0 verified.

The hallucination verifier behaved well on controlled claims: 16 of 16 verbatim claims were accepted, and none of 32 altered or unrelated claims was accepted. On real LLM answers it accepted almost nothing, so it appears sensitive to paraphrase. This is a hypothesis, not a tested conclusion.

The reranker showed no statistically significant effect at this sample size (n = 39).

Larger chunks look better but the evidence is suggestive: at 1,000 characters, hit@3 gained 5 questions and lost 0 versus 200 characters (sign test p = 0.062, n = 25).



All sample sizes are small. Confidence intervals are given where proportions are reported, and differences of one or two questions should be treated as noise.



2\. Setup

2.1 Pipeline under test



backend/app/answer.py, function generate\_answer(question). It retrieves with search\_documents(top\_k=3), reranks those 3, generates an answer with Ollama (qwen2.5:3b, temperature 0), splits the answer into claims, and verifies each claim against its cited passage with an NLI model. Returned fields used by the evaluation: answer, overall\_status, claims, citations, retrieved\_chunks, latency\_ms, error.



2.2 Corpus



Eight FinanceBench 10-K filings, selected because they have the most open-source questions: AMD 2022, American Express 2022, Boeing 2022, PepsiCo 2022, Amcor 2023, 3M 2022, Best Buy 2023, CVS Health 2022. Indexed at 200-character chunks (the pipeline's default, chunk\_size=200, counted in characters) into the policy\_documents collection: 38,230 chunks in total, including an earlier 3M 2015 filing that has no FinanceBench questions.



2.3 Test set (evaluation/dataset\_financebench.json)

39 supported questions from the FinanceBench open-source sample (12 information extraction, 27 reasoning), with reference answers and evidence pages. FinanceBench page numbers are 0-based; the index is 1-based. The offset (+1) was verified against all 39 evidence items.

Unsupported questions (the answer is not in the filings): 3 in the first run, 9 in the final runs. Each was screened by keyword search of the filing's pages and should be confirmed by hand (TODO: confirm each question against its PDF).

Contradictory cases: two cases (CONTRA-1/2) put a false number inside the question. The pipeline verifies its own generated claims, not the question, so these do not test the verifier and were dropped from scoring. The verifier is evaluated directly in section 3.7.



The later end-to-end runs use 25 supported questions from the four filings for which a larger-chunk index was built (AMD, American Express, Boeing, Amcor).



2.4 Metrics

Gold page in context: the filing's evidence page appears among the chunks passed to the LLM. This is a strict proxy, because the same fact can appear on other pages.

Refused: overall\_status = insufficient\_evidence.

Empty reply: the model's reply was only a citation tag, so the claim text is empty.

Answer with real text, Verified (overall\_status = supported), Real text, rejected (answer text produced but not verified).

Proportions carry 95% Wilson confidence intervals. Rank comparisons use the Wilcoxon signed-rank test; paired hit/miss comparisons use an exact sign test.

3\. Experiments and results

3.1 Baseline: pipeline as shipped



44 cases (39 supported, 3 unsupported, 2 contradictory), all returned insufficient\_evidence. The right filing was among the retrieved chunks for 92% of extraction questions (11/12) and 81% of reasoning questions (22/27), but the right page for only 2/12 and 3/27.



Refusing all unsupported questions (3/3) is not evidence of good discrimination, because the system also refused all answerable ones.



3.2 Retrieval depth (n = 39, right page among top K)

Setting	@3	@5	@10	@20	@50	MRR

Vector search, all filings (as the pipeline does)	13%	18%	23%	31%	44%	0.11

Vector search, correct filing only	13%	23%	31%	44%	59%	0.14



Restricting to the correct filing helps modestly. Even with 50 candidates inside the correct filing, 41% of questions never surface their evidence page. Locating the right page inside a long filing is the main retrieval difficulty.



3.3 Reranker (cross-encoder), correct filing only, n = 39

Setting	@3	@5	@10	MRR

Vector only	13%	23%	31%	0.14

Retrieve 20, rerank	21%	28%	31%	0.20

Retrieve 50, rerank	21%	26%	31%	0.19



Paired tests for "retrieve 20, rerank" versus vector only:



Reciprocal rank: better on 8 questions, worse on 12, tied on 19 (Wilcoxon p = 0.723).

Hit@3: gained on 4, lost on 1 (exact sign test p = 0.375).



Conclusion: no significant benefit detected at n = 39. The reranker improves some questions substantially and degrades others. Note that reranking only 3 retrieved chunks (as the shipped pipeline does) can reorder them but cannot recover missing evidence.



3.4 Chunk size (4 filings, 25 supported questions, correct filing only)

Chunk size	Right page in top 3	Gold-number coverage in top 3 (n = 15)	Hit@3 vs 200 chars

200 chars (shipped)	16%	11%	–

600 chars	28%	28%	gained 4, lost 1 (p = 0.375)

1,000 chars	36%	28%	gained 5, lost 0 (p = 0.062)



The embedding model reads at most 256 word-pieces (max\_seq\_length = 256), roughly 1,000 characters of prose, so larger chunks would be partly ignored. Page recall favours larger chunks by construction (fewer chunks per page); gold-number coverage, the share of key figures from the gold evidence present in the top 3 chunks, is the fairer measure. Larger chunks also put more text in the prompt (about 2,800 characters for 1,000-character chunks), and the effect of that on a 3B model was not isolated.



3.5 Oracle run: correct evidence supplied (n = 39)



The pipeline was given the gold evidence passages (at most 2, truncated to 1,800 characters) instead of its own retrieval.



Gold page in context: 39/39.

Refused (insufficient\_evidence): 37/39. Answers produced: 2/39. Verified: 0/39.



With perfect retrieval the shipped pipeline still refused 95%. This isolates a problem after retrieval: the prompt, the model, or both.



3.6 Prompt comparison

Prompt	Description

A	The pipeline's own prompt ("policy-document question-answering assistant")

B	Reworded for 10-K filings, citations requested, numbers may be computed

C	B plus: every sentence must end with an evidence ID, at most three sentences, no commentary

D	C plus: never reply with only an evidence ID



Exact text: run\_best\_config.py (B), run\_prompt\_c.py (C), run\_prompt\_d.py (D), and llm\_probe.py (A and B).



Probe on gold evidence, 15 questions (12 extraction, 3 reasoning): prompt A refused 15/15; prompt B refused 10/15, answered 5, and of the answered questions with numbers, 1 of 3 matched the reference.



Prompts B, C and D were written and adjusted while looking at the same 25 questions, so later prompts have an advantage on them.



3.7 Verifier tested in isolation (verifier\_test.py)



Claims were built from FinanceBench evidence sentences (16 usable cases of 39) and checked against their passage plus 150 characters of context.



Claim type	n	Result

Verbatim sentence	16	16 supported

Same sentence, one number altered	16	13 contradicted, 3 not established, 0 supported

Sentence from a different company's filing	16	13 not established, 3 contradicted, 0 supported



This is a best-case test: a verbatim claim is trivially entailed, several alterations were obvious (for example December 71, 5022), and subtle numeric errors are likely harder. The verifier never accepted a false claim in this test.



3.8 End-to-end comparison (25 supported questions common to all runs)

Run (n=25 supported questions)	Gold page in context	Refused	Empty reply (tag only)	Answer with real text	Verified	Real text, rejected

Pipeline as shipped (200-char chunks, top-3)	4/25 (16%, CI 6-35%)	25/25 (100%, CI 87-100%)	0/25 (0%, CI 0-13%)	0/25 (0%, CI 0-13%)	0/25 (0%, CI 0-13%)	0/25 (0%, CI 0-13%)

Prompt B (1,000-char chunks, pool 20, rerank 3)	9/25 (36%, CI 20-55%)	18/25 (72%, CI 52-86%)	0/25 (0%, CI 0-13%)	7/25 (28%, CI 14-48%)	0/25 (0%, CI 0-13%)	7/25 (28%, CI 14-48%)

Prompt C	9/25 (36%, CI 20-55%)	16/25 (64%, CI 45-80%)	4/25 (16%, CI 6-35%)	5/25 (20%, CI 9-39%)	1/25 (4%, CI 1-20%)	4/25 (16%, CI 6-35%)

Prompt D	9/25 (36%, CI 20-55%)	12/25 (48%, CI 30-67%)	0/25 (0%, CI 0-13%)	13/25 (52%, CI 33-70%)	0/25 (0%, CI 0-13%)	13/25 (52%, CI 33-70%)



Unsupported questions: all 9 were refused under prompts C and D (CI 70-100%); the 2 questions common to all runs were refused in every run. Under prompt D the system answered 52% of supported questions and still refused all 9 unsupported ones, which is weak evidence that refusals are not purely indiscriminate (small n).



Repeatability: prompt C run twice gave the same overall\_status on 34/34 questions. An earlier prompt B configuration run twice differed on one question (20 versus 21 refusals), so one-question differences between runs are within noise. Only the status was compared, not the answer text.



What changed between the shipped run and the later runs: chunk size, retrieval pool and reranking, source filtering to the correct filing, and the prompt all changed together, so the improvement cannot be attributed to any single change.



3.9 Where answers are lost (failure funnel, prompt D, 25 supported questions)

9 refused with the evidence page not in context, and 3 refused with it in context.

13 answered, then rejected (7 without the gold page in context, 6 with it).

0 verified.

Claim-level: 12 uncertain (all not\_established by the verifier), 1 invalid\_citation. No claim was contradicted and none supported.



Under prompt C, four of the nine answered questions contained only a citation tag (for example \[E1]) with no answer text. The pipeline reports these as needs\_review, which hides that no answer was produced. Under prompt B, 14 of 21 claims were missing\_citation; under prompt C this fell to 2 of 11.



4\. Error analysis (hand labels)



TODO: label each answered question in evaluation/results/error\_analysis\_promptD.csv (column your\_label) as correct, partly correct, wrong, or off-topic, then complete the table and the counts below.



Question ID	Your label	Notes

financebench\_id\_01148	TODO	

financebench\_id\_00995	TODO	

financebench\_id\_01279	TODO	

financebench\_id\_00563	TODO	

financebench\_id\_00476	TODO	reference answer is "There are none"

financebench\_id\_01028	TODO	

financebench\_id\_01351	TODO	citation gate flagged an invalid evidence ID

financebench\_id\_01964	TODO	

financebench\_id\_00517	TODO	

financebench\_id\_01290	TODO	

financebench\_id\_00464	TODO	

financebench\_id\_00494	TODO	

financebench\_id\_00585	TODO	



Counts: correct TODO, partly correct TODO, wrong TODO, off-topic TODO.



Failure categories observed so far (to be confirmed against the labels):



Evidence page not retrieved.

Evidence split across 200-character chunks (for example, a table cut mid-row).

LLM refuses although the evidence is present (prompt A, and the 3B model more generally).

Reply with no answer text (a bare citation tag, prompt C).

Answer sentences without citations (prompt B).

Answer rejected by the verifier although it appears consistent with the source (TODO: confirm for each case).

Absence questions ("There are none") that the refusal text cannot distinguish from "not found".

Questions needing calculation or multi-passage reasoning, which a single-passage verifier cannot confirm.



TODO: state whether any wrong answer was marked supported by the verifier in any run. From the runs above, the only verified answer was financebench\_id\_01148 (prompt C), which should be checked by hand.



5\. Limitations

Small samples: 25 supported and 9 unsupported questions in the end-to-end runs (39 supported in the retrieval tests). Most confidence intervals are wide.

The end-to-end, chunk-size and prompt experiments use four filings, and retrieval was restricted to the correct filing, which the shipped pipeline does not do. Those retrieval numbers are upper bounds.

One embedding model (all-MiniLM-L6-v2), one LLM (qwen2.5:3b), one NLI verifier.

Prompts B, C and D were tuned on the same questions they were evaluated on.

Several factors changed at once between the shipped run and the later runs.

"Gold page in context" undercounts retrieval success when the same fact appears on other pages.

Numeric matching against reference answers is a rough heuristic; correctness judgements rely on hand labels by one person.

The verifier test is a best-case setting and does not cover subtle errors or paraphrases.

Only overall\_status was compared for repeatability.

RAGAS metrics (faithfulness, answer relevance, context precision) were not computed. A local 3B model would be an unreliable judge, and a hosted judge was not used. This is a gap against the original plan.

Chunk-size results cover three sizes only, with one overlap setting (20% of the chunk size).

6\. Recommendations for the pipeline



Each recommendation lists its supporting evidence and what remains untested.



Use larger chunks (about 1,000 characters). Hit@3 gained 5 and lost 0 versus 200 characters (p = 0.062). Do not go much larger without changing the embedding model, since it truncates at about 1,000 characters.

Revise the answer prompt for financial documents. The shipped prompt refused 95% of questions even with the correct evidence. Include a rule that a reply must contain answer text, not only a citation tag.

Handle empty or tag-only replies explicitly and report them as a generation failure instead of needs\_review.

Examine the verifier on paraphrased claims. It accepted all verbatim claims but nearly no paraphrased ones in real answers. Test it against gold evidence with human-written paraphrases before relaxing any threshold.

Do not adopt the reranker as an improvement yet. No significant effect was found at n = 39, and reranking 3 chunks out of 3 cannot recover missing evidence. If kept, rerank a larger pool.

Expose top\_k, chunk size and an optional source filter as parameters of generate\_answer, so evaluation can vary them without patching.

Remove the NLI DEBUG print statements in the hallucination detector.

Consider the citation gate: a correct answer (AmEx effective tax rate) was rejected because the model cited an evidence ID that was not in the context.

7\. Reproducing the results



See evaluation/README.md. Scripts are in evaluation/; raw results are in evaluation/results/, and evaluation/results/comparison.md is generated by compare\_runs.py.

