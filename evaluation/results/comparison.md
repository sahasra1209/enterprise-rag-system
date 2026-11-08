| Run (n=25 supported questions) | Gold page in context | Refused | Empty reply (tag only) | Answer with real text | Verified | Real text, rejected |
|---|---|---|---|---|---|---|
| Pipeline as shipped (200-char chunks, top-3) | 4/25 (16%, CI 6-35%) | 25/25 (100%, CI 87-100%) | 0/25 (0%, CI 0-13%) | 0/25 (0%, CI 0-13%) | 0/25 (0%, CI 0-13%) | 0/25 (0%, CI 0-13%) |
| Prompt B (1,000-char chunks, pool 20, rerank 3) | 9/25 (36%, CI 20-55%) | 18/25 (72%, CI 52-86%) | 0/25 (0%, CI 0-13%) | 7/25 (28%, CI 14-48%) | 0/25 (0%, CI 0-13%) | 7/25 (28%, CI 14-48%) |
| Prompt C | 9/25 (36%, CI 20-55%) | 16/25 (64%, CI 45-80%) | 4/25 (16%, CI 6-35%) | 5/25 (20%, CI 9-39%) | 1/25 (4%, CI 1-20%) | 4/25 (16%, CI 6-35%) |
| Prompt D | 9/25 (36%, CI 20-55%) | 12/25 (48%, CI 30-67%) | 0/25 (0%, CI 0-13%) | 13/25 (52%, CI 33-70%) | 0/25 (0%, CI 0-13%) | 13/25 (52%, CI 33-70%) |

| Run (unsupported, n=2 common) | Refused | Answered |
|---|---|---|
| Pipeline as shipped (200-char chunks, top-3) | 2/2 (100%, CI 34-100%) | 0 |
| Prompt B (1,000-char chunks, pool 20, rerank 3) | 2/2 (100%, CI 34-100%) | 0 |
| Prompt C | 2/2 (100%, CI 34-100%) | 0 |
| Prompt D | 2/2 (100%, CI 34-100%) | 0 |

Prompt C, all unsupported questions: refused 9/9 (100%, CI 70-100%)

Prompt D, all unsupported questions: refused 9/9 (100%, CI 70-100%)

Repeatability: prompt C run twice, same overall_status on 34/34 shared questions