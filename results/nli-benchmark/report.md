# Citation verification: lexical overlap vs NLI

38 constructed claims over official GDPR chunks (EN/DE/FR and cross-lingual). Label source: **benchmark**, single curator, not independently annotated.

| Verifier | Binary accuracy | False-accept rate | False-reject rate | 3-class accuracy |
|---|---|---|---|---|
| lexical-overlap-v1 | 0.53 | 0.56 | 0.40 | n/a |
| nli-entailment-v1 | 0.82 | 0.06 | 0.30 | 0.76 |

## By perturbation (binary accuracy)

| Perturbation | Cases | Lexical | NLI |
|---|---|---|---|
| cross_lingual | 3 | 0.00 | 0.67 |
| cross_lingual_negation | 1 | 1.00 | 1.00 |
| negation | 7 | 0.00 | 1.00 |
| number | 6 | 0.50 | 1.00 |
| paraphrase | 5 | 0.00 | 0.20 |
| restatement | 12 | 1.00 | 0.92 |
| unrelated | 4 | 1.00 | 0.75 |

False accepts, lexical: en-breach-negation, en-extension-number-word, en-free-negation, en-erasure-negation, de-breach-negation, de-extension-number-word, de-free-negation, fr-breach-negation, fr-extension-number-word, fr-free-negation

False accepts, NLI: fr-portability-unrelated

NLI model: `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` @ `0a71e92a985b6e1ad1828cf67ce9c459639c1dca`.
