# Improving Quotation Attribution in Literary Texts

This repository contains code for the experiments described in the paper [`Improving Automatic Quotation Attribution in Literary Novels`](https://aclanthology.org/2023.acl-short.64/) (ACL 2023 short paper).

Citation:
```
@inproceedings{vishnubhotla2023improving,
  title={Improving Automatic Quotation Attribution in Literary Novels},
  author={Vishnubhotla, Krishnapriya and Rudzicz, Frank and Hirst, Graeme and Hammond, Adam},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  pages={737--746},
  year={2023}
}
```


## Notes
The code for the various components of the attribution model (character identification, coreference resolution, speaker identification), as well as auxiliary data formatting and evaluation scritps, are not very well integrated as of now. I hope to convert this into a neater set of pipelines in the near future, but it is hold as I am working on other projects for now.

## Data
Source data for the PDNC dataset in `data/pdnc_source`. This data is the starting point for all further experiments.

We also split the set of annotated quotations in PDNC into train, dev, and test subsets for training and evaluating the speaker identification model (`data/train_splits`). We stratify the splits in two ways: (a) split the annotations from each novel (`random`), or (b) split the set of novels (`leave-x-out`), where the quotations from a novel can only be in one of the three splits. We use 5-fold cross-validation for the training experiments.

## Models
- The BookNLP model (pretrained) can be run on PDNC using the `pdnc_run_pipeline.py` scripts. Outputs will be stored in `booknlpen/pdnc_output` (pre-populated in this repo). These outputs are used in a few other places subsequently, particularly training on PDNC.

- The spacy coref model can be run using the `coref/run_spacy.py` script. Outputs will be stored in `coref/outputs/spacy`.

- To inspect BookNLP coreference clusters for one PDNC novel, run
  `python -m booknlpen.analyze_coref_clusters --novel TheSunAlsoRises` from the repository root.
  This reads the novel's existing `.entities` and `.tokens` outputs and writes
  `coref_resolution_summary.json` and `coref_clusters_for_review.csv` under
  `booknlpen/pdnc_output/<novel>/`. The CSV contains each cluster's resolution
  status, PDNC candidates, mention categories, and token-context snippets.
  `coref_llm_context_packages.json` contains one structured package for each
  unresolved cluster with a `PER` mention, including its mention contexts and
  locations, current matching candidates, and the PDNC character catalog.
  BookNLP must have been run first; this analysis script does not run the model
  and reuses the existing reading and matching functions in `create_book_corefs.py`.
  Add `--refresh-matches` to rebuild and overwrite `coref_matches.csv` and
  `booknlp_matched_ents.csv` before generating the report.
  To generate reports for all PDNC novels plus per-novel mean, median, quartiles,
  population standard deviation, and pooled totals/rates, run
  `python -m booknlpen.analyze_coref_clusters --all-novels`. The aggregate files
  are written under `booknlpen/pdnc_output/`.

- To analyze the clusters that BookNLP actually selected as quote speakers, run
  `python -m booknlpen.analyze_speaker_assignments --all-novels`. The script
  writes quote-level decisions, clusters used as speakers, and per-novel and
  pooled summaries under `booknlpen/pdnc_output/`. It also compares resolved
  cluster IDs with PDNC speaker annotations where quote spans overlap. This
  post-hoc name-matching comparison does not modify BookNLP's output.

- To build LLM context packages for unresolved quote-speaker assignments, run
  `python -m booknlpen.build_assignment_llm_context --all-novels`. The generator
  merges overlapping quote-context windows into bounded passages and preserves
  each quote assignment and its BookNLP cluster identity. Resolved assignments
  in the same passage are context-only anchors; the response schema expects one
  decision per unresolved assignment, not one decision per passage. Each package
  reports the expected passage/call count and decision count. Use
  `--max-passage-tokens` to change the passage merge limit and `--context-window`
  to change the surrounding token window. The merge limit is soft: a single
  long quote is kept intact and can make its passage exceed the configured size.
  For a capped pilot, add `--max-decisions-per-passage 10`. Oversized groups are
  split into numbered decision parts that retain the same shared passage text;
  each unresolved assignment is a target in exactly one part. The capped build
  writes separate `speaker_assignment_llm_context_packages_max10.json` and
  `speaker_assignment_llm_context_summary_max10.json` files, leaving the
  uncapped packages untouched.

- To run the grouped packages through a local Ollama server, start the Ollama
  container with its API port published to the host (usually `11434`; check with
  `podman port ollama-rocm 11434/tcp`), then run
  `python -m booknlpen.run_assignment_llm_ollama --novel PrideAndPrejudice`.
  The runner defaults to `qwen2.5:7b`, requests schema-constrained JSON, validates
  every response against the package IDs and allowed PDNC IDs. Its forced-choice
  policy requires one catalog character for every unresolved assignment; the
  model cannot abstain, and should express uncertainty through confidence. It
  saves results
  after each attempt so invalid generations and their token usage remain available
  for later review and an interrupted run can resume. Malformed or invalid
  responses are retried twice by default; use `--retries` to change this, and
  exhausted passages are recorded and skipped so later passages can continue.
  Use `--all-novels` for
  the full corpus, `--limit-passages 1` for a smoke test, `--host` or
  `OLLAMA_HOST` for a non-default API URL, and `--overwrite` to intentionally
  start a fresh result file for that model. Results are saved separately from the
  input package as `speaker_assignment_llm_ollama_results_<model>_forced_choice.json`;
  add `--max-decisions-per-passage 10` to run the matching capped package and
  write a distinct `..._forced_choice_max10.json` result file;
  earlier abstention-policy results remain separate and are not reused. Each
  passage records Ollama prompt/generated token counts and timing for later
  packaging-strategy comparisons, with aggregate token totals in the results
  file. The runner requires `expected_output_decision_count` and expects one
  response decision per unresolved assignment. It asks for a short evidence
  rationale, not hidden chain-of-thought.

- To compare a forced-choice run with the BookNLP assignments and PDNC speaker
  annotations, first generate the baseline quote report with
  `python -m booknlpen.analyze_speaker_assignments --novel PrideAndPrejudice`,
  then run
  `python -m booknlpen.analyze_assignment_llm_results --novel PrideAndPrejudice`.
  The comparison reports BookNLP accuracy on resolved assignments, LLM accuracy
  on its unresolved targets, and the combined pipeline's coverage and
  end-to-end accuracy over all quotes with gold speaker annotations. Missing
  LLM decisions in an incomplete run count as uncovered, not as correct. Token
  totals include invalid/retry attempts and are reported separately from valid
  attempts. The script writes a JSON summary and a quote-level CSV beside the
  novel outputs; use `--model`, `--results`, `--quote-report`, or `--output-dir`
  to select alternate inputs or destinations. Add
  `--max-decisions-per-passage 10` to compare the capped run; its reports also
  use a `_max10` suffix so they do not overwrite uncapped analysis.

- To measure how many decisions are bundled into each grouped LLM passage and
  relate package size to inference completion, retries, token cost, and accuracy,
  run
  `python -m booknlpen.analyze_assignment_package_sizes --novel PrideAndPrejudice`.
  The report gives an exact per-size distribution and grouped bands (1-4, 5-8,
  9-14, and 15+ decisions), with both passage-weighted and decision-weighted
  shares. It also writes passage-level details and, when Ollama results and the
  baseline quote report are available, reports completion, token, retry, and
  gold-accuracy metrics by size. This is observational analysis of the current
  run, not by itself proof that larger packages caused lower accuracy. Add
  `--max-decisions-per-passage 10` to analyze the capped package and matching
  inference results; its reports use a `_max10` suffix.

### Resultado do piloto com teto de 10 decisões

Em 6 de novembro de 2026, foi executada a comparação entre os pacotes agrupados
originais (sem teto de decisões) e os pacotes limitados a 10 decisões por
passagem. O escopo contém somente os seis livros para os quais há resultados
registrados nesta execução: *Pride and Prejudice*, *Alice in Wonderland*,
*A Room with a View*, *The Invisible Man*, *Daisy Miller* e *The Sign of the
Four*. *Anne of Green Gables* não foi incluído porque a inferência com teto 10
ainda não foi executada. Os demais livros também não fazem parte deste
recorte.

Ambas as variantes usam `qwen2.5:7b`, a política forced-choice e os mesmos
relatórios gold por livro. A comparação preserva a saída sem teto e grava os
artefatos da variante com sufixo `_max10`. Assim, o experimento compara duas
execuções do pipeline sobre os mesmos conjuntos de quotes; não é uma
reavaliação do BookNLP nem uma mudança na definição de gold. Os totais
agregados abaixo são microagregados: os numeradores e denominadores foram
somados entre livros antes do cálculo, em vez de tirar a média simples das
porcentagens de cada romance.

#### Resultado agregado

| Métrica nos seis livros | Sem teto | Teto 10 | Diferença |
|---|---:|---:|---:|
| Quotes no conjunto | 7.896 | 7.896 | — |
| Assignments destinados à LLM | 3.418 | 3.418 | — |
| Passagens/chamadas esperadas | 686 | 802 | +116 (+16,9%) |
| Assignments concluídos | 3.262 / 3.418 (95,4%) | 3.401 / 3.418 (99,5%) | +139 concluídos |
| Assignments ainda pendentes | 156 | 17 | -139 (-89,1%) |
| Passagens concluídas | 676 / 686 (98,5%) | 796 / 802 (99,3%) | +120 |
| Passagens sem resultado válido | 10 | 6 | -4 |
| Tentativas totais | 743 | 819 | +76 (+10,2%) |
| Tentativas com resposta inválida | 67 / 743 (9,0%) | 23 / 819 (2,8%) | -44; taxa -6,2 p.p. |
| Tokens totais (inclui retries) | 3.137.722 | 3.634.114 | +496.392 (+15,8%) |
| Tokens em tentativas inválidas | 395.023 | 122.585 | -272.438 (-69,0%) |

O denominador de **3.418 assignments** é idêntico nas duas variantes: o teto
altera a divisão em chamadas, não quais quotes são alvo da LLM. A acurácia dos
assignments-alvo entre os que têm predição aumentou de **62,86% (1.899/3.021)**
para **63,72% (2.009/3.153)**, avanço de **0,86 ponto percentual**. Ao incluir
no denominador todos os 3.167 assignments-alvo com gold, inclusive os que
permaneceram sem predição, a taxa de acerto subiu de **59,96%** para **63,44%**.

No pipeline combinado BookNLP + LLM, a acurácia entre quotes gold-comparáveis
com uma predição disponível passou de **74,05% (5.443/7.350)** para
**74,22% (5.553/7.482)**, aumento de apenas **0,16 p.p.**. Já a taxa de acerto
end-to-end, contando quotes sem predição como não acertadas, subiu de
**72,61% (5.443/7.496)** para **74,08% (5.553/7.496)**, ou **+1,47 p.p.**. A
cobertura do pipeline combinado aumentou de **98,05% (7.350/7.496)** para
**99,81% (7.482/7.496)**. Portanto, o ganho agregado mais claro do teto neste
retrato é a conclusão/cobertura, e não uma grande mudança na acurácia
condicional.

A acurácia do BookNLP entre as quotes gold-comparáveis nas quais ele produziu
uma predição foi **81,87% (3.544/4.329)** no conjunto desses livros. O pipeline
com teto 10 ficou **7,65 p.p. abaixo** desse valor (74,22%); sem teto, a
distância era **7,81 p.p.** (74,05%). O teto aproxima ligeiramente a acurácia
condicional agregada da referência do BookNLP, mas ainda não a iguala.

#### Comparação por livro

As acurácias abaixo usam a métrica `accuracy_when_predicted`: quotes sem
predição são excluídas do cálculo. “Acurácia LLM” é somente nos assignments
alvo da LLM; “pipeline combinado” usa a predição BookNLP para os assignments
que ele resolveu e a predição LLM para os alvos. O último campo da tabela
mostra a mudança do pipeline combinado com teto 10 em relação à execução sem
teto.

| Livro | BookNLP resolvido | LLM alvo sem teto → 10 | Combinado sem teto → 10 | Variação combinado |
|---|---:|---:|---:|---:|
| *Pride and Prejudice* | 67,75% | 66,67% → 68,14% | 67,06% → 68,00% | +0,94 p.p. |
| *Alice in Wonderland* | 97,24% | 66,07% → 65,48% | 92,20% → 92,11% | -0,10 p.p. |
| *A Room with a View* | 64,32% | 58,04% → 61,09% | 60,81% → 62,49% | +1,69 p.p. |
| *The Invisible Man* | 88,83% | 79,45% → 76,03% | 87,69% → 87,27% | -0,42 p.p. |
| *Daisy Miller* | 90,84% | 75,00% → 75,00% | 87,24% → 87,24% | 0,00 p.p. |
| *The Sign of the Four* | 75,68% | 54,40% → 50,76% | 64,72% → 62,47% | -2,25 p.p. |

O resultado não melhora uniformemente: o teto aumentou a acurácia combinada em
*Pride and Prejudice* e *A Room with a View*, manteve-a praticamente estável
em *Daisy Miller* e a reduziu nos outros três livros. Comparando a distância
absoluta entre a acurácia combinada e o BookNLP, o teto aproximou os resultados
em dois livros, afastou-os em três e deixou *Daisy Miller* essencialmente
empatado. Assim, a melhoria global de cobertura não deve ser apresentada como
uma melhoria universal de acurácia.

| Livro | Assignments concluídos sem teto → 10 | Pendentes sem teto → 10 | Passagens sem teto → 10 | Tentativas inválidas sem teto → 10 | Tokens sem teto → 10 |
|---|---:|---:|---:|---:|---:|
| *Pride and Prejudice* | 1.073/1.163 → 1.159/1.163 | 90 → 4 | 198 → 242 | 34 → 8 | 1.136.470 → 1.266.109 (+11,4%) |
| *Alice in Wonderland* | 219/224 → 219/224 | 5 → 5 | 59 → 63 | 5 → 4 | 242.755 → 268.932 (+10,8%) |
| *A Room with a View* | 1.155/1.179 → 1.178/1.179 | 24 → 1 | 194 → 245 | 18 → 3 | 1.002.825 → 1.259.081 (+25,6%) |
| *The Invisible Man* | 189/189 → 189/189 | 0 → 0 | 71 → 72 | 0 → 1 | 198.236 → 209.514 (+5,7%) |
| *Daisy Miller* | 175/176 → 175/176 | 1 → 1 | 61 → 62 | 4 → 3 | 151.557 → 154.826 (+2,2%) |
| *The Sign of the Four* | 451/487 → 481/487 | 36 → 6 | 103 → 118 | 6 → 4 | 405.879 → 475.652 (+17,2%) |

O teto reduziu o número de assignments pendentes em quatro dos seis livros;
*Alice in Wonderland* e *Daisy Miller* mantiveram a mesma quantidade pendente.
O maior ganho absoluto de conclusão ocorreu em *Pride and Prejudice*
(86 assignments adicionais concluídos), seguido por *The Sign of the Four*
(30) e *A Room with a View* (23). *The Invisible Man* já estava completo nas
duas execuções.

#### Custo e falhas

O teto gerou **116 chamadas esperadas a mais** para o mesmo conjunto de
assignments. O total de tokens cresceu **15,8%**, embora tenha caído o volume de
tokens associado a respostas inválidas. O número de respostas inválidas
diminuiu de 67 para 23, e sua proporção em relação às tentativas totais caiu de
9,0% para 2,8%. Isso é compatível com a hipótese de que solicitar menos
decisões por resposta facilita cumprir o formato exigido, mas não demonstra
causalidade: as tentativas foram feitas em execuções separadas e o modelo pode
variar entre chamadas. O custo adicional de contexto e a repetição do trecho
entre partes também ajudam a explicar o aumento de tokens.

Entre os assignments concluídos, o consumo médio agregado subiu de
aproximadamente **962 tokens por assignment** para **1.069**. Esse cálculo
divide todos os tokens registrados, inclusive retries, pelo número de
assignments concluídos; é um indicador operacional, não uma medida de tokens
da resposta isolada por assignment.

#### Interpretação e limitações

1. **O objetivo de aproximar a acurácia do BookNLP foi atendido apenas
   modestamente no agregado.** A acurácia combinada entre predições aumentou
   0,16 p.p. (74,05% → 74,22%), enquanto o BookNLP nos assignments resolvidos
   alcançou 81,87%. O intervalo entre os dois permanece em 7,65 p.p.
2. **O ganho mais forte é cobertura/conclusão.** Com teto 10, houve 139
   assignments concluídos a mais, 139 a menos pendentes e cobertura combinada
   de 99,81%, frente a 98,05% sem teto. Isso elevou a taxa end-to-end de acerto
   de 72,61% para 74,08%.
3. **Há heterogeneidade entre livros.** O pipeline combinado melhorou em dois,
   piorou em três e ficou estável em um. *The Sign of the Four* merece atenção:
   completou 30 assignments adicionais, mas sua acurácia combinada caiu
   2,25 p.p.; terminar mais decisões não garante que elas estejam corretas.
4. **A comparação de acurácia condicional não é perfeitamente pareada.**
   `accuracy_when_predicted` exclui quotes não cobertas, e o conjunto de
   predições disponíveis muda entre as execuções. O pipeline combinado é
   avaliado sobre o universo completo de quotes gold-comparáveis, mas a acurácia
   do BookNLP considera apenas as quotes para as quais ele já produziu uma
   predição resolvida. Esses valores são referências úteis, porém não medem
   exatamente o mesmo subconjunto.
5. **São snapshots de execuções independentes, não repetições controladas.**
   O conjunto de quotes e as anotações gold são os mesmos, mas cada variante
   fez chamadas próprias ao modelo. O resultado observado não mede variância
   entre repetições e, por si só, não prova que o teto causou as diferenças.
6. **A execução com teto ainda não está completa em cinco livros.** Somente
   *The Invisible Man* concluiu todos os assignments; restam 17 assignments
   pendentes nos demais livros. Os relatórios devem ser atualizados após
   retomar essas execuções, e o resultado de *Anne of Green Gables* deve ser
   acrescentado quando estiver disponível.

#### Arquivos de comparação

Cada relatório JSON contém as métricas e os caminhos dos arquivos de entrada
usados. Os arquivos abaixo permitem conferir lado a lado os resultados sem
teto e com teto 10:

| Livro | Sem teto | Teto 10 |
|---|---|---|
| *Pride and Prejudice* | [JSON](booknlpen/pdnc_output/PrideAndPrejudice/speaker_assignment_llm_comparison_qwen2.5_7b.json) | [JSON](booknlpen/pdnc_output/PrideAndPrejudice/speaker_assignment_llm_comparison_qwen2.5_7b_max10.json) |
| *Alice in Wonderland* | [JSON](booknlpen/pdnc_output/AliceInWonderland/speaker_assignment_llm_comparison_qwen2.5_7b.json) | [JSON](booknlpen/pdnc_output/AliceInWonderland/speaker_assignment_llm_comparison_qwen2.5_7b_max10.json) |
| *A Room with a View* | [JSON](booknlpen/pdnc_output/ARoomWithAView/speaker_assignment_llm_comparison_qwen2.5_7b.json) | [JSON](booknlpen/pdnc_output/ARoomWithAView/speaker_assignment_llm_comparison_qwen2.5_7b_max10.json) |
| *The Invisible Man* | [JSON](booknlpen/pdnc_output/TheInvisibleMan/speaker_assignment_llm_comparison_qwen2.5_7b.json) | [JSON](booknlpen/pdnc_output/TheInvisibleMan/speaker_assignment_llm_comparison_qwen2.5_7b_max10.json) |
| *Daisy Miller* | [JSON](booknlpen/pdnc_output/DaisyMiller/speaker_assignment_llm_comparison_qwen2.5_7b.json) | [JSON](booknlpen/pdnc_output/DaisyMiller/speaker_assignment_llm_comparison_qwen2.5_7b_max10.json) |
| *The Sign of the Four* | [JSON](booknlpen/pdnc_output/SignOfFour/speaker_assignment_llm_comparison_qwen2.5_7b.json) | [JSON](booknlpen/pdnc_output/SignOfFour/speaker_assignment_llm_comparison_qwen2.5_7b_max10.json) |

Para acrescentar *Anne of Green Gables* ou atualizar os resultados após novas
execuções, mantenha a mesma variante nos três passos (build, runner e análise):

```bash
uv run -m booknlpen.build_assignment_llm_context --novel AnneOfGreenGables --max-decisions-per-passage 10
uv run -m booknlpen.run_assignment_llm_ollama --novel AnneOfGreenGables --max-decisions-per-passage 10
uv run -m booknlpen.analyze_assignment_llm_results --novel AnneOfGreenGables --max-decisions-per-passage 10
```

- To reproduce the original per-assignment context package for later comparison,
  run `python -m booknlpen.build_assignment_llm_context_per_assignment --all-novels`.
  This writes separate `speaker_assignment_llm_context_packages_per_assignment.json`
  files and does not overwrite the grouped packages. The original format stores
  one quote/mention context entry per unresolved assignment, grouped under its
  unresolved speaker cluster; its response schema expects cluster-level
  decisions. Token-usage and accuracy comparisons between packaging strategies
  are not computed by either generator.

- GutenTag is currently not integrated in these scripts. It needs to be run as a standalone application (http://www.cs.toronto.edu/~jbrooke/gutentag/), and only works with English corpora from GutenTag. I have included its outputs for the set of novels in PDNC in the `gutentag/data` folder. I would love to port the Character Identification algorithm from GutenTag to a more accessible version at some point.

- Data and scripts for training the BookNLP speaker identification model on PDNC are the `training` folder. Data is formatted in the required format in the `training/data/pdnc` folder, for both the `random` and `leave-x-out` splits.
The `train_speaker.py` scripts can be run to train and save the model in `training/trained_models`. Predictions on the dev and test sets are also saved in the same folder.

## Evaluation
The `eval_functions.py` script is a repository of functions that evaluate the above models on character identification (gutentag, spacy, booknlp), coreference resolution (spacy, booknlp), and speaker identification (booknlp). 
[I have added comments indicating what each function is evaluating but this is the prime candidate to start the clean-up process.]


### Contact
The primary author can be contacted via email: vkpriya@cs.toronto.edu


## Proximos passos
Arrumar os clusters de um romance e verificar o ganho final do pipeline
- Comparando os corefs (BookNLP vs LLM)
- Observando a acurácia da atribuição final do pipeline.