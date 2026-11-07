# Relatório: tamanho dos pacotes e inferência de speakers com LLM

**Data do retrato dos resultados:** 5 de novembro de 2026  
**Modelo:** `qwen2.5:7b`  
**Política:** forced-choice; a LLM precisa escolher um `pdncID` válido para cada assignment pendente.  
**Novelas incluídas:** *A Room with a View*, *Alice in Wonderland*, *Anne of Green Gables*, *Daisy Miller*, *Pride and Prejudice*, *The Sign of the Four* e *The Invisible Man*.  
**Novela excluída:** *The Age of Innocence*. A inferência não foi finalizada e, conforme solicitado, seus dados não entram em nenhuma soma ou conclusão deste relatório.

## Resumo executivo

Os dados dão suporte à hipótese de que passagens com muitos assignments são mais difíceis para esta configuração da LLM, mas não demonstram que o tamanho, sozinho, cause a queda. O sinal mais claro aparece a partir de **15 decisões por pacote**:

- Há **78 pacotes com 15 ou mais decisões** (8,5% dos pacotes), que reúnem **1.641 assignments** (34,5% do total).
- Nesse grupo, a conclusão foi **92,4%**, a acurácia entre predições avaliáveis foi **60,7%**, e a taxa de acerto contando também as decisões gold pendentes foi **56,2%**.
- Houve **40 respostas inválidas em 113 tentativas** para essa faixa (35,4%); cinco pacotes 15+ terminaram sem resposta válida, deixando 125 decisões pendentes.
- Nos pacotes com até 15 decisões, a acurácia entre predições avaliáveis foi **73,1%**, e a taxa de acerto sobre os assignments gold foi **71,6%**.
- Um teto de 15 — que divide apenas pacotes com **mais de** 15 decisões — afetaria 67 pacotes e exigiria uma estimativa de **71 chamadas adicionais** para processar os mesmos assignments em blocos de no máximo 15.
- Um teto de 10 afetaria 121 pacotes e exigiria cerca de **164 chamadas adicionais**. Um teto de 5 afetaria 264 pacotes e exigiria cerca de **511 chamadas adicionais**.

**Recomendação provisória:** se for preciso escolher um primeiro experimento, testar um teto de **15 decisões** parece o compromisso mais proporcional entre o sinal observado e o aumento estimado de chamadas. Os dados também justificam comparar 10 como segunda condição. Não recomendo transformar qualquer teto em regra definitiva antes de uma comparação pareada, pois os resultados variam por novela, várias execuções continuam parciais e dividir uma passagem pode repetir contexto e alterar as inferências.

## 1. Escopo e estado das execuções

Foram encontrados resultados de inferência e relatórios de tamanho de pacote para sete novelas. Os sete relatórios foram atualizados a partir dos pacotes, resultados do Ollama e arquivos de quote-level disponíveis no repositório, para que os números abaixo reflitam o mesmo retrato.

| Novela | Passagens | Assignments-alvo | Decisões concluídas | Conclusão | Passagens sem resultado concluído |
|---|---:|---:|---:|---:|---:|
| *A Room with a View* | 194 | 1.179 | 1.155 | 98,0% | 2 |
| *Alice in Wonderland* | 59 | 224 | 219 | 97,8% | 1 |
| *Anne of Green Gables* | 236 | 1.338 | 1.306 | 97,6% | 2 |
| *Daisy Miller* | 61 | 176 | 175 | 99,4% | 1 |
| *Pride and Prejudice* | 198 | 1.163 | 1.073 | 92,3% | 4 |
| *The Sign of the Four* | 103 | 487 | 451 | 92,6% | 2 |
| *The Invisible Man* | 71 | 189 | 189 | 100,0% | 0 |
| **Total** | **922** | **4.756** | **4.568** | **96,0%** | **12** |

Somente *The Invisible Man* está completa. As demais seis novelas têm **188 decisões pendentes** no conjunto. O histórico registra **993 tentativas de geração**: 910 válidas e 83 inválidas/retries. As tentativas acumularam **4.581.514 tokens** (4.005.906 de prompt e 575.608 gerados). Desses, 497.223 tokens foram consumidos em tentativas inválidas. O tempo agregado informado pela API é aproximadamente **3 h 23 min**; é a soma dos tempos de geração registrados, não uma estimativa de tempo de processamento em paralelo.

### Falhas de resposta e retries

“Inválida” aqui significa que houve uma resposta do modelo, mas ela não passou pela validação do runner (por exemplo, quantidade incorreta de decisões ou IDs inconsistentes). Esta contagem **não** inclui falhas de conexão/API; o runner interrompe a execução se uma exceção de transporte subir antes da validação.

O runner usa `--retries 2` por padrão: uma chamada inicial e até duas tentativas adicionais. Quando uma resposta falha na validação, o erro é acrescentado ao prompt da próxima tentativa. O histórico observado é cumulativo, então uma passagem pode ter mais de três tentativas se uma execução posterior foi retomada; nos dados atuais seis passagens têm seis tentativas registradas. Assim, “duas tentativas adicionais” é o limite por execução, não necessariamente o limite de todo o histórico.

| Indicador | Resultado observado |
|---|---:|
| Passagens tentadas | 922 |
| Respostas válidas | 910 |
| Respostas inválidas | 83 / 993 tentativas (8,4%) |
| Passagens com pelo menos uma resposta inválida | 36 / 922 (3,9%) |
| Passagens recuperadas após erro (terminaram com resposta válida) | 24 / 36 (66,7%) |
| Passagens ainda sem resposta válida | 12 / 922 (1,3%) |
| Decisões pendentes dentro dessas 12 passagens | 188 |
| Tokens gastos em respostas inválidas | 497.223 / 4.581.514 (10,9%) |

O item “tentativas inválidas recuperadas” deve ser interpretado por passagem: 24 passagens que tiveram pelo menos uma resposta inválida posteriormente obtiveram uma resposta válida. Não significa que cada uma das 83 tentativas inválidas foi seguida diretamente por sucesso.

As **993 tentativas acumuladas** distribuem-se assim por passagem: 886 passagens tiveram uma tentativa; 20 tiveram duas; 9 tiveram três; 1 teve quatro; e 6 tiveram seis. As seis com seis tentativas ultrapassaram o máximo padrão por passagem de uma única execução porque o runner preserva o histórico e novas execuções podem retomar os casos ainda sem decisão válida.

#### Tipo de erro encontrado

Os 83 erros de validação foram categorizados pelo erro registrado pelo runner:

| Motivo da rejeição | Tentativas | % dos erros inválidos | Exemplos do erro |
|---|---:|---:|---|
| Número de decisões diferente do esperado | 61 | 73,5% | Retornou menos ou mais decisões que o número de assignments-alvo da passagem |
| IDs/identidade de assignment ou cluster inconsistentes | 18 | 21,7% | `assignment_id`, `cluster_key` ou `COREF` inesperado/incorreto; IDs duplicados também entram nesta classe |
| Campo inválido ou personagem fora do catálogo | 4 | 4,8% | Confiança inválida ou `pdncID` que não pertence ao catálogo permitido |
| **Total** | **83** | **100%** | |

O problema predominante é, portanto, a LLM não respeitar exatamente a cardinalidade e a identidade das decisões exigidas, e não apenas escolher um speaker incorreto. Como o protocolo valida o pacote inteiro, uma única decisão ausente, extra ou com ID incompatível invalida a resposta completa daquela passagem; as decisões parciais daquela geração não entram no resultado final.

#### Falhas e recuperação por tamanho do pacote

| Decisões por pacote | Passagens tentadas | Passagens com resposta inválida | Recuperadas após retry | Ainda sem resposta válida | Decisões pendentes | Tentativas | Inválidas | Taxa de tentativas inválidas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–4 | 599 | 4 | 3 | 1 | 1 | 604 | 6 | 1,0% |
| 5–8 | 166 | 3 | 1 | 2 | 11 | 171 | 7 | 4,1% |
| 9–14 | 79 | 12 | 8 | 4 | 51 | 105 | 30 | 28,6% |
| **15+** | **78** | **17** | **12** | **5** | **125** | **113** | **40** | **35,4%** |
| **Total** | **922** | **36** | **24** | **12** | **188** | **993** | **83** | **8,4%** |

Os grupos 9–14 e 15+ contêm **17,0% das passagens** (157/922), mas concentram **70/83 tentativas inválidas (84,3%)**. No grupo 15+, 17/78 pacotes (21,8%) tiveram pelo menos uma resposta inválida. Desses 17, doze (70,6%) acabaram recuperados e cinco (29,4%) continuam sem resposta válida. Esses cinco pacotes respondem por **125/188 (66,5%) das decisões ainda pendentes**.

#### Retries por novela

| Novela | Tentativas | Inválidas | Taxa inválida | Pacotes com erro | Recuperados | Sem resposta válida | Decisões pendentes nesses pacotes |
|---|---:|---:|---:|---:|---:|---:|---:|
| *A Room with a View* | 210 | 18 | 8,6% | 8 | 6 | 2 | 24 |
| *Alice in Wonderland* | 63 | 5 | 7,9% | 3 | 2 | 1 | 5 |
| *Anne of Green Gables* | 250 | 16 | 6,4% | 10 | 8 | 2 | 32 |
| *Daisy Miller* | 64 | 4 | 6,3% | 2 | 1 | 1 | 1 |
| *Pride and Prejudice* | 228 | 34 | 14,9% | 11 | 7 | 4 | 90 |
| *The Sign of the Four* | 107 | 6 | 5,6% | 2 | 0 | 2 | 36 |
| *The Invisible Man* | 71 | 0 | 0,0% | 0 | 0 | 0 | 0 |
| **Total** | **993** | **83** | **8,4%** | **36** | **24** | **12** | **188** |

O status “passagem sem resultado concluído” não significa necessariamente perda permanente: o runner conserva resultados e pode retomar passagens pendentes. Os dados de acurácia e custo são, portanto, um retrato parcial da inferência.

## 2. Distribuição de decisões por pacote

Os intervalos abaixo são **faixas mutuamente exclusivas**. “15+” significa **15 ou mais**, inclusive pacotes de exatamente 15 decisões.

| Decisões no pacote | Pacotes | % dos pacotes | Assignments | % dos assignments | Conclusão de assignments | Acurácia nas predições gold concluídas | Acertos / todos os assignments gold | Tentativas inválidas | Tokens | Tokens / assignment esperado |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–4 | 599 | 65,0% | 1.227 | 25,8% | 99,9% | 74,3% (796/1.072) | 74,2% (796/1.073) | 6 | 2.159.969 | 1.760 |
| 5–8 | 166 | 18,0% | 1.025 | 21,6% | 98,9% | 70,6% (670/949) | 70,0% (670/957) | 7 | 927.914 | 905 |
| 9–14 | 79 | 8,6% | 863 | 18,1% | 94,1% | 74,5% (574/770) | 69,9% (574/821) | 30 | 716.539 | 830 |
| **15+** | **78** | **8,5%** | **1.641** | **34,5%** | **92,4%** | **60,7% (890/1.467)** | **56,2% (890/1.585)** | **40** | **777.092** | **474** |
| **Total** | **922** | **100%** | **4.756** | **100%** | **96,0%** | **—** | **—** | **83** | **4.581.514** | **963** |

Alguns pontos importantes:

1. Os pacotes 15+ representam só 8,5% das passagens, mas contêm mais de um terço dos assignments.
2. A acurácia de 60,7% no grupo 15+ considera apenas assignments que têm speaker gold e predição concluída. Ao incluir os assignments gold que ainda estão pendentes no denominador, a taxa de acerto observada cai para 56,2%.
3. Pacotes 9–14 também apresentam uma taxa alta de tentativas inválidas: 30/105 tentativas (28,6%). No grupo 15+, foram 40/113 (35,4%). Juntos, esses dois grupos concentram 70 das 83 tentativas inválidas.
4. Pacotes maiores são mais eficientes em tokens por assignment na execução observada, pois compartilham o contexto textual entre várias decisões. Isso não significa necessariamente menor custo se forem divididos: subpacotes podem repetir o texto de contexto.
5. Os pacotes 9–14 e 15+ são 17,0% das passagens, mas acumulam 84,3% das respostas inválidas. Isso torna o padrão de falhas um indício adicional a favor de testar um limite, sem provar que dividir os pacotes resolveria o problema.

## 3. Interpretação de “limite máximo”

Uma **faixa 15+** não é exatamente o mesmo que impor um **teto de 15**:

- A faixa **15+** inclui pacotes com exatamente 15 decisões.
- Um teto de **15** só precisa dividir pacotes com **mais de 15** decisões.

Há **11 pacotes com exatamente 15 decisões**, somando 165 assignments. Por isso, a comparação de desempenho “15+” usa 78 pacotes/1.641 decisions; já a simulação de teto 15 usa 67 pacotes/1.476 decisions.

Para estimar chamadas adicionais, considerei dividir cada pacote acima do teto em `ceil(decisões / teto)` partes. A estimativa supõe uma chamada válida por parte e não inclui retries futuros. Também não prediz o custo em tokens, pois não mede a repetição do contexto nem possíveis mudanças na resposta da LLM.

## 4. Comparação dos cenários de teto

As linhas “até o limite” e “acima do limite” são grupos observados na execução atual, **não resultados de uma reexecução com os pacotes efetivamente divididos**.

| Teto hipotético | Grupo atual | Pacotes | Assignments | Conclusão | Acurácia nas predições gold concluídas | Acertos / todos os assignments gold | Tentativas inválidas | Tokens observados |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **5** | Até 5 | 658 (71,4%) | 1.522 (32,0%) | 99,6% | 71,7% (961/1.340) | 71,4% (961/1.345) | 9 | 2.456.493 |
| **5** | Acima de 5 | 264 (28,6%) | 3.234 (68,0%) | 94,4% | 67,5% (1.969/2.918) | 63,7% (1.969/3.091) | 74 | 2.125.021 |
| **10** | Até 10 | 801 (86,9%) | 2.592 (54,5%) | 99,2% | 71,8% (1.676/2.335) | 71,2% (1.676/2.354) | 19 | 3.365.253 |
| **10** | Acima de 10 | 121 (13,1%) | 2.164 (45,5%) | 92,3% | 65,2% (1.254/1.923) | 60,2% (1.254/2.082) | 64 | 1.216.261 |
| **15** | Até 15 | 855 (92,7%) | 3.280 (69,0%) | 98,1% | 73,1% (2.156/2.951) | 71,6% (2.156/3.011) | 46 | 3.912.669 |
| **15** | Acima de 15 | 67 (7,3%) | 1.476 (31,0%) | 91,5% | 59,2% (774/1.307) | 54,3% (774/1.425) | 37 | 668.845 |

### Chamadas estimadas para impor cada teto

| Teto | Pacotes atuais acima do teto | Assignments nesses pacotes | Chamadas atuais esperadas | Chamadas estimadas depois da divisão | Chamadas adicionais estimadas | Aumento sobre chamadas esperadas |
|---:|---:|---:|---:|---:|---:|---:|
| **5** | 264 | 3.234 | 922 | 1.433 | **511** | **55,5%** |
| **10** | 121 | 2.164 | 922 | 1.086 | **164** | **17,8%** |
| **15** | 67 | 1.476 | 922 | 993 | **71** | **7,7%** |

“Chamadas atuais esperadas” é uma chamada por cada uma das 922 passagens originais. A execução real fez 993 tentativas por causa de retries e resultados incompletos; as chamadas estimadas acima não tentam prever retries futuros.

### Chamadas adicionais estimadas por novela

| Novela | Teto 5 | Teto 10 | Teto 15 |
|---|---:|---:|---:|
| *A Room with a View* | 140 | 51 | 22 |
| *Alice in Wonderland* | 18 | 4 | 3 |
| *Anne of Green Gables* | 153 | 48 | 22 |
| *Daisy Miller* | 7 | 1 | 0 |
| *Pride and Prejudice* | 135 | 44 | 20 |
| *The Sign of the Four* | 50 | 15 | 4 |
| *The Invisible Man* | 8 | 1 | 0 |
| **Total** | **511** | **164** | **71** |

## 5. Detalhe do grupo 15+ por novela

“Acurácia” é calculada somente nos assignments com anotação gold e decisão concluída. “Acerto / gold” inclui no denominador todos os assignments com anotação gold, incluindo os que ficaram pendentes.

| Novela | Pacotes 15+ | Assignments | Conclusão dos assignments | Acurácia nas decisões gold concluídas | Acertos / todos os gold | Tentativas inválidas | Tokens observados | Chamadas adicionais estimadas com teto 15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *A Room with a View* | 25 | 533 | 100,0% | 52,2% (273/523) | 52,2% (273/523) | 4 | 216.973 | 22 |
| *Alice in Wonderland* | 3 | 59 | 100,0% | 20,5% (8/39) | 20,5% (8/39) | 1 | 30.507 | 3 |
| *Anne of Green Gables* | 23 | 461 | 95,9% | 73,0% (317/434) | 70,0% (317/453) | 6 | 167.125 | 22 |
| *Daisy Miller* | 0 | 0 | — | — | — | 0 | 0 | 0 |
| *Pride and Prejudice* | 21 | 466 | 83,7% | 63,9% (242/379) | 53,7% (242/451) | 26 | 293.714 | 20 |
| *The Sign of the Four* | 6 | 122 | 75,4% | 54,3% (50/92) | 42,0% (50/119) | 3 | 68.773 | 4 |
| *The Invisible Man* | 0 | 0 | — | — | — | 0 | 0 | 0 |
| **Total** | **78** | **1.641** | **92,4%** | **60,7% (890/1.467)** | **56,2% (890/1.585)** | **40** | **777.092** | **71** |

O grupo não é homogêneo. *Anne of Green Gables* tem acurácia observada de 73,0% nos pacotes 15+, enquanto *Alice in Wonderland* tem 20,5%, mas esta última estimativa vem de apenas três pacotes e 39 decisões gold concluídas. *Pride and Prejudice* e *The Sign of the Four* também têm proporção maior de decisões 15+ pendentes, o que reduz a taxa “acertos / todos os gold”.

## 6. Ganho agregado do pipeline nas sete novelas

A avaliação abaixo compara o BookNLP com o pipeline combinado sobre **o mesmo conjunto de quotes com speaker gold**. Uma predição é correta se seu `pdncID` aparece no conjunto de speakers gold da quote.

| Métrica agregada | BookNLP resolvido | BookNLP + LLM |
|---|---:|---:|
| Gold-comparáveis | 9.235 | 9.235 |
| Quotes com predição | 4.799 (52,0%) | 9.057 (98,1%) |
| Predições corretas | 3.947 | 6.877 |
| Acurácia entre as quotes com predição | 82,2% | 75,9% |
| Acertos sobre todas as quotes gold-comparáveis | 42,7% | 74,5% |

O pipeline combinado acrescentou **2.930 acertos** no conjunto avaliado e aumentou a taxa sobre todos os gold-comparáveis em **31,7 pontos percentuais**. Ao mesmo tempo, a acurácia apenas entre quotes que receberam predição caiu de 82,2% para 75,9%: a LLM acrescenta predições para assignments originalmente não resolvidos, que são casos mais difíceis. Assim, a melhora principal é de cobertura e de acertos totais, não uma melhora da precisão de cada predição.

| Novela | BookNLP: acertos / gold | Combinado: acertos / gold | Ganho |
|---|---:|---:|---:|
| *A Room with a View* | 28,0% | 60,1% | **+32,1 p.p.** |
| *Alice in Wonderland* | 81,2% | 91,9% | **+10,6 p.p.** |
| *Anne of Green Gables* | 23,2% | 82,5% | **+59,3 p.p.** |
| *Daisy Miller* | 70,1% | 87,1% | **+17,0 p.p.** |
| *Pride and Prejudice* | 23,3% | 63,7% | **+40,3 p.p.** |
| *The Sign of the Four* | 35,4% | 62,4% | **+27,0 p.p.** |
| *The Invisible Man* | 78,0% | 87,7% | **+9,7 p.p.** |

O ganho foi positivo nas sete novelas observadas, mas variou de **+9,7 a +59,3 p.p.** A média simples dos ganhos por novela é **+28,0 p.p.** A soma ponderada por todas as quotes gold-comparáveis dá +31,7 p.p.; são agregações diferentes, ambas apresentadas para evitar que uma novela grande ou pequena esconda a variação entre obras.

## 7. O que os dados permitem concluir sobre os limites

### Teto 5

- É o cenário que mais altera os pacotes: 28,6% das passagens, contendo 68,0% dos assignments.
- Nas passagens acima de 5, a acurácia observada é 67,5%, contra 71,7% até 5; a diferença é de 4,2 p.p.
- Exigiria aproximadamente 511 chamadas adicionais antes de retries.
- A diferença de desempenho é relativamente modesta frente ao impacto operacional estimado. Os dados atuais não justificam começar por um teto global de 5.

### Teto 10

- Afeta 13,1% das passagens e 45,5% dos assignments.
- A acurácia observada é 65,2% acima de 10 e 71,8% até 10; diferença de 6,6 p.p.
- O grupo acima de 10 tem 92,3% de conclusão, comparado a 99,2% até 10, e concentra 64 das 83 tentativas inválidas.
- A divisão estimada acrescenta 164 chamadas. É uma alternativa intermediária plausível, mas ainda muda quase metade dos assignments e pode duplicar contexto.

### Teto 15

- Afeta 7,3% das passagens e 31,0% dos assignments.
- A acurácia observada acima de 15 é 59,2%, contra 73,1% até 15; diferença de 13,9 p.p. Na taxa sobre todos os assignments gold, a diferença é 17,3 p.p.
- O grupo acima de 15 tem 91,5% de conclusão, contra 98,1% até 15, e concentra 37 das 83 tentativas inválidas.
- A faixa descritiva 15+ teve 40 respostas inválidas em 113 tentativas, e cinco passagens desse grupo (125 decisões) continuam sem resultado válido.
- A estimativa é de 71 chamadas adicionais, o menor acréscimo entre os limites testados que ainda isola o grupo de pior desempenho agregado.

**Leitura provisória:** o teto 15 apresenta o sinal mais forte de risco em pacotes grandes com um aumento estimado de chamadas menor que os cenários 10 e 5. Isso o torna um bom primeiro experimento, não uma decisão já validada. O tamanho do pacote pode estar correlacionado com extensão do texto, número de clusters, distribuição de personagens, densidade de fala e dificuldade inerente à novela. Os grupos de tamanho também têm composição diferente entre novelas; por exemplo, *Daisy Miller* e *The Invisible Man* não têm pacotes 15+.

## 8. Limitações e proposta de experimento

1. **Inferência parcial:** seis das sete novelas estão incompletas; há 188 assignments pendentes. A acurácia nos grupos grandes pode mudar quando essas passagens forem concluídas. O denominador “acertos / todos os gold” torna explícita a falta de cobertura atual, mas não substitui os resultados finais.
2. **Associação não é causalidade:** as métricas agrupam saídas já produzidas; elas não medem o que a mesma LLM faria se a mesma passagem fosse dividida. Um grupo maior pode simplesmente conter trechos ou casos mais difíceis.
3. **Variação entre novelas:** o grupo 15+ varia de 20,5% a 73,0% nas obras com dados, e algumas amostras são pequenas. A acurácia agregada não deve ser interpretada como uma taxa universal.
4. **Custo da divisão:** o número de chamadas é uma estimativa estrutural. Não estima o total de tokens após dividir, porque o texto de contexto pode ser repetido em cada subpacote; também não prevê retries.
5. **Mesmo contexto, decisões repartidas:** para o teste ser informativo, manter o trecho textual e as âncoras de contexto originais, repartindo apenas a lista de assignments-alvo em blocos de até 15 ou 10. Se também for reduzido o contexto textual, o experimento muda duas variáveis ao mesmo tempo.
6. **Comparação pareada:** usar os mesmos assignments grandes, modelo, política forced-choice, temperatura, `num_ctx` e prompt. Comparar por assignment o resultado atual e o resultado dividido; registrar chamadas, tokens prompt/gerados, retries, cobertura, acurácia por decisão gold e resultado combinado end-to-end.
7. **Manter as faixas comparáveis:** reportar separadamente os casos 15–19, 20–29 e 30+ se houver amostra suficiente. Um único agregado 15+ combina pacotes de 15 até 41 decisões.

### Decisão sugerida

Com os dados atuais, **não mudar todos os packages para teto 5**. Fazer primeiro um piloto pareado com teto 15 para os pacotes acima de 15, mantendo todo o contexto textual. Se a acurácia ou taxa de resposta melhorar de modo consistente sem aumento excessivo de tokens, comparar o mesmo conjunto com teto 10. O teto 5 só parece justificável se o experimento mostrar benefício incremental claro frente aos custos de chamadas e repetição do contexto.

## 9. Arquivos de origem

Os resumos abaixo foram regenerados para as sete novelas incluídas. Cada pasta também contém um CSV com métricas por passagem e outro agrupado por quantidade de decisões.

- [Análise de tamanho de pacote — Pride and Prejudice](booknlpen/pdnc_output/PrideAndPrejudice/speaker_assignment_package_size_analysis_qwen2.5_7b.json)
- [Análise de tamanho de pacote — Sign of Four](booknlpen/pdnc_output/SignOfFour/speaker_assignment_package_size_analysis_qwen2.5_7b.json)
- [Análise de tamanho de pacote — Anne of Green Gables](booknlpen/pdnc_output/AnneOfGreenGables/speaker_assignment_package_size_analysis_qwen2.5_7b.json)
- [Análise de tamanho de pacote — A Room with a View](booknlpen/pdnc_output/ARoomWithAView/speaker_assignment_package_size_analysis_qwen2.5_7b.json)
- [Análise de tamanho de pacote — Alice in Wonderland](booknlpen/pdnc_output/AliceInWonderland/speaker_assignment_package_size_analysis_qwen2.5_7b.json)
- [Análise de tamanho de pacote — Daisy Miller](booknlpen/pdnc_output/DaisyMiller/speaker_assignment_package_size_analysis_qwen2.5_7b.json)
- [Análise de tamanho de pacote — The Invisible Man](booknlpen/pdnc_output/TheInvisibleMan/speaker_assignment_package_size_analysis_qwen2.5_7b.json)
- [Script de análise de tamanho e desempenho](booknlpen/analyze_assignment_package_sizes.py)
- [Script de comparação BookNLP + LLM](booknlpen/analyze_assignment_llm_results.py)

## 10. Definições das métricas

- **Assignment/decisão:** uma quote pendente para a qual o package exige uma decisão de speaker.
- **Conclusão:** número de assignments em passagens com resposta válida dividido pelo número esperado de assignments. Como a validação é por passagem, uma passagem incompleta não contribui decisões concluídas.
- **Acurácia nas predições gold concluídas:** decisões corretas divididas pelas decisões concluídas que têm um speaker gold comparável.
- **Acertos / todos os assignments gold:** decisões corretas divididas por todos os assignments com gold comparável naquela faixa, incluindo os pendentes no denominador.
- **Acurácia BookNLP / combinado:** acertos divididos por quotes gold-comparáveis em que o respectivo sistema produziu uma predição.
- **Taxa de acerto end-to-end:** acertos divididos por todas as quotes gold-comparáveis; quotes sem predição ou sem decisão correta não aumentam o numerador.
- **Tentativa inválida:** geração que falhou na validação e foi registrada para retry. Não é sinônimo de decisão semanticamente errada.
- **15+:** faixa descritiva de pacotes com 15 ou mais decisões.
- **Teto 15:** cenário hipotético que divide apenas pacotes com mais de 15 decisões, pois pacotes de exatamente 15 já respeitam o limite.
- **Chamadas adicionais:** soma de `ceil(n / teto) - 1` para cada pacote com `n > teto`; estima chamadas novas antes de retries, não tokens.
