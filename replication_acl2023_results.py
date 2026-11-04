import os
import csv
import sys
from contextlib import redirect_stdout
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

from booknlpen.common.pipelines import Token
from booknlpen.english.bert_qa import QuotationAttribution
import booknlpen.create_book_corefs as create_book_corefs
import pdnc_utils

HOME = str(Path.home())
# MODEL_FILE = os.path.join(
#     HOME, "booknlp_models", "speaker_google_bert_uncased_L-8_H-768_A-12-v1.0.model"
# )
MODEL_FILE = os.path.join(pdnc_utils.REPO_ROOT, "training", "trained_models", "lxo_split_0_uncased", "speaker_bert-base-uncased.model")
OUT_ROOT = "booknlpen/pdnc_output"
DATA_ROOT = "data/pdnc_source"
PERSON_CANDIDATE_TYPES = {"PRON_PER", "PROP_PER", "NOM_PER"}
# The merged CSV uses the canonical source names produced by the pipeline:
# 'pdnc', 'exp', and 'booknlp'.
CANDIDATE_PRIORITY = {"pdnc": 3, "exp": 2, "booknlp": 1}


def load_tokens_from_checkpoint(novel):
    tok_path = os.path.join(OUT_ROOT, novel, f"{novel}.beforeQA.tokens")
    tok_df = pd.read_csv(tok_path, sep="\t", keep_default_na=False, quoting=3)
    tokens = []
    for _, r in tok_df.iterrows():
        word = str(r["word"]) if pd.notna(r["word"]) else ""
        lemma = str(r["lemma"]) if pd.notna(r["lemma"]) else ""
        pos = str(r["POS_tag"]) if pd.notna(r["POS_tag"]) else ""
        fine_pos = str(r["fine_POS_tag"]) if pd.notna(r["fine_POS_tag"]) else ""
        deprel = (
            str(r["dependency_relation"]) if pd.notna(r["dependency_relation"]) else ""
        )
        t = Token(
            int(r["paragraph_ID"]),
            int(r["sentence_ID"]),
            int(r["token_ID_within_sentence"]),
            int(r["token_ID_within_document"]),
            word,
            pos,
            fine_pos,
            lemma,
            deprel,
            int(r["syntactic_head_ID"]),
            None,
            int(r["byte_onset"]),
        )
        t.inQuote = bool(r.get("inQuote", False))
        tokens.append(t)
    tok_df.set_index("token_ID_within_document", inplace=True)
    return tokens, tok_df


def preload_novel_static_data(novel):
    tokens, tok_df = load_tokens_from_checkpoint(novel)
    qdf = pd.read_csv(os.path.join(OUT_ROOT, novel, f"{novel}.quotes"), sep="\t")
    quotes = list(zip(qdf["quote_start"], qdf["quote_end"]))

    raw_entities = []
    with open(
        os.path.join(OUT_ROOT, novel, f"{novel}.beforeQA.entities"),
        mode="r",
        encoding="utf-8",
    ) as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                raw_entities.append((int(row[0]), int(row[1]), row[2], row[3]))

    coref_map = {}
    cm_file = os.path.join(OUT_ROOT, novel, "coref_matches.csv")
    if os.path.exists(cm_file):
        cm_df = pd.read_csv(cm_file)
        coref_map = dict(zip(cm_df["COREF"], cm_df["pdncID"]))

    ent_orig_file = os.path.join(OUT_ROOT, novel, f"{novel}.entities")
    orig_corefs = []
    if os.path.exists(ent_orig_file):
        ent_orig_df = pd.read_csv(ent_orig_file, sep="\t")
        orig_corefs = ent_orig_df["COREF"].tolist()

    char_info = pdnc_utils.read_char_info(novel, read_root=DATA_ROOT)
    enhanced_char_info = pdnc_utils.read_enchar_info(novel, read_root=DATA_ROOT)
    e_name2id = enhanced_char_info["name2id"]

    gold_df = pd.read_csv(os.path.join(DATA_ROOT, novel, "quote_info.csv"))
    qb2pid = {}
    for _, row in gold_df.iterrows():
        sb, eb = row["startByte"], row["endByte"]
        qtt = row["qText"]
        asb, aeb = create_book_corefs.get_offset_bytes(qtt, sb, eb)
        for b in range(asb, aeb):
            qb2pid[b] = row["speakerID"]

    quote_gold_speakers = []
    for q_start, q_end in quotes:
        st, et = q_start, q_end
        if tok_df.loc[st]["word"] in ['"', "“", "”", "'"]:
            st = min(st + 1, et)
        if tok_df.loc[et]["word"] in ['"', "“", "”", "'"]:
            et = max(et - 1, st)
        sb = tok_df.loc[st]["byte_onset"]
        eb = tok_df.loc[et]["byte_offset"]
        speakers = {qb2pid[b] for b in range(sb, eb) if b in qb2pid}
        quote_gold_speakers.append(list(speakers))

    entity_oracle_by_span = {}
    analyzed_entities = []
    for idx, (s_tok, e_tok, cat, text) in enumerate(raw_entities):
        orig_coref = orig_corefs[idx] if idx < len(orig_corefs) else -1
        orig_pid = coref_map.get(orig_coref, -1)

        clean_text = text.lower().strip()
        gold_pid = e_name2id.get(clean_text, None)

        if gold_pid is not None:
            entity_oracle_by_span[(int(s_tok), int(e_tok))] = int(gold_pid)

        analyzed_entities.append(
            {
                "tuple": (s_tok, e_tok, cat, text),
                "orig_pid": orig_pid,
                "gold_pid": gold_pid,
            }
        )

    return {
        "novel": novel,
        "tokens": tokens,
        "tok_df": tok_df,
        "quotes": quotes,
        "quote_gold_speakers": quote_gold_speakers,
        "analyzed_entities": analyzed_entities,
        "entity_oracle_by_span": entity_oracle_by_span,
    }


def is_valid_candidate(item):
    """Legacy filter: keep only person-like mentions with a valid cluster."""
    _, _, entity_type, _ = item["tuple"]
    has_valid_cluster = (item["orig_pid"] != -1) or (item["gold_pid"] is not None)
    return has_valid_cluster and entity_type in PERSON_CANDIDATE_TYPES


def byte_span_to_token_span(tok_df, start_byte, end_byte):
    """Map a byte span to the contiguous token interval that overlaps it."""
    if start_byte >= end_byte:
        return None

    matches = tok_df[
        (tok_df["byte_onset"] < end_byte) & (tok_df["byte_offset"] > start_byte)
    ]
    if matches.empty:
        return None

    token_ids = matches.index.astype(int)
    return int(token_ids.min()), int(token_ids.max())


def intervals_overlap(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) <= min(a_end, b_end)


def build_article_candidate_set(data):
    """Use the canonical merged candidate set from BookNLP + PDNC + explicit aliases.

    The merged CSV is a candidate pool, not the oracle. It is used only to create
    the set of viable entity spans. The true gold label is resolved separately by
    the oracle lookup and may differ from the candidate PID when the merge was
    built using a rough fallback.
    """
    tok_df = data["tok_df"]
    merged_file = os.path.join(OUT_ROOT, data["novel"], "merged_mentions.csv")

    candidates = []

    if os.path.exists(merged_file):
        mdf = pd.read_csv(merged_file)
        for _, row in mdf.iterrows():
            source = str(row.get("source", "")).strip().lower()
            pid = row.get("pdncID", -1)
            if pid is None or pd.isna(pid):
                continue
            pid = int(pid)
            if pid == -1:
                continue
            if source not in {"pdnc", "exp", "booknlp"}:
                continue

            span = byte_span_to_token_span(
                tok_df, int(row["startByte"]), int(row["endByte"])
            )
            if span is None:
                continue

            start, end = span
            candidates.append(
                {
                    "source": source,
                    "start": start,
                    "end": end,
                    "pid": pid,
                    "gold_pid": None,
                    "tuple": (start, end, "PROP_PER", str(row.get("text", "")).strip()),
                }
            )
    else:
        for item in data["analyzed_entities"]:
            orig_pid = item["orig_pid"]
            gold_pid = item["gold_pid"]
            if orig_pid == -1 and gold_pid is None:
                continue
            if orig_pid == -1 and gold_pid is not None:
                pid = gold_pid
            else:
                pid = orig_pid
            start, end, _, text = item["tuple"]
            candidates.append(
                {
                    "source": "booknlp",
                    "start": start,
                    "end": end,
                    "pid": pid,
                    "gold_pid": gold_pid,
                    "tuple": (start, end, "PROP_PER", text),
                }
            )

    final_candidates = []
    for entry in candidates:
        overlapping = False
        for idx, existing in enumerate(final_candidates):
            if intervals_overlap(existing["start"], existing["end"], entry["start"], entry["end"]):
                if CANDIDATE_PRIORITY.get(entry["source"], 0) >= CANDIDATE_PRIORITY.get(existing["source"], 0):
                    final_candidates[idx] = entry
                overlapping = True
                break
        if not overlapping:
            final_candidates.append(entry)

    for cand in final_candidates:
        oracle_pid = resolve_oracle_pid_for_candidate(cand, data)
        cand["gold_pid"] = oracle_pid

    return final_candidates


def resolve_oracle_pid_for_candidate(candidate, data):
    """Resolve the independent oracle label for a candidate span.

    This intentionally separates the candidate identity from the oracle label.
    The candidate PID may come from the merge or from BookNLP, while the oracle is
    computed from the gold-compatible span/name map and only used for evaluation.
    """
    start, end = candidate["start"], candidate["end"]
    oracle_by_span = data.get("entity_oracle_by_span", {})
    oracle_pid = oracle_by_span.get((int(start), int(end)))
    if oracle_pid is not None:
        return int(oracle_pid)

    clean_text = str(candidate["tuple"][3]).lower().strip()
    for item in data["analyzed_entities"]:
        cand_start, cand_end, _, cand_text = item["tuple"]
        if int(start) == int(cand_start) and int(end) == int(cand_end):
            if item["gold_pid"] is not None:
                return int(item["gold_pid"])
            if item["orig_pid"] != -1:
                return int(item["orig_pid"])
    if clean_text:
        char_info = pdnc_utils.read_char_info(data["novel"], read_root=DATA_ROOT)
        enhanced_char_info = pdnc_utils.read_enchar_info(data["novel"], read_root=DATA_ROOT)
        if clean_text in enhanced_char_info["name2id"]:
            return int(enhanced_char_info["name2id"][clean_text])
    return candidate["pid"]


def calc_accuracy(mentions, assignments, quotes, quote_gold_speakers):
    """Função genérica de avaliação para evitar repetição."""
    matches = []
    for idx, mention_idx in enumerate(mentions):
        gold_speakers = quote_gold_speakers[idx]
        if mention_idx is not None and len(gold_speakers) == 1:
            pred_speaker = assignments[mention_idx]
            matches.append(pred_speaker != -1 and pred_speaker == gold_speakers[0])
        else:
            matches.append(False)
    return Counter(matches)[True] / len(quotes) if len(quotes) > 0 else 0.0


def main():
    print(f"Carregando BERT QA: {MODEL_FILE}")
    qa_model = QuotationAttribution(MODEL_FILE)
    novels = create_book_corefs.novels

    print("\n--- Carregando dados estáticos ---")
    cached_novels = [preload_novel_static_data(novel) for novel in novels]

    # Dicionários para armazenar as acurácias de cada configuração
    acc_og_pred = []  # 1. BookNLP-OG (Baseline)
    acc_og_gold = []  # 2. BookNLP-OG + Oráculo de Rótulo
    acc_plus_pred = []  # 3. BookNLP+ (Candidate Pruning)
    acc_plus_gold = []  # 4. BookNLP+ + Oráculo de Rótulo (Teto Absoluto do Pipeline)
    coverage = {
        "BookNLP-OG": {
            "quotes": 0,
            "with_candidates": 0,
            "processed": 0,
            "attributed": 0,
        },
        "BookNLP+": {
            "quotes": 0,
            "with_candidates": 0,
            "processed": 0,
            "attributed": 0,
        },
    }

    print("\n--- Executando as 4 Configurações Determinísticas ---")

    for data in cached_novels:
        tokens = data["tokens"]
        quotes = data["quotes"]
        quote_gold_speakers = data["quote_gold_speakers"]

        # ==========================================
        # CONJUNTO 1: BookNLP-OG (Todos os Candidatos)
        # ==========================================
        ent_og = []
        labels_og_pred = []
        labels_og_gold = []

        for item in data["analyzed_entities"]:
            ent_og.append(item["tuple"])
            labels_og_pred.append(item["orig_pid"])

            # CORREÇÃO FALTANTE NO SEU ARQUIVO: O Fallback agora protege a Baseline
            gold_fallback = (
                item["gold_pid"] if item["gold_pid"] is not None else item["orig_pid"]
            )
            labels_og_gold.append(gold_fallback)

        # ==========================================
        # CONJUNTO 2: BookNLP+ (Poda de Candidatos)
        # ==========================================
        # BookNLP+ como descrito no artigo: coleta resolved clusters do BookNLP, PDNC annotations
        # e explicit aliases e resolve sobreposições por prioridade: PDNC > explicit > BookNLP.
        article_candidates = build_article_candidate_set(data)
        ent_plus = []
        labels_plus_pred = []
        labels_plus_gold = []

        for cand in article_candidates:
            ent_plus.append(cand["tuple"])
            pid = cand["pid"]
            labels_plus_pred.append(pid)

            oracle_pid = cand["gold_pid"] if cand["gold_pid"] is not None else pid
            labels_plus_gold.append(oracle_pid)

        # Match the model's own candidate-window logic to count quotes it processes.
        og_representation = qa_model.get_representation(quotes, ent_og, tokens)
        plus_representation = qa_model.get_representation(quotes, ent_plus, tokens)

        # ==========================================
        # INFERÊNCIA NEURAL
        # ==========================================
        with torch.no_grad():
            with open(os.devnull, "w") as devnull:
                with redirect_stdout(devnull):
                    # Roda o BERT no conjunto sujo (OG)
                    mentions_og = qa_model.tag(quotes, ent_og, tokens)
                    # Roda o BERT no conjunto limpo (BookNLP+)
                    mentions_plus = qa_model.tag(quotes, ent_plus, tokens)

        for name, mentions, representation in [
            ("BookNLP-OG", mentions_og, og_representation),
            ("BookNLP+", mentions_plus, plus_representation),
        ]:
            metas = representation[1]
            quote_indexes = representation[-1]
            coverage[name]["quotes"] += len(quotes)
            coverage[name]["with_candidates"] += sum(bool(meta[1]) for meta in metas)
            coverage[name]["processed"] += len(quote_indexes)
            coverage[name]["attributed"] += sum(
                mention_idx is not None for mention_idx in mentions
            )

        # ==========================================
        # AVALIAÇÃO DAS 4 MÉTRICAS
        # ==========================================
        acc_og_pred.append(
            calc_accuracy(mentions_og, labels_og_pred, quotes, quote_gold_speakers)
        )
        acc_og_gold.append(
            calc_accuracy(mentions_og, labels_og_gold, quotes, quote_gold_speakers)
        )

        acc_plus_pred.append(
            calc_accuracy(mentions_plus, labels_plus_pred, quotes, quote_gold_speakers)
        )
        acc_plus_gold.append(
            calc_accuracy(mentions_plus, labels_plus_gold, quotes, quote_gold_speakers)
        )

    torch.cuda.empty_cache()

    # Resultados consolidados
    print("\n========= RESULTADOS DEFINITIVOS PARA O TCC =========")
    print(f"1. BookNLP-OG (Baseline do Pipeline):         {np.mean(acc_og_pred):.4f}")
    print(f"2. BookNLP-OG + Oráculo de Rótulos:           {np.mean(acc_og_gold):.4f}")
    print(f"3. BookNLP+ (Filtragem de Candidatos):        {np.mean(acc_plus_pred):.4f}")
    print(f"4. BookNLP+ + Oráculo de Rótulos (Teto Real): {np.mean(acc_plus_gold):.4f}")
    print("\nCobertura por conjunto de candidatos:")
    for name, counts in coverage.items():
        quote_count = counts["quotes"]
        candidate_count = counts["with_candidates"]
        processed_count = counts["processed"]
        attributed_count = counts["attributed"]
        candidate_coverage = candidate_count / quote_count if quote_count else 0.0
        attribution_coverage = attributed_count / quote_count if quote_count else 0.0
        print(
            f"{name}: quotes={quote_count}, com candidatos utilizáveis="
            f"{candidate_count} ({candidate_coverage:.4f}), "
            f"representadas/processadas={processed_count}, "
            f"processadas sem candidato utilizável={processed_count - candidate_count}, "
            f"sem candidato utilizável={quote_count - candidate_count}, "
            f"saídas não nulas={attributed_count} "
            f"({attribution_coverage:.4f} do total), "
            f"processadas sem saída não nula={processed_count - attributed_count}"
        )
    print("=====================================================")

    # Salva o CSV para seus registros
    coverage_regime = {
        "BookNLP-OG": "BookNLP-OG",
        "BookNLP-OG (Oracle Label)": "BookNLP-OG",
        "BookNLP+": "BookNLP+",
        "BookNLP+ (Oracle Label)": "BookNLP+",
    }
    coverage_columns = {
        "Quotes_Totais": [],
        "Quotes_Com_Candidatos_Usaveis": [],
        "Quotes_Representadas_Processadas": [],
        "Quotes_Sem_Candidatos_Usaveis": [],
        "Cobertura_Candidatos": [],
        "Saidas_Nao_Nulas": [],
        "Cobertura_Saidas_Nao_Nulas": [],
        "Processadas_Sem_Candidato_Usavel": [],
        "Processadas_Sem_Saida_Nao_Nula": [],
    }
    for config in [
        "BookNLP-OG",
        "BookNLP-OG (Oracle Label)",
        "BookNLP+",
        "BookNLP+ (Oracle Label)",
    ]:
        counts = coverage[coverage_regime[config]]
        quote_count = counts["quotes"]
        candidate_count = counts["with_candidates"]
        processed_count = counts["processed"]
        attributed_count = counts["attributed"]
        coverage_columns["Quotes_Totais"].append(quote_count)
        coverage_columns["Quotes_Com_Candidatos_Usaveis"].append(candidate_count)
        coverage_columns["Quotes_Representadas_Processadas"].append(processed_count)
        coverage_columns["Quotes_Sem_Candidatos_Usaveis"].append(
            quote_count - candidate_count
        )
        coverage_columns["Cobertura_Candidatos"].append(
            candidate_count / quote_count if quote_count else 0.0
        )
        coverage_columns["Saidas_Nao_Nulas"].append(attributed_count)
        coverage_columns["Cobertura_Saidas_Nao_Nulas"].append(
            attributed_count / quote_count if quote_count else 0.0
        )
        coverage_columns["Processadas_Sem_Candidato_Usavel"].append(
            processed_count - candidate_count
        )
        coverage_columns["Processadas_Sem_Saida_Nao_Nula"].append(
            processed_count - attributed_count
        )

    df_results = pd.DataFrame(
        {
            "Configuração": [
                "BookNLP-OG",
                "BookNLP-OG (Oracle Label)",
                "BookNLP+",
                "BookNLP+ (Oracle Label)",
            ],
            "Acurácia_Média": [
                np.mean(acc_og_pred),
                np.mean(acc_og_gold),
                np.mean(acc_plus_pred),
                np.mean(acc_plus_gold),
            ],
            **coverage_columns,
        }
    )
    df_results.to_csv("replicacao_acl2023_results.csv", index=False)


if __name__ == "__main__":
    main()
