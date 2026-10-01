import os, re, sys, json, csv, gzip, string
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import booknlpen.create_book_corefs as create_book_corefs
import pickle as pkl

"""Evaluation utilities for the speaker-attribution experiments.

The functions in this module mirror the paper's protocol:
- M-Clus: cluster-to-character assignment quality
- M-Res: mention-level resolution quality
- named-coref evaluation: entity recognition and cluster purity/quality
- QA evaluation: quote attribution accuracy for BookNLP's speaker model

The pipeline always compares model predictions against the PDNC gold annotations
stored in the novel-specific data folders.
"""


def get_offset_bytes(qtext, sb, eb):
    """Adjust byte offsets when the gold span contains leading/trailing whitespace.

    In many extracted quote/entity spans, the stored text includes extra spaces or
    quotes around the real content. This helper trims only the whitespace so the
    byte interval still aligns with the actual textual payload.
    """
    i = 0
    while qtext[i] in string.whitespace:
        i += 1
    sb = sb + i

    i = 0
    while (qtext[-(i + 1)] in string.whitespace):
        i += 1
    eb = eb - i
    return sb, eb


def mcus_eval_booknlp(novel, limit_speakers=False):
    """Evaluate cluster-to-character mapping for BookNLP (M-Clus).

    For each predicted coreference cluster, we map its surface names to canonical
    PDNC character IDs. The cluster is then classified as:
      - unique: all mentions map to one character ID
      - multiple: mentions map to more than one character ID
      - none: no PDNC match is found

    This is a cluster-level evaluation, so we count clusters and mentions rather
    than individual token-level matches.
    """

    # Read the BookNLP entity table and keep only person entities.
    entdf = pd.read_csv(os.path.join('booknlpen/pdnc_output/', novel, novel + '.entities'), \
                    sep='\t', quoting=3, lineterminator='\n')
    entdf = entdf[entdf['cat'] == 'PER']

    if limit_speakers:
        # Optional restriction: keep only characters that appear in quote attribution.
        qadf = pd.read_csv(os.path.join('booknlpen/pdnc_output/', novel, novel + '.quotes'), \
                                    sep='\t', quoting=3, lineterminator='\n')
        ass_entids = qadf['char_id'].unique().tolist()
        entdf = entdf[entdf['COREF'].isin(ass_entids)]

    # Build reverse maps of: cluster -> names and name -> clusters.
    coref2names = {}
    name2corefs = {}
    for coref, text in zip(entdf['COREF'], entdf['text']):
        name = str(text).lower()
        if coref not in coref2names:
            coref2names[coref] = []
        coref2names[coref].append(name)
        if name not in name2corefs:
            name2corefs[name] = []
        name2corefs[str(text).lower()].append(coref)

    # Map string names to PDNC character IDs.
    charInfo = create_book_corefs.read_char_info(novel)
    e_name2id = create_book_corefs.get_enhanced_char_list(charInfo['name2id'])
    e_id2names = {}
    for n, i in e_name2id.items():
        if i not in e_id2names:
            e_id2names[i] = []
        e_id2names[i].append(n)

    coref2pids = {}
    for coref, names in coref2names.items():
        if coref not in coref2pids:
            coref2pids[coref] = []
        for name in names:
            if name.lower() in e_name2id:
                coref2pids[coref].append(e_name2id[name.lower()])

    name2pids = {}
    for name, corefs in name2corefs.items():
        if name not in name2pids:
            name2pids[name] = []
        for coref in corefs:
            name2pids[name].extend(coref2pids[coref])

    # Count cluster-level outcomes.
    num_unique = 0
    num_mult = 0
    num_none = 0

    uniq_res = 0
    mult_res = 0
    non_res = 0
    tot_res = 0

    for coref, pids in coref2pids.items():
        pids_set = set(pids)
        if len(pids_set) == 1:
            num_unique += 1
            uniq_res += len(coref2names[coref])
        elif len(pids_set) == 0:
            num_none += 1
            non_res += len(coref2names[coref])
        else:
            num_mult += 1
            mult_res += len(coref2names[coref])
        tot_res += len(coref2names[coref])

    # Columns: novel, total clusters, unique/multiple/none clusters, and mention totals.
    coref_row = [novel, len(coref2pids), num_unique, num_mult, num_none, uniq_res, mult_res, non_res, tot_res]

    return coref2pids, coref_row

def coref_eval_booknlp(novel, coref2pids):
    """Evaluate mention-level resolution for BookNLP (M-Res).

    This step aligns the predicted person mentions with the gold PDNC mentions by
    byte overlap. For each gold mention that can be linked to exactly one
    predicted character ID, we check whether that ID matches the gold PDNC ID.
    """

    gold_mens = create_book_corefs.pdnc_process_novel(novel)
    gold_mens.sort_values(by='start_byte', inplace=True)

    gold_mens['mID'] = ['M_' + str(i) for i in range(len(gold_mens))]

    # Build a mapping from each byte offset in the gold mention text to the gold
    # mention ID and its PDNC character ID.
    byte2mid = {}
    mid2pid = {}
    for _, row in gold_mens.iterrows():
        sb, eb = row['start_byte'], row['end_byte']
        mtt = row['menText']
        asb, aeb = get_offset_bytes(mtt, sb, eb)
        mtt = mtt.strip()
        for b in range(asb, aeb):
            byte2mid[b] = row['mID']
        mid2pid[row['mID']] = row['pdncID']

    # Load predicted BookNLP person entities and convert their token offsets to
    # byte offsets.
    entdf_path = os.path.join(create_book_corefs.COREF_ROOT, novel, novel + '.entities')
    entdf = create_book_corefs.read_booknlp_df(entdf_path)
    entdf = entdf[entdf['cat'] == 'PER']

    tok_df = create_book_corefs.read_booknlp_df(os.path.join(create_book_corefs.COREF_ROOT, novel, novel + '.tokens'))
    tok_df.set_index('token_ID_within_document', inplace=True)

    entdf['start_byte'] = [tok_df.loc[x]['byte_onset'] for x in entdf['start_token']]
    entdf['end_byte'] = [tok_df.loc[x]['byte_offset'] for x in entdf['end_token']]

    entdf.sort_values(by='start_byte', inplace=True)
    entdf['cID'] = ['C_' + str(i) for i in range(len(entdf))]

    # For each predicted mention, map it to the single PDNC ID implied by its
    # cluster when that cluster resolves uniquely.
    byte2cid = {}
    cid2pid = {}
    for _, row in entdf.iterrows():
        sb, eb = row['start_byte'], row['end_byte']
        for b in range(sb, eb):
            byte2cid[b] = row['cID']
        coref = row['COREF']
        coref_pids = list(set(coref2pids[coref]))
        if len(coref_pids) == 1:
            cid2pid[row['cID']] = coref_pids[0]

    # Align gold mention bytes with predicted mention spans and keep only exact
    # one-to-one matches that can be resolved to a single PID.
    mid2cids = {}
    mid2pids = {}
    for byte in byte2mid:
        if byte in byte2cid:
            mid = byte2mid[byte]
            cid = byte2cid[byte]
            if cid not in cid2pid:
                continue
            if mid not in mid2cids:
                mid2cids[mid] = set()
                mid2pids[mid] = set()
            mid2cids[mid].add(cid)
            mid2pids[mid].add(cid2pid[cid])

    single_mids = [x for x, y in mid2pids.items() if len(y) == 1]

    evaldf = gold_mens[gold_mens['mID'].isin(single_mids)]
    evaldf['matchPID'] = [list(mid2pids[x])[0] for x in evaldf['mID']]
    evaldf['match'] = evaldf['pdncID'] == evaldf['matchPID']
    return [len(gold_mens), len(evaldf), Counter(evaldf['match'])[True]]

def men_eval_booknlp():
    """Aggregate BookNLP cluster and mention metrics across all novels.

    Equivalent to men_eval_spacy(), but using the BookNLP output files instead of
    the spaCy generated results. The columns are the same and represent the same
    evaluation protocol: cluster purity and mention-level character assignment
    accuracy against the PDNC gold annotations.
    """

    rows = []
    for novel in create_book_corefs.novels:
        coref2pids, crow1 = mcus_eval_booknlp(novel)
        crow2 = coref_eval_booknlp(novel, coref2pids)
        rows.append(crow1 + crow2)

    resdf = pd.DataFrame(
        rows,
        columns=['novel', 'num_corefs', 'uc', 'mc', 'nc', 'ut', 'mt', 'nt', 'tt', 'ann_mens', 'eval_mens', 'correct']
    )
    resdf['acc'] = resdf['correct'] / resdf['eval_mens']
    return resdf


def mcus_eval_spacy(novel):
    """Evaluate cluster-to-character mapping for spaCy (M-Clus).

    This mirrors the BookNLP M-Clus logic, but reads the spaCy coreference output
    from the generated CSV files. It groups predicted mentions by cluster, maps the
    names to PDNC character IDs, and then classifies each cluster as unique,
    multiple, or unresolved.
    """

    SPACY_OUTF = 'coref/outputs/spacy'
    # entdf = pd.read_csv(os.path.join(SPACY_OUTF, novel, 'entities.csv'), header=None, names=['text', 'startByte', 'endByte'])
    entdf = pd.read_csv(os.path.join(SPACY_OUTF, novel, 'corefs.csv'), header=None, names=['clusID', 'text', 'startByte', 'endByte'])

    # Build cluster -> names and name -> clusters maps.
    coref2names = {}
    name2corefs = {}
    for coref, text in zip(entdf['clusID'], entdf['text']):
        name = str(text).lower()
        if coref not in coref2names:
            coref2names[coref] = []
        coref2names[coref].append(name)
        if name not in name2corefs:
            name2corefs[name] = []
        name2corefs[str(text).lower()].append(coref)

    # Map names to PDNC character IDs.
    charInfo = create_book_corefs.read_char_info(novel)
    e_name2id = create_book_corefs.get_enhanced_char_list(charInfo['name2id'])
    e_id2names = {}
    for n, i in e_name2id.items():
        if i not in e_id2names:
            e_id2names[i] = []
        e_id2names[i].append(n)

    coref2pids = {}
    for coref, names in coref2names.items():
        if coref not in coref2pids:
            coref2pids[coref] = []
        for name in names:
            if name.lower() in e_name2id:
                coref2pids[coref].append(e_name2id[name.lower()])

    name2pids = {}
    for name, corefs in name2corefs.items():
        if name not in name2pids:
            name2pids[name] = []
        for coref in corefs:
            name2pids[name].extend(coref2pids[coref])

    # Count how many clusters are uniquely assigned, confused across multiple
    # characters, or not mapped to any PDNC character.
    num_unique = 0
    num_mult = 0
    num_none = 0

    uniq_res = 0
    mult_res = 0
    non_res = 0
    tot_res = 0

    for coref, pids in coref2pids.items():
        pids_set = set(pids)
        if len(pids_set) == 1:
            num_unique += 1
            uniq_res += len(coref2names[coref])
        elif len(pids_set) == 0:
            num_none += 1
            non_res += len(coref2names[coref])
        else:
            num_mult += 1
            mult_res += len(coref2names[coref])
        tot_res += len(coref2names[coref])

    coref_row = [novel, len(coref2pids), num_unique, num_mult, num_none, uniq_res, mult_res, non_res, tot_res]
    return coref2pids, coref_row


def coref_eval_spacy(novel, coref2pids):
    """Evaluate mention-level resolution for spaCy (M-Res).

    This follows the same byte-overlap strategy as the BookNLP version: align the
    gold PDNC mentions with predicted spans, resolve each predicted mention to a
    single character ID when possible, and check whether that ID matches the gold
    PDNC character ID.
    """

    SPACY_OUTF = 'coref/outputs/spacy'
    gold_mens = create_book_corefs.pdnc_process_novel(novel)
    gold_mens.sort_values(by='start_byte', inplace=True)

    gold_mens['mID'] = ['M_' + str(i) for i in range(len(gold_mens))]

    # Map each gold mention byte to its mention ID and character ID.
    byte2mid = {}
    mid2pid = {}
    for _, row in gold_mens.iterrows():
        sb, eb = row['start_byte'], row['end_byte']
        mtt = row['menText']
        asb, aeb = get_offset_bytes(mtt, sb, eb)
        mtt = mtt.strip()
        for b in range(asb, aeb):
            byte2mid[b] = row['mID']
        mid2pid[row['mID']] = row['pdncID']

    # Load spaCy coreference predictions and turn their token offsets into byte
    # offsets so they can be compared with gold mentions.
    entdf = pd.read_csv(os.path.join(SPACY_OUTF, novel, 'corefs.csv'), header=None, names=['clusID', 'text', 'startByte', 'endByte'])

    entdf.sort_values(by='startByte', inplace=True)
    entdf['cID'] = ['C_' + str(i) for i in range(len(entdf))]

    # For each predicted entity span, keep only the character ID if the cluster
    # resolves uniquely to a single PDNC character.
    byte2cid = {}
    cid2pid = {}
    for _, row in entdf.iterrows():
        sb, eb = row['startByte'], row['endByte']
        for b in range(sb, eb):
            byte2cid[b] = row['cID']
        coref = row['clusID']
        coref_pids = list(set(coref2pids[coref]))
        if len(coref_pids) == 1:
            cid2pid[row['cID']] = coref_pids[0]

    mid2cids = {}
    mid2pids = {}
    for byte in byte2mid:
        if byte in byte2cid:
            mid = byte2mid[byte]
            cid = byte2cid[byte]
            if cid not in cid2pid:
                continue
            if mid not in mid2cids:
                mid2cids[mid] = set()
                mid2pids[mid] = set()
            mid2cids[mid].add(cid)
            mid2pids[mid].add(cid2pid[cid])

    single_mids = [x for x, y in mid2pids.items() if len(y) == 1]

    evaldf = gold_mens[gold_mens['mID'].isin(single_mids)]
    evaldf['matchPID'] = [list(mid2pids[x])[0] for x in evaldf['mID']]
    evaldf['match'] = evaldf['pdncID'] == evaldf['matchPID']
    return [len(gold_mens), len(evaldf), Counter(evaldf['match'])[True]]


def men_eval_spacy():
    """Aggregate spaCy cluster and mention metrics across all novels.

    This combines the per-novel M-Clus and M-Res results into a single DataFrame.

    Column meanings:
      - novel: novel identifier
      - num_corefs: total predicted coreference clusters
      - uc / mc / nc: number of clusters with one, multiple, or no PDNC character match
      - ut / mt / nt: total mentions in those three cluster categories
      - tt: total mentions across all clusters
      - ann_mens: total gold PDNC mentions for the novel
      - eval_mens: number of gold mentions that could be evaluated after alignment
      - correct: number of evaluated mentions whose assigned character matches the gold PDNC label
      - acc: correct / eval_mens

    This is the high-level summary table used in the notebook for comparing spaCy
    outputs by novel.
    """

    rows = []
    for novel in create_book_corefs.novels:
        coref2pids, crow1 = mcus_eval_spacy(novel)
        crow2 = coref_eval_spacy(novel, coref2pids)
        rows.append(crow1 + crow2)

    resdf = pd.DataFrame(
        rows,
        columns=[
            'novel', 'num_corefs',
            'uc', 'mc', 'nc',
            'ut', 'mt', 'nt', 'tt',
            'ann_mens', 'eval_mens', 'correct'
        ]
    )
    resdf['acc'] = resdf['correct'] / resdf['eval_mens']
    return resdf


def named_coref_eval_booknlp():
    """Evaluate named-entity recognition and character clustering for BookNLP.

    This measures whether the system identifies the right PDNC characters and
    whether the clusters really correspond to a single character instead of mixing
    multiple people under the same cluster.
    """

    rows = []
    for novel in create_book_corefs.novels:
        entdf_path = os.path.join(create_book_corefs.COREF_ROOT, novel, novel + '.entities')
        entdf = create_book_corefs.read_booknlp_df(entdf_path)

        charInfo = create_book_corefs.read_char_info(novel)
        e_name2id = create_book_corefs.get_enhanced_char_list(charInfo['name2id'])
        id2names = {}
        for n, i in e_name2id.items():
            if i not in id2names:
                id2names[i] = []
            id2names[i].append(n)
        charNames = set(e_name2id.keys())

        # Keep only predicted mentions whose text matches a known character name.
        medf = entdf[entdf['text'].isin(e_name2id)]
        coref2names = defaultdict(set)
        name2corefs = defaultdict(set)

        for c, n in zip(medf['COREF'], medf['text']):
            coref2names[c].add(n)
            name2corefs[n].add(c)

        # Expand aliases so a cluster containing variant names is treated as one
        # entity group. This approximates the linkage between different name forms.
        name2als = defaultdict(set)
        for n, cs in name2corefs.items():
            for c in cs:
                name2als[n].update(coref2names[c])
                for n_ in coref2names[c]:
                    for c_ in name2corefs[n_]:
                        name2als[n].update(coref2names[c_])

        for n, als in name2als.items():
            name2als[n] = sorted(list(als))

        named_clusters = ["_".join(x) for x in name2als.values()]
        named_clusters = list(set(named_clusters))
        named_clusters = [x.split("_") for x in named_clusters]

        # Convert each cluster to PDNC character IDs.
        named_pids = [[e_name2id[x.lower()] for x in y] for y in named_clusters]
        iden_pids_set = set([x for sublist in named_pids for x in sublist])
        pure_pids = set()

        num_pure = 0
        for named_pid in named_pids:
            if len(set(named_pid)) == 1:
                num_pure += 1
                pure_pids.update(set(named_pid))

        rows.append([novel, len(id2names), len(iden_pids_set), len(named_pids), num_pure, len(pure_pids)])
    rdf = pd.DataFrame(rows, columns=['novel', '# entities', '# iden_entities', '# iden_clusters', '# pure_clusters', '# pure_entities'])
    rdf['entity_rec'] = rdf['# iden_entities'] / rdf['# entities']
    rdf['clus_purity'] = rdf['# pure_clusters'] / rdf['# iden_clusters']
    rdf['clus_quality'] = rdf['# pure_clusters'] / rdf['# pure_entities']

    return rdf


def named_coref_eval_gutentag():

    # Character Identification (recognition, clustering)

    with open('../gutentag/gutentag_ents.json', 'r') as j:
        novel2ents = json.load(j)

    rows = []
    for novel in create_book_corefs.novels:
        if novel not in novel2ents:
            continue
            
        charInfo = create_book_corefs.read_char_info(novel)
        e_name2id = create_book_corefs.get_enhanced_char_list(charInfo['name2id'])
        id2names = {}
        for n, i in e_name2id.items():
            if i not in id2names:
                id2names[i] = []
            id2names[i].append(n)
        charNames = set(e_name2id.keys())
        crows = []
        for crow in novel2ents[novel]:
            names = crow[-1]
            for name in names:
                if name.lower() in e_name2id:
                    crows.append(crow)
                    break
                    
        named_clusters = [[x.lower() for x in row[-1]] for row in crows]
        named_clusters = [[x for x in y if x in e_name2id] for y in named_clusters]
        named_pids = [[e_name2id[x] for x in y] for y in named_clusters]
        iden_pids_set = set([x for sublist in named_pids for x in sublist])
        pure_pids = set()
        num_pure = 0
        for named_pid in named_pids:
            if len(set(named_pid)) == 1:
                num_pure += 1
                pure_pids.update(set(named_pid))

        rows.append([novel, len(id2names), len(iden_pids_set), len(named_pids), num_pure,  len(pure_pids)])

    gdf = pd.DataFrame(rows, columns=['novel', '# entities', '# iden_entities', '# iden_clusters', '# pure_clusters',\
                                  '# pure_entites'])
    gdf['entity_rec'] = gdf['# iden_entities'] / gdf['# entities']
    gdf['clus_purity'] = gdf['# pure_clusters'] / gdf['# iden_clusters']
    gdf['clus_quality'] = gdf['# pure_clusters'] / gdf['# pure_entites']
    return gdf
    
def named_coref_eval_spacy():
    """Evaluate spaCy character identification and clustering quality.

    This function checks how well the spaCy coreference output recovers the PDNC
    character inventory and whether its predicted clusters are pure, i.e. they do
    not mix multiple distinct PDNC characters.
    """

    rows = []
    SPACY_OUTF = 'coref/outputs/spacy'
    for novel in create_book_corefs.novels:
        # Load mention/entity files produced by spaCy.
        edf = pd.read_csv(os.path.join(SPACY_OUTF, novel, 'entities.csv'), header=None, names=['text', 'startByte', 'endByte'])
        cdf = pd.read_csv(os.path.join(SPACY_OUTF, novel, 'corefs.csv'), header=None, names=['clusID', 'text', 'startByte', 'endByte'])

        # Map PDNC character names to canonical IDs for this novel.
        charInfo = create_book_corefs.read_char_info(novel)
        e_name2id = create_book_corefs.get_enhanced_char_list(charInfo['name2id'])
        id2names = {}
        for n, i in e_name2id.items():
            if i not in id2names:
                id2names[i] = []
            id2names[i].append(n)

        # Attach a PDNC ID to every spaCy cluster mention that matches a known
        # character name; otherwise set it to -1.
        matches = []
        for ent in cdf['text']:
            if ent.lower() in e_name2id:
                matches.append(e_name2id[ent.lower()])
            else:
                matches.append(-1)

        cdf['pdncID'] = matches
        cedf = cdf[cdf['pdncID'] != -1]

        # Build cluster-name relationships for the valid mapped mentions.
        coref2names = defaultdict(set)
        name2corefs = defaultdict(set)

        for c, n in zip(cedf['clusID'], cedf['text']):
            coref2names[c].add(n)
            name2corefs[n].add(c)

        # Expand alias information so a cluster containing alternative names is
        # treated as a single named entity grouping.
        name2als = defaultdict(set)
        for n, cs in name2corefs.items():
            for c in cs:
                name2als[n].update(coref2names[c])
                for n_ in coref2names[c]:
                    for c_ in name2corefs[n_]:
                        name2als[n].update(coref2names[c_])

        for n, als in name2als.items():
            name2als[n] = sorted(list(als))

        # Convert the final clusters into PDNC IDs and separate pure vs mixed clusters.
        named_clusters = ["_".join(x) for x in name2als.values()]
        named_clusters = list(set(named_clusters))
        named_clusters = [x.split("_") for x in named_clusters]

        named_pids = [[e_name2id[x.lower()] for x in y] for y in named_clusters]
        iden_pids_set = set([x for sublist in named_pids for x in sublist])
        pure_pids = set()
        num_pure = 0
        for named_pid in named_pids:
            if len(set(named_pid)) == 1:
                num_pure += 1
                pure_pids.update(set(named_pid))

        rows.append([novel, len(id2names), len(iden_pids_set), len(named_pids), num_pure, len(pure_pids)])

    rdf = pd.DataFrame(rows, columns=['novel', '# entities', '# iden_entities', '# iden_clusters', '# pure_clusters',\
                                  '# pure_entites'])
    rdf['entity_rec'] = rdf['# iden_entities'] / rdf['# entities']
    rdf['clus_purity'] = rdf['# pure_clusters'] / rdf['# iden_clusters']
    rdf['clus_quality'] = rdf['# pure_clusters'] / rdf['# pure_entites']

    return rdf

def eval_qa_booknlp(novel):
    """Evaluate BookNLP quote-speaker attribution accuracy.

    The gold data contains quote spans and the speaker identity for each quote.
    This function compares each predicted quote attribution to the gold PDNC
    speaker ID via byte overlap, and then checks whether the predicted character
    assignment agrees with the gold speaker.
    """

    qa = create_book_corefs.read_booknlp_df(os.path.join(create_book_corefs.COREF_ROOT, novel, novel + '.quotes'))
    coref = pd.read_csv(os.path.join(create_book_corefs.SAVE_ROOT, novel, 'coref_matches.csv'))
    tokdf_path = os.path.join(create_book_corefs.COREF_ROOT, novel, novel + '.tokens')
    tokdf = create_book_corefs.read_booknlp_df(tokdf_path)

    tokdf.set_index('token_ID_within_document', inplace=True)

    gold_df = pd.read_csv(os.path.join('data/pdnc_source', novel, 'quote_info.csv'))

    # Build mappings from quote bytes to gold quote ID and gold speaker ID.
    qb2qid = {}
    qb2pid = {}
    for _, row in gold_df.iterrows():
        sb, eb = row['startByte'], row['endByte']
        qtt = row['qText']
        asb, aeb = create_book_corefs.get_offset_bytes(qtt, sb, eb)
        for b in range(asb, aeb):
            qb2qid[b] = row['qID']
            qb2pid[b] = row['speakerID']

    qa_qids = defaultdict(set)
    qa_pids = defaultdict(set)
    sbs = []
    ebs = []
    for _, row in qa.iterrows():
        st, et = row['quote_start'], row['quote_end']
        if tokdf.loc[st]['word'] == '"':
            st += 1
        if tokdf.loc[et]['word'] == '"':
            et -= 1
        sb, eb = tokdf.loc[st]['byte_onset'], tokdf.loc[et]['byte_offset']
        sbs.append(sb)
        ebs.append(eb)
        for b in range(sb, eb):
            if b in qb2qid:
                qa_qids[sb].add(qb2qid[b])
                qa_pids[sb].add(qb2pid[b])

    # Count how many predicted quote spans match 1, multiple, or zero gold quotes.
    single = 0
    mult = 0
    none = 0
    for sb, items in qa_qids.items():
        if len(items) > 1:
            mult += 1
        elif len(items) == 1:
            single += 1
        else:
            none += 1
    print(novel, single, mult, none, single + mult + none)

    qa['start_byte'] = sbs
    qa['end_byte'] = ebs
    qa['qID_matches'] = [qa_qids[x] for x in qa['start_byte']]
    qa['pID_matches'] = [qa_pids[x] for x in qa['start_byte']]

    # Map coreference IDs to PDNC character IDs to compare with the quote speaker.
    coref2char = {}
    for coref, pid in zip(coref['COREF'], coref['pdncID']):
        coref2char[coref] = pid

    qa['corefPID'] = [coref2char.get(c, -1) for c in qa['char_id']]

    matches = []
    iden = 0
    for pID, corefID in zip(qa['pID_matches'], qa['corefPID']):
        if (len(list(pID)) == 1) and (corefID == list(pID)[0]):
            matches.append(True)
        else:
            matches.append(False)

    return [novel, len(gold_df), single + mult, len(matches), Counter(matches)[True]]
