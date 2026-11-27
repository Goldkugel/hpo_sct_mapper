import sys

sys.dont_write_bytecode = True

from logger         import Logger
from adapter        import labelClass, synonymClass, referenceClass, writeHugeCSV
from adapter        import semanticClass, sourceType, sourceAbbreviation
from adapter.BaseAdapterUtils import additionalSeparator
from MappingUtils   import (
    loadInputs, keepHPOConcepts, buildAttributeLookup,
    build_child_map, build_parent_map, make_descendant_lookup,
    make_ancestor_distance_lookup, make_hierarchy_distance,
    extract_snomed_domain, strip_snomed_domain, getRepresentativeLabel,
    acceptableDomains, llm_mappings, exact_lexical_mappings,
    embedding_mappings, jaccard_mappings, ALL_METHODS, METHOD_LABELS,
    llm_filtered_path, llm_review_path, gold_path, error_analysis_dir,
    validity_column, confidence_column, step_column, hpo_id_column, sct_id_column,
    validity_accepted, validity_domain_mismatch, validity_hierarchy_reversals,
    validity_best_candidate, validity_last_resort, sct_prefix, umls_prefix,
    hpo_umls_sct_mappings, umls_hpo_sct_mappings, REVIEW_REASON,
    FILTER_STEPS,
)
import os
import re
import ast
import unicodedata
import pandas as pd
from collections import Counter, defaultdict

# ==========================================================================
# MappingErrorAnalysis.py
#
# Error analysis of the final alignment (auto-accepted + LLM-accepted pairs,
# see MappingEvaluation) against the expert's judgment (Graham file). Where
# MappingEvaluation measures how good the alignment is, this script asks why
# the pipeline and the expert disagree. Each section answers one group of
# questions and writes its cases to a CSV in error_analysis_dir:
#
#   3. Error categories of the expert-judged HPO classes
#   4. Wrong targets: how they relate to the expert's targets, and which
#      generation method, filter route, LLM confidence and similarity
#      produced them                                   (wrong_targets.csv)
#   5. Missed correspondences: where the expert's target was lost
#                                                      (missed_targets.csv)
#   6. Semantic tags: morphology next to disorder/finding targets and
#      same-name finding/disorder twins                (tag_twins.csv)
#   7. Aligned classes without a direct correspondence, by the expert's
#      match group                                     (outside_gold.csv)
#   8. Distribution of the errors over HPO branches, label length and
#      shared targets
#   9. Reasons for the disagreement between pipeline and expert, one flag
#      per reason and a primary reason per pair       (disagreements.csv)
#  10. Exact lexical matches against the expert's judgment
#                                                      (exact_lexical_vs_gold.csv)
#  11. HPO cross-references against exact lexical matches
#                                                      (xref_vs_exact_lexical.csv)
#  12. HPO classes with several cross-references: which one the expert and
#      the pipeline choose                            (multiple_xrefs.csv)
#  13. HPO cross-references against HPO → UMLS → SNOMED CT
#                                                      (xref_vs_umls_bridge.csv)
#  14. HPO cross-references against a shared UMLS CUI (xref_vs_shared_cui.csv)
#  15. Auto-accepted pairs against the LLM's decision, and the reasons of
#      the LLM's rejections                           (llm_rejected_auto.csv)
#
# "Gold" are the expert's direct correspondences (match groups "One to one
# match" and "One to many", see loadGold). The other match groups (post-
# coordinated expressions, parent matches, out of scope, clarification) are
# read from the raw expert file for sections 7, 10 and 11.
# ==========================================================================

EXAMPLE_LIMIT = 10

# Targets up to this is_a distance from an expert target are near misses
# (1 = parent or child, 2 = sibling, grandparent or grandchild).
NEAR_MISS_DISTANCE = 2

# Terms of at most this many characters count as short (abbreviations,
# acronyms), as do upper-case terms of at most SHORT_UPPER_LENGTH.
SHORT_TERM_LENGTH  = 5
SHORT_UPPER_LENGTH = 10

# Root of the HPO phenotypic abnormality subontology; its children are the
# top-level branches of section 8.
PHENOTYPIC_ABNORMALITY = "HP:0000118"

CLINICAL_TAGS  = {"disorder", "finding"}
MORPHOLOGY_TAG = "morphologic abnormality"

# Match groups of the expert's file.
GROUP_ONE_TO_ONE    = "One to one match"
GROUP_ONE_TO_MANY   = "One to many"
GROUP_EXPRESSION    = "One to one exp"
GROUP_PARENT        = "Parent match"
GROUP_OUT_OF_SCOPE  = "Out of scope"
GROUP_CLARIFICATION = "Clarification"
NOT_IN_EXPERT_FILE  = "not in the expert's file"
DIRECT_GROUPS       = {GROUP_ONE_TO_ONE, GROUP_ONE_TO_MANY}

# Columns of the expert's file.
EXPERT_ID_COLUMN    = "HP_ID"
EXPERT_LABEL_COLUMN = "Preferred Label"
EXPERT_GROUP_COLUMN = "Match group"
EXPERT_EXPRESSION_COLUMNS = ["canonical view", "SNOMED term/exp"]

# Relation of an SCT concept to another one.
REL_EQUAL     = "equal"
REL_BROADER   = "broader"
REL_NARROWER  = "narrower"
REL_LATERAL   = "lateral"
REL_UNRELATED = "no common ancestor"
REL_MISSING   = "not in SNOMED CT"

# Kinds of terms in the exact lexical matches.
HPO_LABEL   = "HPO label"
HPO_SYNONYM = "HPO exact synonym"
SCT_FSN     = "SCT FSN"
SCT_SYNONYM = "SCT synonym"

GENERATION_METHODS = [m for m in ALL_METHODS if m != llm_mappings]

# Reasons in the LLM's rationales (section 15), as (name, description,
# pattern on the lower-cased rationale). A rationale can match several; the
# first match is its primary reason. The direction patterns read "SNOMED CT
# concept ... broader" and "HPO concept ... more specific" the same way.
LLM_REASON_PATTERNS = [
    ("specificity_rule", "specificity rule: another accepted target is more fitting",
     r"specificity rule|redundant|accepting both|already accepted|accepted candidate"),
    ("target_broader", "SNOMED CT concept broader than the HPO class",
     r"snomed ct concept[^.]*\b(broader|more general|less specific|encompass|wider|parent)"
     r"|hpo concept[^.]*\b(more specific|narrower)\b"),
    ("target_narrower", "SNOMED CT concept narrower than the HPO class",
     r"snomed ct concept[^.]*\b(more specific|narrower|too specific|subtype)"
     r"|hpo concept[^.]*\b(broader|more general|less specific)\b"),
    ("specificity_unclear", "broader or narrower, direction unclear",
     r"broader|narrower|more specific|less specific|more general"),
    ("morphology", "morphologic abnormality instead of a clinical concept", r"morpholog"),
    ("finding_vs_disorder", "finding vs. disorder, sign vs. diagnosis",
     r"is a disorder|is a finding|diagnosis|syndrome|symptom or sign|\bsign\b"),
    ("anatomical_site", "different or missing anatomical site", r"anatomic|\bsite\b|location"),
    ("etiology", "etiology, congenital vs. acquired",
     r"etiolog|congenital|acquired|caused|genetic"),
    ("severity", "severity or degree", r"severity|\bmild\b|\bsevere\b|complete|degree"),
    ("different_entity", "different clinical entity",
     r"not (clinically )?equivalent|different (clinical )?entity|unrelated|not related|"
     r"different condition"),
    ("scope_or_context", "scope or clinical context", r"scope|context|clinical intent"),
]


# --------------------------------------------------------------------------
# Helpers without state
# --------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float) -> float:
    """numerator / denominator, 0.0 for an empty denominator."""
    return numerator / denominator if denominator else 0.0


def _parse_additional(entry) -> dict:
    """The additional column of an EAV row as a dict ({} if unparsable)."""
    if isinstance(entry, dict):
        return entry
    if not isinstance(entry, str):
        return {}
    try:
        parsed = ast.literal_eval(entry)
    except (ValueError, SyntaxError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tokens(term) -> list:
    """
    Lower-cased word tokens of a term without accents and punctuation, with
    a simple singular form ("ies" -> "y", trailing "s" removed from words of
    more than three letters, but not "ss", "us" or "is").
    """
    if not isinstance(term, str):
        return []
    text = unicodedata.normalize("NFKD", term)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    tokens = []
    for token in re.split(r"[^a-z0-9]+", text):
        if not token:
            continue
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            token = token[:-1]
        tokens.append(token)
    return tokens


def variant_key(term) -> str:
    """
    Normal form of a term that ignores case, accents, punctuation, plural
    forms and word order: two terms with the same key are lexical variants.
    """
    return " ".join(sorted(_tokens(term)))


def jaccard(a, b) -> float:
    """Jaccard similarity of the token sets of two terms."""
    set_a, set_b = set(_tokens(a)), set(_tokens(b))
    return _safe_div(len(set_a & set_b), len(set_a | set_b))


def twin_key(fsn) -> str:
    """
    Name of an SCT concept without its semantic tag and without a trailing
    tag word, so that "Azoospermia (finding)" and "Azoospermia disorder
    (disorder)" share the key "azoospermia".
    """
    tag  = extract_snomed_domain(fsn)
    name = strip_snomed_domain(fsn)
    if not isinstance(name, str):
        return ""
    if tag and name.lower().endswith(" " + tag):
        name = name[: -len(tag) - 1]
    return variant_key(name)


def is_short(term) -> bool:
    """True for short terms and upper-case acronyms (see SHORT_TERM_LENGTH)."""
    if not isinstance(term, str):
        return False
    term = term.strip()
    return len(term) <= SHORT_TERM_LENGTH or (term.isupper() and len(term) <= SHORT_UPPER_LENGTH)


def first_sctid(expression) -> str:
    """
    The first SCTID of an expression of the expert's file (its focus
    concept), prefixed, or None.
    """
    if not isinstance(expression, str):
        return None
    match = re.search(r"(?<!\d)(\d{6,18})(?!\d)", expression)
    return sct_prefix + match.group(1) if match else None


def similarity_bin(value) -> str:
    """Bin of a cosine or Jaccard similarity; "none" if the method did not find the pair."""
    if value is None or pd.isna(value):
        return "none"
    for lower in (1.0, 0.99, 0.95, 0.90, 0.85, 0.80):
        if value >= lower:
            return f"≥ {lower:.2f}"
    return "< 0.80"


def jaccard_bin(value: float) -> str:
    """Bin of a best Jaccard similarity of two term lists."""
    if value >= 1.0:
        return "1.0"
    if value >= 0.5:
        return "0.5–<1.0"
    if value > 0:
        return ">0–<0.5"
    return "0"


if __name__ == "__main__":

    l = Logger()
    l.printHeader("Error analysis of the HPO → SNOMED CT alignment")

    # ======================================================================
    l.printHeader("1. Loading data")
    # ======================================================================

    hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology, graham = loadInputs(
        ";", required=[llm_filtered_path]
    )
    # UMLS is reduced to its HPO and SNOMED CT references (sections 13 and
    # 14) in section 2.

    id_col    = hpo_ontology.config.id_column
    val_col   = hpo_ontology.config.value_column
    attr_col  = hpo_ontology.config.attribute_column
    add_col   = hpo_ontology.config.additional_column
    sct_id    = sct_ontology.config.id_column
    sct_val   = sct_ontology.config.value_column
    sct_attr  = sct_ontology.config.attribute_column

    l.log("Loading LLM-reviewed mappings (llm_filtered.csv)...")
    final_df = pd.read_csv(llm_filtered_path, sep=";", dtype={step_column: "string"})
    final_df = keepHPOConcepts(final_df, id_col)
    l.log(f"LLM-reviewed mappings: {len(final_df)} rows.")
    l.log("Loading LLM-reviewed mappings (llm_filtered.csv) completed.")

    review = pd.read_csv(llm_review_path, sep=";") if os.path.exists(llm_review_path) else None
    if review is None:
        l.log(f"{os.path.basename(llm_review_path)} not found: the LLM decisions are "
              f"left out.")

    # The HPO data of loadInputs keeps only exact synonyms (the ones the
    # pipeline compares); sections 5 and 11 also need the broad, narrow and
    # related synonyms, so the adapter's CSV is read once more.
    l.log("Reading all HPO synonyms...")
    hpo_csv  = os.path.join(hpo_ontology.config.output_folder, hpo_ontology.config.output_file)
    hpo_all  = pd.read_csv(hpo_csv, sep=";", engine="python")
    hpo_all  = keepHPOConcepts(hpo_all[hpo_all[attr_col] == synonymClass], id_col)
    l.log(f"HPO synonyms of all classes: {len(hpo_all)}.")
    l.log("Reading all HPO synonyms completed.")

    # The raw expert file: every match group, not only the direct
    # correspondences of loadGold.
    l.log("Reading the expert's file with all match groups...")
    expert = pd.read_csv(gold_path, sep=";", dtype=str)
    expression_column = next(
        (c for c in EXPERT_EXPRESSION_COLUMNS if c in expert.columns), None
    )
    expert = expert[expert[EXPERT_ID_COLUMN].notna()]
    l.log(f"Expert's file: {len(expert)} rows, {expert[EXPERT_ID_COLUMN].nunique()} HPO "
          f"classes, expressions read from '{expression_column}'.")
    for group, count in expert[EXPERT_GROUP_COLUMN].fillna("(empty)").value_counts().items():
        l.log(f"  {group + ':':<24}{count:7} rows")
    l.log("Reading the expert's file with all match groups completed.")

    os.makedirs(error_analysis_dir, exist_ok=True)

    # ======================================================================
    l.printHeader("2. Building lookup tables")
    # ======================================================================

    # --- SNOMED CT -------------------------------------------------------
    sct_concepts   = set(sct[sct_id].unique())
    sct_labels     = buildAttributeLookup(sct, sct_ontology, labelClass)
    sct_synonyms   = buildAttributeLookup(sct, sct_ontology, synonymClass)

    def sct_label(concept: str) -> str:
        """FSN of an SCT concept (see getRepresentativeLabel), else its ID."""
        return getRepresentativeLabel(sct_labels.get(concept, [])) or concept

    def sct_tag(concept: str) -> str:
        """Semantic tag of an SCT concept's FSN; REL_MISSING if not loaded."""
        if concept not in sct_concepts:
            return REL_MISSING
        return extract_snomed_domain(sct_label(concept)) or "(no tag)"

    def sct_terms(concept: str) -> list:
        """(term, kind) of an SCT concept: FSNs without tag and synonyms."""
        terms = [(strip_snomed_domain(t), SCT_FSN) for t in sct_labels.get(concept, [])
                 if isinstance(t, str)]
        terms += [(t, SCT_SYNONYM) for t in sct_synonyms.get(concept, []) if isinstance(t, str)]
        return terms

    sct_up             = make_ancestor_distance_lookup(build_parent_map(sct, sct_ontology))
    hierarchy_distance = make_hierarchy_distance(sct_up)

    def relation(concept: str, other: str) -> tuple:
        """
        (is_a distance, relation of concept to other): equal, broader (an
        ancestor of other), narrower, lateral, or why there is none.
        """
        if concept not in sct_concepts or other not in sct_concepts:
            return None, REL_MISSING
        if concept == other:
            return 0, REL_EQUAL
        distance = hierarchy_distance(concept, other)
        if distance is None:
            return None, REL_UNRELATED
        if concept in sct_up(other):
            return distance, REL_BROADER
        if other in sct_up(concept):
            return distance, REL_NARROWER
        return distance, REL_LATERAL

    def nearest(concept: str, targets) -> tuple:
        """
        (distance, target, relation of concept to it) for the nearest of
        targets; ties go to the smallest ID. (None, None, reason) if no
        target has a distance.
        """
        best = (None, None, REL_MISSING if concept not in sct_concepts else REL_UNRELATED)
        for target in sorted(targets):
            distance, rel = relation(concept, target)
            if distance is not None and (best[0] is None or distance < best[0]):
                best = (distance, target, rel)
        if best[1] is None and targets and all(t not in sct_concepts for t in targets):
            best = (None, None, REL_MISSING)
        return best

    def distance_bucket(distance) -> str:
        """Readable bucket of an is_a distance."""
        if distance is None:
            return "none"
        return str(distance) if distance <= 4 else "5+"

    # Same-name concepts with different semantic tags (twins), keyed by
    # twin_key of their FSN, restricted to the acceptable domains.
    l.log("Building the index of same-name SCT concepts...")
    twin_index = defaultdict(set)
    for concept in sct_concepts:
        tag = sct_tag(concept)
        if tag in acceptableDomains or tag == "(no tag)":
            key = twin_key(sct_label(concept))
            if key:
                twin_index[key].add(concept)
    twin_index = {k: v for k, v in twin_index.items()
                  if len({sct_tag(c) for c in v}) > 1}
    concept_twin_key = {c: k for k, members in twin_index.items() for c in members}
    l.log(f"Names shared by SCT concepts with different semantic tags: {len(twin_index)} "
          f"({sum(len(v) for v in twin_index.values())} concepts).")
    l.log("Building the index of same-name SCT concepts completed.")

    def are_twins(a: str, b: str) -> bool:
        """True if a and b share a name but differ in their semantic tag."""
        key = concept_twin_key.get(a)
        return key is not None and key == concept_twin_key.get(b) and sct_tag(a) != sct_tag(b)

    # --- HPO -------------------------------------------------------------
    hpo_concepts = set(hpo[id_col].unique())
    hpo_labels   = buildAttributeLookup(hpo, hpo_ontology, labelClass)

    def hpo_label(concept: str) -> str:
        """First label of an HPO class, else its ID."""
        terms = hpo_labels.get(concept, [])
        return terms[0] if terms else concept

    # Exact synonyms (compared by the pipeline) with their type, e.g.
    # "abbreviation" or "plural"; "direct" means untyped.
    hpo_exact_synonyms = defaultdict(list)
    hpo_syn_rows = hpo[hpo[attr_col] == synonymClass]
    for concept, term, additional in zip(hpo_syn_rows[id_col], hpo_syn_rows[val_col],
                                         hpo_syn_rows[add_col]):
        if isinstance(term, str):
            syn_type = str(_parse_additional(additional).get(sourceType, "direct"))
            hpo_exact_synonyms[concept].append((term, syn_type))

    # Non-exact synonyms (broad, narrow, related), only for the checks.
    hpo_other_synonyms = defaultdict(list)
    for concept, term, additional in zip(hpo_all[id_col], hpo_all[val_col], hpo_all[add_col]):
        syn_class = str(_parse_additional(additional).get(semanticClass, ""))
        if isinstance(term, str) and "exact" not in syn_class:
            hpo_other_synonyms[concept].append((term, syn_class or "unknown"))

    def hpo_terms(concept: str, all_synonyms: bool = False) -> list:
        """Label and exact synonyms of an HPO class; all_synonyms adds the non-exact ones."""
        terms = [t for t in hpo_labels.get(concept, []) if isinstance(t, str)]
        terms += [t for t, _ in hpo_exact_synonyms.get(concept, [])]
        if all_synonyms:
            terms += [t for t, _ in hpo_other_synonyms.get(concept, [])]
        return terms

    hpo_parent_map = build_parent_map(hpo, hpo_ontology)
    hpo_child_map  = build_child_map(hpo, hpo_ontology)
    hpo_ancestors  = make_descendant_lookup(hpo_parent_map)
    hpo_descendants = make_descendant_lookup(hpo_child_map)
    top_branches   = set(hpo_child_map.get(PHENOTYPIC_ABNORMALITY, set()))

    def hpo_branches(concept: str) -> list:
        """
        Top-level branches below "Phenotypic abnormality" of an HPO class
        (several with multiple inheritance).
        """
        branches = (set(hpo_ancestors(concept)) | {concept}) & top_branches
        if not branches:
            return ["(outside Phenotypic abnormality)"]
        return sorted(hpo_label(b) for b in branches)

    # SNOMED CT cross-references in HPO, also those to concepts that are not
    # active in the loaded release (MappingGeneration keeps only active ones).
    hpo_refs = hpo[(hpo[attr_col] == referenceClass) &
                   (hpo[val_col].astype(str).str.startswith(sct_prefix))]
    xrefs_all = defaultdict(set)
    for concept, target in zip(hpo_refs[id_col], hpo_refs[val_col]):
        xrefs_all[concept].add(target)
    l.log(f"HPO classes with SNOMED CT cross-references: {len(xrefs_all)}, "
          f"{sum(len(v) for v in xrefs_all.values())} cross-references, "
          f"{sum(1 for v in xrefs_all.values() for t in v if t not in sct_concepts)} to "
          f"concepts not in the loaded SNOMED CT.")
    # Cross-references to loaded concepts: the ones the pipeline uses.
    xrefs_active = {h: {s for s in v if s in sct_concepts} for h, v in xrefs_all.items()}
    xrefs_active = {h: v for h, v in xrefs_active.items() if v}

    # UMLS cross-references in HPO (CUIs, "UMLS:C..."), also to CUIs that
    # are not in the loaded UMLS.
    hpo_umls_refs = hpo[(hpo[attr_col] == referenceClass) &
                        (hpo[val_col].astype(str).str.startswith(umls_prefix))]
    hpo_umls_xrefs = defaultdict(set)
    for concept, cui in zip(hpo_umls_refs[id_col], hpo_umls_refs[val_col]):
        hpo_umls_xrefs[concept].add(cui)

    # --- UMLS ------------------------------------------------------------
    # As in MappingGeneration: reference rows CUI -> code with the source
    # vocabulary in the additional column; SCT codes are bare and get the
    # SCT prefix. The maps keep codes that are missing from the loaded HPO
    # or SNOMED CT, so that sections 13 and 14 can tell why they are lost.
    l.log("Collecting HPO and SNOMED CT references in UMLS...")
    u_id, u_attr, u_val, u_add = (umls_ontology.config.id_column,
                                  umls_ontology.config.attribute_column,
                                  umls_ontology.config.value_column,
                                  umls_ontology.config.additional_column)
    umls_cuis = set(umls[u_id].astype(str).unique())
    umls_refs = umls[umls[u_attr] == referenceClass]
    additional = umls_refs[u_add].astype(str)
    parsed_sources = {
        entry: str(_parse_additional(entry).get(sourceAbbreviation, "")).split(additionalSeparator)
        for entry in additional.unique()
    }
    sources = additional.map(parsed_sources)
    is_hpo_ref = sources.map(lambda s: "HPO" in s).to_numpy()
    is_sct_ref = sources.map(lambda s: "SNOMEDCT_US" in s).to_numpy()

    cui_hpo   = defaultdict(set)   # CUI -> HPO classes listed under it
    hpo_cuis  = defaultdict(set)   # HPO class -> CUIs that list it
    for cui, code in zip(umls_refs.loc[is_hpo_ref, u_id].astype(str),
                         umls_refs.loc[is_hpo_ref, u_val].astype(str)):
        cui_hpo[cui].add(code)
        hpo_cuis[code].add(cui)
    cui_sct   = defaultdict(set)   # CUI -> SCT concepts listed under it
    sct_cuis  = defaultdict(set)   # SCT concept -> CUIs that list it
    for cui, code in zip(umls_refs.loc[is_sct_ref, u_id].astype(str),
                         umls_refs.loc[is_sct_ref, u_val].astype(str)):
        code = code if code.startswith(sct_prefix) else sct_prefix + code
        cui_sct[cui].add(code)
        sct_cuis[code].add(cui)
    del umls, umls_refs, additional, sources, umls_ontology  # free the memory

    def cui_targets(cuis) -> set:
        """Loaded SCT concepts listed under any of the CUIs."""
        return {s for c in cuis for s in cui_sct.get(c, ()) if s in sct_concepts}

    # The two UMLS methods of MappingGeneration, recomputed with all IDs.
    umls_bridge = {h: cui_targets(c for c in cuis if c in umls_cuis)
                   for h, cuis in hpo_umls_xrefs.items() if h in hpo_concepts}
    umls_bridge = {h: v for h, v in umls_bridge.items() if v}
    shared_cui  = {h: cui_targets(cuis) for h, cuis in hpo_cuis.items() if h in hpo_concepts}
    shared_cui  = {h: v for h, v in shared_cui.items() if v}
    l.log(f"UMLS: {len(umls_cuis)} CUIs; {len(cui_hpo)} list HPO classes "
          f"({len(hpo_cuis)} HPO IDs, {sum(1 for h in hpo_cuis if h not in hpo_concepts)} "
          f"of them not in the loaded HPO); {len(cui_sct)} list SCT concepts "
          f"({len(sct_cuis)} SCT IDs, {sum(1 for s in sct_cuis if s not in sct_concepts)} "
          f"of them not in the loaded SNOMED CT).")
    l.log(f"HPO classes with UMLS cross-references: {len(hpo_umls_xrefs)}; with "
          f"HPO → UMLS → SNOMED CT targets: {len(umls_bridge)}; with shared-CUI "
          f"targets: {len(shared_cui)}.")
    l.log("Collecting HPO and SNOMED CT references in UMLS completed.")

    # --- Pipeline output -------------------------------------------------
    gen_rows = final_df[final_df[attr_col] != llm_mappings]

    pair_validity = (
        gen_rows.groupby([id_col, val_col])[validity_column].first().to_dict()
    )
    pair_methods = (
        gen_rows.groupby([id_col, val_col])[attr_col].agg(lambda m: frozenset(m)).to_dict()
    )

    def best_similarity(method: str) -> dict:
        """(hpo, sct) -> highest confidence of one similarity method."""
        rows = gen_rows[gen_rows[attr_col] == method]
        return rows.groupby([id_col, val_col])[confidence_column].max().to_dict()

    pair_cosine  = best_similarity(embedding_mappings)
    pair_jaccard = best_similarity(jaccard_mappings)

    universe_pairs = set(pair_validity)
    auto_pairs     = {p for p, v in pair_validity.items() if pd.notna(v) and v == validity_accepted}
    llm_pairs      = set(zip(final_df.loc[final_df[attr_col] == llm_mappings, id_col],
                             final_df.loc[final_df[attr_col] == llm_mappings, val_col]))
    final_pairs    = auto_pairs | llm_pairs
    kept_rejected  = auto_pairs - llm_pairs

    # Filter step that accepted a pair (section 15); missing in old runs.
    pair_step = (gen_rows.groupby([id_col, val_col])[step_column].first().to_dict()
                 if step_column in gen_rows.columns else {})

    # (decision, confidence, pass, rationale) and the number of candidates
    # reviewed together, per pair.
    review_info = {}
    review_n_sct = {}
    if review is not None:
        for row in review.itertuples(index=False):
            row = row._asdict()
            pair = (row[id_col], row[val_col])
            review_info[pair] = (
                row.get("llm_decision"), row.get(confidence_column),
                row.get("llm_pass"), row.get("llm_rationale"),
            )
            review_n_sct[pair] = row.get("n_sct")

    def llm_confidence(pair) -> str:
        """LLM confidence (1–5) of a reviewed pair, or "not reviewed"."""
        info = review_info.get(pair)
        if info is None or pd.isna(info[1]):
            return "not reviewed"
        return str(int(round(float(info[1]) * 5)))

    def route(pair) -> str:
        """How a final pair got into the alignment."""
        if pair in auto_pairs and pair in llm_pairs:
            return "auto-accepted, LLM accepted"
        if pair in auto_pairs:
            return "auto-accepted, LLM rejected (kept)"
        code = pair_validity.get(pair)
        if pd.notna(code) and code in validity_hierarchy_reversals:
            return "LLM-accepted conflict (codes 1–7)"
        if pd.notna(code) and code in (validity_best_candidate, validity_last_resort):
            return "LLM-accepted fallback (codes 8–9)"
        return "LLM-accepted, other code"

    def methods_label(pair) -> str:
        """Generation methods that found a pair, as readable text."""
        methods = pair_methods.get(pair, frozenset())
        return " + ".join(METHOD_LABELS.get(m, m) for m in GENERATION_METHODS if m in methods) or "-"

    candidates_per_hpo = defaultdict(set)
    for h, s in universe_pairs:
        candidates_per_hpo[h].add(s)
    final_per_hpo = defaultdict(set)
    final_per_sct = defaultdict(set)
    for h, s in final_pairs:
        final_per_hpo[h].add(s)
        final_per_sct[s].add(h)

    # --- Expert ----------------------------------------------------------
    gold_pairs       = set(zip(graham[hpo_id_column], graham[sct_id_column]))
    gold_hpo         = {h for h, _ in gold_pairs}
    gold_per_hpo     = defaultdict(set)
    gold_hpo_per_sct = defaultdict(set)
    for h, s in gold_pairs:
        gold_per_hpo[h].add(s)
        gold_hpo_per_sct[s].add(h)

    expert_groups = defaultdict(set)
    expert_focus  = defaultdict(set)   # focus concepts of expressions, parents, clarifications
    expert_label  = {}
    # Read by column: itertuples would rename the columns with spaces.
    for hid, group, label, expression in zip(
        expert[EXPERT_ID_COLUMN], expert[EXPERT_GROUP_COLUMN], expert[EXPERT_LABEL_COLUMN],
        expert[expression_column] if expression_column else [None] * len(expert),
    ):
        group = group if isinstance(group, str) else "(empty)"
        expert_groups[hid].add(group)
        if isinstance(label, str):
            expert_label.setdefault(hid, label)
        if group not in DIRECT_GROUPS:
            focus = first_sctid(expression)
            if focus:
                expert_focus[hid].add(focus)

    def expert_group(concept: str) -> str:
        """Match group(s) of an HPO class in the expert's file."""
        groups = expert_groups.get(concept)
        return " | ".join(sorted(groups)) if groups else NOT_IN_EXPERT_FILE

    l.log(f"Candidate pairs: {len(universe_pairs)}; final pairs: {len(final_pairs)} for "
          f"{len(final_per_hpo)} HPO classes ({len(kept_rejected)} kept although the LLM "
          f"rejected them).")
    l.log(f"Direct correspondences of the expert: {len(gold_pairs)} for {len(gold_hpo)} "
          f"HPO classes.")

    # --- Report helpers --------------------------------------------------
    def log_counter(title: str, counter: Counter, total: int = None, indent: str = "  ",
                    limit: int = None) -> None:
        """Counts of a Counter, largest first, with their share of total."""
        total = sum(counter.values()) if total is None else total
        l.log(title)
        items = counter.most_common(limit)
        for key, count in items:
            l.log(f"{indent}{str(key) + ':':<60}{count:7} ({_safe_div(count, total):7.2%})")
        if limit is not None and len(counter) > limit:
            rest = sum(c for _, c in counter.most_common()[limit:])
            l.log(f"{indent}{'(other)':<61}{rest:7} ({_safe_div(rest, total):7.2%})")

    def log_precision_by(title: str, pairs, key, limit: int = None) -> None:
        """Pairs, correct pairs and precision per value of key(pair)."""
        totals, correct = Counter(), Counter()
        for pair in pairs:
            k = key(pair)
            totals[k] += 1
            correct[k] += pair in gold_pairs
        l.log(title)
        for k, n in totals.most_common(limit):
            l.log(f"  {str(k) + ':':<60} pairs {n:7} | correct {correct[k]:6} | "
                  f"P {_safe_div(correct[k], n):.3f}")
        if limit is not None and len(totals) > limit:
            l.log(f"  ... and {len(totals) - limit} more values.")

    def pair_text(pair) -> str:
        """HPO class and SCT concept of a pair with their labels."""
        h, s = pair
        return f"{h} [{hpo_label(h)}]  →  {s} [{sct_label(s)}]"

    def write_table(rows: list, name: str) -> None:
        """Write a list of dicts to error_analysis_dir/name."""
        path = os.path.join(error_analysis_dir, name)
        table = pd.DataFrame(rows)
        if table.empty:
            l.log(f"{name}: no rows, not written.")
            return
        writeHugeCSV(table, path)
        l.log(f"{name}: {len(table)} rows written to {error_analysis_dir}.")

    l.log("Building lookup tables completed.")

    # ======================================================================
    l.printHeader("3. Error categories of the expert-judged HPO classes")
    # ======================================================================
    # Every HPO class with a direct correspondence falls in exactly one
    # category. "Exactly" and "all targets plus others" are correct, the
    # others are errors (sections 4 and 5).

    CAT_EXACT     = "exactly the expert's targets"
    CAT_EXTRA     = "all expert targets plus other targets"
    CAT_PARTIAL   = "some expert targets missing"
    CAT_WRONG     = "only targets the expert did not choose"
    CAT_CANDIDATE = "candidates, but no target in the alignment"
    CAT_NONE      = "no candidate"

    def class_category(concept: str) -> str:
        """Error category of an expert-judged HPO class."""
        ours, gold = final_per_hpo.get(concept, set()), gold_per_hpo[concept]
        if not ours:
            return CAT_CANDIDATE if candidates_per_hpo.get(concept) else CAT_NONE
        if not ours & gold:
            return CAT_WRONG
        if not gold <= ours:
            return CAT_PARTIAL
        return CAT_EXACT if ours == gold else CAT_EXTRA

    category = {h: class_category(h) for h in gold_hpo}
    log_counter(f"Expert-judged HPO classes ({len(gold_hpo)}) by category:",
                Counter(category.values()), total=len(gold_hpo))

    # ======================================================================
    l.printHeader("4. Wrong targets (aligned, but not the expert's target)")
    # ======================================================================
    # Only pairs of expert-judged HPO classes: for the others there is no
    # direct target to compare with (section 7).

    judged_final = {p for p in final_pairs if p[0] in gold_hpo}
    wrong_pairs  = judged_final - gold_pairs
    l.log(f"Final pairs of expert-judged HPO classes: {len(judged_final)}, of which "
          f"{len(wrong_pairs)} are not the expert's target "
          f"(precision {_safe_div(len(judged_final) - len(wrong_pairs), len(judged_final)):.3f}).")

    wrong_info = {}
    for pair in wrong_pairs:
        h, s = pair
        distance, target, rel = nearest(s, gold_per_hpo[h])
        wrong_info[pair] = dict(distance=distance, target=target, relation=rel)

    log_counter("A. Relation of the wrong target to the nearest expert target "
                "(is_a distance, relation):",
                Counter((distance_bucket(i["distance"]), i["relation"]) for i in wrong_info.values()),
                total=len(wrong_pairs))
    near = sum(1 for i in wrong_info.values()
               if i["distance"] is not None and i["distance"] <= NEAR_MISS_DISTANCE)
    l.log(f"  Near misses (distance ≤ {NEAR_MISS_DISTANCE}): {near} "
          f"({_safe_div(near, len(wrong_pairs)):.2%} of the wrong targets).")

    log_counter("B. Semantic tag of the wrong target → tag of the nearest (or first) "
                "expert target:",
                Counter(
                    (sct_tag(s), sct_tag(i["target"] or sorted(gold_per_hpo[h])[0]))
                    for (h, s), i in wrong_info.items()
                ),
                total=len(wrong_pairs), limit=12)

    log_precision_by("C. Final pairs of expert-judged classes by route into the alignment:",
                     judged_final, route)
    log_precision_by("D. Final pairs of expert-judged classes by generation methods "
                     "(exactly these methods):", judged_final, methods_label, limit=15)
    log_precision_by("E. Final pairs of expert-judged classes by LLM confidence (1–5):",
                     judged_final, llm_confidence)
    log_precision_by("F. Final pairs of expert-judged classes by best cosine similarity:",
                     judged_final, lambda p: similarity_bin(pair_cosine.get(p)))
    log_precision_by("G. Final pairs of expert-judged classes by best Jaccard similarity:",
                     judged_final, lambda p: similarity_bin(pair_jaccard.get(p)))

    def xref_count(pair) -> str:
        """Number of active cross-references of the pair's HPO class (see section 12)."""
        n = len(xrefs_active.get(pair[0], ()))
        return "no cross-reference" if n == 0 else "1 cross-reference" if n == 1 else (
            f"{n} cross-references" if n < 4 else "4+ cross-references")

    log_precision_by("G2. Final pairs of expert-judged classes by number of cross-references "
                     "of the class (whether the target is one of them or not):",
                     judged_final, xref_count)

    l.log("H. Examples of wrong targets, nearest first:")
    for pair in sorted(wrong_pairs, key=lambda p: (
            wrong_info[p]["distance"] if wrong_info[p]["distance"] is not None else 99, p))[:EXAMPLE_LIMIT]:
        info = wrong_info[pair]
        l.log(f"  {pair_text(pair)}")
        if info["target"]:
            l.log(f"    {info['relation']} (distance {info['distance']}) to expert target "
                  f"{info['target']} [{sct_label(info['target'])}]")

    # ======================================================================
    l.printHeader("5. Missed correspondences (expert target not in the alignment)")
    # ======================================================================

    missed_pairs = gold_pairs - final_pairs
    l.log(f"Expert correspondences not in the final alignment: {len(missed_pairs)} of "
          f"{len(gold_pairs)}.")

    LOST_HPO_MISSING   = "HPO class not in the loaded HPO (obsolete)"
    LOST_SCT_MISSING   = "expert target not in the loaded SNOMED CT (inactive)"
    LOST_NOT_GENERATED = "never generated"
    LOST_NOT_REACHED   = "generated, not reached by any filter step"
    LOST_DOMAIN        = "removed by the domain filter"
    LOST_CONFLICT      = "conflict (codes 1–7), rejected by the LLM"
    LOST_FALLBACK      = "fallback (codes 8–9), rejected by the LLM"
    LOST_OTHER         = "other"

    def where_lost(pair) -> str:
        """Pipeline stage at which an expert correspondence was lost."""
        h, s = pair
        if h not in hpo_concepts:
            return LOST_HPO_MISSING
        if s not in sct_concepts:
            return LOST_SCT_MISSING
        if pair not in pair_validity:
            return LOST_NOT_GENERATED
        code = pair_validity[pair]
        if pd.isna(code):
            return LOST_NOT_REACHED
        if code == validity_domain_mismatch:
            return LOST_DOMAIN
        if code in validity_hierarchy_reversals:
            return LOST_CONFLICT
        if code in (validity_best_candidate, validity_last_resort):
            return LOST_FALLBACK
        return LOST_OTHER

    def best_term_jaccard(concept: str, target: str) -> tuple:
        """
        (best Jaccard over all HPO terms incl. non-exact synonyms × all SCT
        terms, lexical variant found, exact match with a non-exact synonym).
        """
        h_terms = hpo_terms(concept, all_synonyms=True)
        s_terms = [t for t, _ in sct_terms(target)]
        best = max((jaccard(a, b) for a in h_terms for b in s_terms), default=0.0)
        s_keys = {variant_key(t) for t in s_terms}
        variant = any(variant_key(a) in s_keys for a in h_terms)
        s_lower = {t.lower().strip() for t in s_terms}
        non_exact = any(t.lower().strip() in s_lower for t, _ in hpo_other_synonyms.get(concept, []))
        return best, variant, non_exact

    missed_rows = []
    lost_counter = Counter()
    lexical_counter = Counter()
    for pair in sorted(missed_pairs):
        h, s = pair
        lost = where_lost(pair)
        lost_counter[lost] += 1
        row = dict(hpo_id=h, hpo_label=hpo_label(h), sct_id=s, sct_label=sct_label(s),
                   sct_tag=sct_tag(s), where_lost=lost, category=category.get(h),
                   original_validity=pair_validity.get(pair),
                   methods=methods_label(pair) if pair in pair_methods else "")
        if lost == LOST_NOT_GENERATED:
            best, variant, non_exact = best_term_jaccard(h, s)
            row.update(best_jaccard=round(best, 3), lexical_variant=variant,
                       matches_non_exact_synonym=non_exact)
            lexical_counter["lexical variant (case, punctuation, plural, word order)" if variant
                            else "exact match with a broad/narrow/related HPO synonym" if non_exact
                            else f"best token Jaccard {jaccard_bin(best)}"] += 1
        info = review_info.get(pair)
        if info is not None:
            row.update(llm_decision=info[0], llm_confidence=llm_confidence(pair),
                       llm_rationale=info[3])
        found = final_per_hpo.get(h, set()) & gold_per_hpo[h]
        if found:
            _, other, rel = nearest(s, found)
            row.update(other_expert_target_found=other, relation_to_found_target=rel)
        missed_rows.append(row)

    log_counter("A. Where the expert's correspondences were lost:", lost_counter,
                total=len(missed_pairs))
    log_counter("B. Correspondences never generated, by lexical similarity of the "
                "terms (all HPO synonyms × SCT terms):", lexical_counter)
    l.log("   A lexical variant or a non-exact synonym match points to a gap in the "
          "generation methods, a low Jaccard to a difference in wording.")

    cand_no_align = [h for h, c in category.items() if c == CAT_CANDIDATE]
    with_gold_cand = sum(1 for h in cand_no_align if candidates_per_hpo[h] & gold_per_hpo[h])
    l.log(f"C. Classes with candidates but no target in the alignment: {len(cand_no_align)}; "
          f"the expert's target was among the candidates for {with_gold_cand} "
          f"({_safe_div(with_gold_cand, len(cand_no_align)):.2%}).")

    partial = [h for h, c in category.items() if c == CAT_PARTIAL]
    partial_rel = Counter()
    for h in partial:
        found = final_per_hpo[h] & gold_per_hpo[h]
        for s in gold_per_hpo[h] - final_per_hpo[h]:
            distance, _, rel = nearest(s, found)
            partial_rel[(distance_bucket(distance), rel)] += 1
    log_counter(f"D. Classes with some expert targets missing ({len(partial)}): relation "
                f"of the missed target to the found one (distance, relation):", partial_rel)

    rejected_correct = [p for p in missed_pairs
                        if where_lost(p) in (LOST_CONFLICT, LOST_FALLBACK)]
    log_counter(f"E. Expert correspondences rejected by the LLM ({len(rejected_correct)}), "
                f"by LLM confidence:", Counter(llm_confidence(p) for p in rejected_correct))
    l.log("   Examples with the highest LLM confidence:")
    for pair in sorted(rejected_correct, key=lambda p: (llm_confidence(p), p),
                       reverse=True)[:EXAMPLE_LIMIT // 2]:
        l.log(f"    {pair_text(pair)}")
        rationale = review_info.get(pair, (None,) * 4)[3]
        if isinstance(rationale, str):
            l.log(f"      LLM: {rationale[:240]}")

    write_table(missed_rows, "missed_targets.csv")

    # ======================================================================
    l.printHeader("6. Semantic tags: morphology and same-name twins")
    # ======================================================================

    log_counter("A. Semantic tags of the final targets (all final pairs):",
                Counter(sct_tag(s) for _, s in final_pairs), total=len(final_pairs))
    log_counter("   Semantic tags of the expert's targets:",
                Counter(sct_tag(s) for _, s in gold_pairs), total=len(gold_pairs))
    log_precision_by("   Final pairs of expert-judged classes by semantic tag:",
                     judged_final, lambda p: sct_tag(p[1]))

    morph_classes = [h for h, targets in final_per_hpo.items()
                     if any(sct_tag(s) == MORPHOLOGY_TAG for s in targets)
                     and any(sct_tag(s) in CLINICAL_TAGS for s in targets)]
    morph_targets = [(h, s) for h in morph_classes for s in final_per_hpo[h]
                     if sct_tag(s) == MORPHOLOGY_TAG]
    l.log(f"B. HPO classes with a morphologic abnormality and a disorder/finding target: "
          f"{len(morph_classes)} ({len(morph_targets)} morphology targets, "
          f"{sum(p in gold_pairs for p in morph_targets)} of them the expert's target, "
          f"{sum(p[0] in gold_hpo for p in morph_targets)} of expert-judged classes).")
    for h in sorted(morph_classes)[:EXAMPLE_LIMIT // 2]:
        l.log(f"  {h} [{hpo_label(h)}]: " + "; ".join(
            sct_label(s) for s in sorted(final_per_hpo[h])))

    # Twins among the candidates of a class: which one the pipeline kept,
    # which one the expert chose and how far the other one got.
    twin_rows = []
    pattern_counter = Counter()
    for h in sorted(candidates_per_hpo):
        by_key = defaultdict(set)
        for s in candidates_per_hpo[h]:
            key = concept_twin_key.get(s)
            if key:
                by_key[key].add(s)
        for key, members in by_key.items():
            if len({sct_tag(s) for s in members}) < 2:
                continue
            final_tags = sorted(sct_tag(s) for s in members if (h, s) in final_pairs)
            gold_tags  = sorted(sct_tag(s) for s in members if (h, s) in gold_pairs)
            other_gold = sorted(sct_tag(s) for s in gold_per_hpo.get(h, set()) if s not in members)
            pattern_counter[(
                "+".join(final_tags) or "none",
                "+".join(gold_tags) or ("other concept" if other_gold
                                        else "not judged" if h not in gold_hpo else "none"),
            )] += 1
            for s in sorted(members):
                info = review_info.get((h, s), (None,) * 4)
                twin_rows.append(dict(
                    hpo_id=h, hpo_label=hpo_label(h), twin_name=key, sct_id=s,
                    sct_label=sct_label(s), sct_tag=sct_tag(s),
                    original_validity=pair_validity.get((h, s)),
                    in_final=(h, s) in final_pairs, in_gold=(h, s) in gold_pairs,
                    expert_group=expert_group(h), llm_decision=info[0],
                    llm_confidence=llm_confidence((h, s)),
                ))
    n_twin_classes = len({r["hpo_id"] for r in twin_rows})
    log_counter(f"C. Same-name twins with different tags among the candidates "
                f"({n_twin_classes} HPO classes): (tags in the alignment, tags chosen "
                f"by the expert):", pattern_counter)
    write_table(twin_rows, "tag_twins.csv")

    # ======================================================================
    l.printHeader("7. Aligned classes without a direct correspondence")
    # ======================================================================

    outside = sorted(h for h in final_per_hpo if h not in gold_hpo)
    l.log(f"Aligned HPO classes without a direct correspondence of the expert: {len(outside)}.")
    log_counter("A. By match group of the expert:",
                Counter(expert_group(h) for h in outside), total=len(outside))

    outside_rows = []
    focus_counter = defaultdict(Counter)
    for h in outside:
        group = expert_group(h)
        focus = expert_focus.get(h, set())
        for s in sorted(final_per_hpo[h]):
            distance, target, rel = nearest(s, focus) if focus else (None, None, "no focus concept")
            if focus:
                focus_counter[group][(distance_bucket(distance), rel)] += 1
            outside_rows.append(dict(
                hpo_id=h, hpo_label=hpo_label(h), sct_id=s, sct_label=sct_label(s),
                sct_tag=sct_tag(s), expert_group=group,
                expert_focus=",".join(sorted(focus)),
                expert_focus_label="; ".join(sct_label(f) for f in sorted(focus)),
                focus_distance=distance, relation_to_focus=rel, route=route((h, s)),
                in_phenotypic_abnormality=(PHENOTYPIC_ABNORMALITY in hpo_ancestors(h)
                                           or h == PHENOTYPIC_ABNORMALITY),
            ))
    l.log("B. Relation of the final target to the expert's focus concept (the first "
          "concept of an expression, the parent concept, ...):")
    for group, counter in sorted(focus_counter.items()):
        log_counter(f"  {group}:", counter, indent="    ")
    l.log("   For expressions, \"narrower\" or \"equal\" means the pre-coordinated "
          "target refines or equals the expression's focus concept.")

    not_in_file = [h for h in outside if h not in expert_groups]
    in_pa = sum(1 for h in not_in_file if PHENOTYPIC_ABNORMALITY in hpo_ancestors(h))
    l.log(f"C. Aligned classes not in the expert's file: {len(not_in_file)}, "
          f"{in_pa} of them below \"Phenotypic abnormality\".")
    log_counter("   By top-level branch (a class can be in several):",
                Counter(b for h in not_in_file for b in hpo_branches(h)),
                total=len(not_in_file), limit=10)

    # Version differences between the expert's file and the loaded HPO.
    obsolete = sorted(h for h in expert_groups if h not in hpo_concepts)
    renamed  = sorted(h for h, label in expert_label.items()
                      if h in hpo_concepts and label.strip().lower() != hpo_label(h).strip().lower())
    l.log(f"D. HPO classes of the expert's file not in the loaded HPO (obsolete or "
          f"replaced): {len(obsolete)}.")
    l.log(f"   HPO classes whose label differs from the label in the expert's file "
          f"(renamed since): {len(renamed)}.")
    for h in renamed[:EXAMPLE_LIMIT // 2]:
        l.log(f"    {h}: expert \"{expert_label[h]}\", HPO \"{hpo_label(h)}\"")
    write_table(outside_rows, "outside_gold.csv")

    # ======================================================================
    l.printHeader("8. Distribution of the errors")
    # ======================================================================

    def log_categories_by(title: str, key) -> None:
        """
        Share of each error category per value of key(hpo class); a list of
        values counts the class once under each.
        """
        groups = defaultdict(Counter)
        for h, c in category.items():
            values = key(h)
            for value in (values if isinstance(values, list) else [values]):
                groups[value][c] += 1
        l.log(title)
        l.log(f"  {'':<48}{'classes':>8} {'exact':>7} {'+extra':>7} {'partial':>8} "
              f"{'wrong':>7} {'no align.':>9} {'no cand.':>9}")
        for k, counter in sorted(groups.items(), key=lambda x: -sum(x[1].values())):
            n = sum(counter.values())
            shares = [_safe_div(counter[c], n) for c in
                      (CAT_EXACT, CAT_EXTRA, CAT_PARTIAL, CAT_WRONG, CAT_CANDIDATE, CAT_NONE)]
            l.log(f"  {str(k)[:47]:<48}{n:8} " + " ".join(f"{s:7.1%}" for s in shares[:3])
                  + f" {shares[3]:7.1%} {shares[4]:9.1%} {shares[5]:9.1%}")

    log_categories_by("A. Error categories by top-level HPO branch (a class with several "
                      "branches counts in each):", hpo_branches)

    def label_length(h: str) -> str:
        n = len(_tokens(hpo_label(h)))
        return f"{n} word(s)" if n < 5 else "5+ words"

    log_categories_by("B. Error categories by length of the HPO label:", label_length)

    def depth_bucket(h: str) -> str:
        depth = len(hpo_ancestors(h))
        return "1–5 ancestors" if depth <= 5 else "6–10 ancestors" if depth <= 10 else "11+ ancestors"

    log_categories_by("C. Error categories by number of HPO ancestors:", depth_bucket)

    def shared_bucket(pair) -> str:
        n = len(final_per_sct[pair[1]])
        return "target of 1 HPO class" if n == 1 else (
            "target of 2 HPO classes" if n == 2 else
            "target of 3–5 HPO classes" if n <= 5 else "target of 6+ HPO classes")

    log_precision_by("D. Final pairs of expert-judged classes by the number of HPO "
                     "classes sharing the target:", judged_final, shared_bucket)

    # ======================================================================
    l.printHeader("9. Reasons for the disagreement between pipeline and expert")
    # ======================================================================
    # Every disagreement (a wrong target or a missed correspondence of an
    # expert-judged class) gets one flag per possible reason and a primary
    # reason: the first flag in the order of the lists below.

    exact_pairs_pipeline = {p for p, m in pair_methods.items() if exact_lexical_mappings in m}

    def expert_used_for_related(pair) -> bool:
        """The SCT target is the expert's target of an HPO ancestor or descendant."""
        h, s = pair
        related = set(hpo_ancestors(h)) | set(hpo_descendants(h))
        return bool(gold_hpo_per_sct.get(s, set()) & related)

    def expert_collapse(h: str) -> bool:
        """The expert gives the same target to this class and a related one."""
        related = set(hpo_ancestors(h)) | set(hpo_descendants(h))
        return any(gold_hpo_per_sct[s] & related for s in gold_per_hpo[h])

    WRONG_REASONS = [
        ("expert_target_inactive",   "the expert's targets are not in the loaded SNOMED CT"),
        ("same_name_other_tag",      "same name as the expert's target, other semantic tag"),
        ("other_xref_of_class",      "one of several cross-references; the expert chose another"),
        ("near_miss_broader",       f"broader than the expert's target (≤ {NEAR_MISS_DISTANCE})"),
        ("near_miss_narrower",       f"narrower than the expert's target (≤ {NEAR_MISS_DISTANCE})"),
        ("near_miss_lateral",        f"sibling of the expert's target (≤ {NEAR_MISS_DISTANCE})"),
        ("expert_target_of_related", "the expert uses this target for a related HPO class"),
        ("other_semantic_tag",       "semantic tag differs from all expert targets"),
        ("exact_name_match",         "exact lexical match the expert did not choose"),
        ("hpo_label_changed",        "HPO label changed since the expert's judgment"),
        ("distant",                  f"is_a distance > {NEAR_MISS_DISTANCE}"),
        ("no_common_ancestor",       "no is_a path to the expert's target"),
    ]
    MISSED_REASONS = [
        ("hpo_class_obsolete",       "HPO class not in the loaded HPO"),
        ("expert_target_inactive",   "the expert's target is not in the loaded SNOMED CT"),
        ("expert_target_out_of_domain", "the expert's target has a tag outside the domain filter"),
        ("other_target_found",       "another expert target of the class is aligned (1:n)"),
        ("twin_aligned",             "a same-name twin with another tag is aligned"),
        ("other_xref_aligned",       "one of several cross-references; another one is aligned"),
        ("near_target_aligned",      f"a target ≤ {NEAR_MISS_DISTANCE} away is aligned"),
        ("expert_collapse",          "the expert uses the target also for a related HPO class"),
        ("hpo_label_changed",        "HPO label changed since the expert's judgment"),
        ("lexical_gap",              "never generated, lexical variant or non-exact synonym"),
        ("wording_difference",       "never generated, different wording"),
        ("rejected_by_llm",          "generated and rejected by the LLM"),
        ("lost_in_filter",           "generated, lost in the rule-based filter"),
    ]

    renamed_set = set(renamed)

    def label_changed(h: str) -> bool:
        """The HPO label differs from the label in the expert's file."""
        return h in renamed_set
    disagreement_rows = []
    primary_wrong, primary_missed = Counter(), Counter()
    flag_wrong, flag_missed = Counter(), Counter()

    for pair in sorted(wrong_pairs):
        h, s = pair
        info  = wrong_info[pair]
        gold  = gold_per_hpo[h]
        gold_tags = {sct_tag(g) for g in gold}
        near_ = info["distance"] is not None and info["distance"] <= NEAR_MISS_DISTANCE
        flags = dict(
            expert_target_inactive=all(g not in sct_concepts for g in gold),
            same_name_other_tag=any(are_twins(s, g) for g in gold),
            other_xref_of_class=(len(xrefs_active.get(h, ())) > 1 and s in xrefs_active[h]
                                 and bool(gold & xrefs_active[h])),
            near_miss_broader=near_ and info["relation"] == REL_BROADER,
            near_miss_narrower=near_ and info["relation"] == REL_NARROWER,
            near_miss_lateral=near_ and info["relation"] == REL_LATERAL,
            expert_target_of_related=expert_used_for_related(pair),
            other_semantic_tag=sct_tag(s) not in gold_tags,
            exact_name_match=pair in exact_pairs_pipeline,
            hpo_label_changed=label_changed(h),
            distant=info["distance"] is not None and info["distance"] > NEAR_MISS_DISTANCE,
            no_common_ancestor=info["distance"] is None,
        )
        primary = next((name for name, _ in WRONG_REASONS if flags[name]), "unexplained")
        primary_wrong[primary] += 1
        flag_wrong.update(k for k, v in flags.items() if v)
        disagreement_rows.append(dict(
            kind="wrong target", hpo_id=h, hpo_label=hpo_label(h), sct_id=s,
            sct_label=sct_label(s), expert_targets="; ".join(sct_label(g) for g in sorted(gold)),
            primary_reason=primary, route=route(pair), methods=methods_label(pair), **flags,
        ))

    for pair in sorted(missed_pairs):
        h, s = pair
        lost    = where_lost(pair)
        ours    = final_per_hpo.get(h, set())
        found   = ours & gold_per_hpo[h]
        others  = ours - gold_per_hpo[h]
        distance, _, _ = nearest(s, others) if others else (None, None, None)
        lexical = None
        if lost == LOST_NOT_GENERATED:
            best, variant, non_exact = best_term_jaccard(h, s)
            lexical = variant or non_exact
        flags = dict(
            hpo_class_obsolete=lost == LOST_HPO_MISSING,
            expert_target_inactive=lost == LOST_SCT_MISSING,
            expert_target_out_of_domain=(s in sct_concepts
                                         and sct_tag(s) not in acceptableDomains
                                         and sct_tag(s) != "(no tag)"),
            other_target_found=bool(found),
            twin_aligned=any(are_twins(s, o) for o in others),
            other_xref_aligned=(len(xrefs_active.get(h, ())) > 1 and s in xrefs_active[h]
                                and bool(others & xrefs_active[h])),
            near_target_aligned=distance is not None and distance <= NEAR_MISS_DISTANCE,
            expert_collapse=bool(gold_hpo_per_sct[s] &
                                 (set(hpo_ancestors(h)) | set(hpo_descendants(h)))),
            hpo_label_changed=label_changed(h),
            lexical_gap=lexical is True,
            wording_difference=lexical is False,
            rejected_by_llm=lost in (LOST_CONFLICT, LOST_FALLBACK),
            lost_in_filter=lost in (LOST_DOMAIN, LOST_NOT_REACHED),
        )
        primary = next((name for name, _ in MISSED_REASONS if flags[name]), "unexplained")
        primary_missed[primary] += 1
        flag_missed.update(k for k, v in flags.items() if v)
        disagreement_rows.append(dict(
            kind="missed target", hpo_id=h, hpo_label=hpo_label(h), sct_id=s,
            sct_label=sct_label(s), expert_targets="; ".join(sct_label(g) for g in sorted(gold_per_hpo[h])),
            primary_reason=primary, where_lost=lost, **flags,
        ))

    def log_reasons(title: str, reasons: list, primary: Counter, flags: Counter, total: int) -> None:
        l.log(title)
        l.log(f"  {'reason':<64}{'primary':>9}{'':>10}{'flagged':>9}")
        for name, text in reasons + [("unexplained", "none of the reasons")]:
            l.log(f"  {text[:63]:<64}{primary[name]:9} ({_safe_div(primary[name], total):6.1%})"
                  f"{flags[name]:9} ({_safe_div(flags[name], total):6.1%})")

    log_reasons(f"A. Wrong targets ({len(wrong_pairs)}):", WRONG_REASONS,
                primary_wrong, flag_wrong, len(wrong_pairs))
    log_reasons(f"B. Missed correspondences ({len(missed_pairs)}):", MISSED_REASONS,
                primary_missed, flag_missed, len(missed_pairs))

    # Do the expert-side reasons occur more often in disagreements than in
    # agreements? Higher rates point to the expert's file, not the pipeline.
    agreed_classes    = {h for h, c in category.items() if c in (CAT_EXACT, CAT_EXTRA)}
    disagree_classes  = set(gold_hpo) - agreed_classes
    l.log("C. Expert-side indicators in classes with and without disagreement:")
    for text, test in [
        ("HPO label changed since the expert's judgment", label_changed),
        ("expert gives the same target to a related HPO class", expert_collapse),
        ("expert target not in the loaded SNOMED CT",
         lambda h: any(g not in sct_concepts for g in gold_per_hpo[h])),
        ("expert target outside the domain filter's tags",
         lambda h: any(g in sct_concepts and sct_tag(g) not in acceptableDomains
                       and sct_tag(g) != "(no tag)" for g in gold_per_hpo[h])),
    ]:
        a = sum(1 for h in agreed_classes if test(h))
        d = sum(1 for h in disagree_classes if test(h))
        l.log(f"  {text + ':':<56} agreement {a:6} ({_safe_div(a, len(agreed_classes)):6.2%}) | "
              f"disagreement {d:6} ({_safe_div(d, len(disagree_classes)):6.2%})")

    write_table(disagreement_rows, "disagreements.csv")

    # ======================================================================
    l.printHeader("10. Exact lexical matches against the expert's judgment")
    # ======================================================================
    # Recomputed as in MappingGeneration (labels and exact synonyms of HPO,
    # FSNs without tag and synonyms of SCT, lower-cased and trimmed), but
    # with the terms that matched.

    l.log("Recomputing the exact lexical matches with their terms...")
    sct_lookup = defaultdict(list)
    for concept, attribute, term in zip(sct[sct_id], sct[sct_attr], sct[sct_val]):
        if not isinstance(term, str) or not term.strip():
            continue
        if attribute == labelClass:
            text, kind = strip_snomed_domain(term), SCT_FSN
        elif attribute == synonymClass:
            text, kind = term, SCT_SYNONYM
        else:
            continue
        sct_lookup[text.lower().strip()].append((concept, kind, text))

    exact_match = defaultdict(list)   # (hpo, sct) -> [(hpo term, hpo kind, sct term, sct kind)]
    for h in sorted(hpo_concepts):
        h_terms = [(t, HPO_LABEL) for t in hpo_labels.get(h, []) if isinstance(t, str)]
        h_terms += [(t, f"{HPO_SYNONYM} ({syn_type})") for t, syn_type in hpo_exact_synonyms.get(h, [])]
        for term, kind in h_terms:
            if not term.strip():
                continue
            for concept, s_kind, s_term in sct_lookup.get(term.lower().strip(), []):
                exact_match[(h, concept)].append((term, kind, s_term, s_kind))
    exact_pairs = set(exact_match)
    l.log(f"Exact lexical pairs: {len(exact_pairs)} for {len({h for h, _ in exact_pairs})} "
          f"HPO classes.")
    pipeline_exact = {p for p in exact_pairs_pipeline}
    l.log(f"Agreement with the exact_lexical pairs of the pipeline: "
          f"{len(exact_pairs & pipeline_exact)} shared, {len(exact_pairs - pipeline_exact)} "
          f"only recomputed, {len(pipeline_exact - exact_pairs)} only in the pipeline.")
    l.log("Recomputing the exact lexical matches with their terms completed.")

    exact_per_hpo = defaultdict(set)
    for h, s in exact_pairs:
        exact_per_hpo[h].add(s)

    def match_kinds(pair) -> tuple:
        """(HPO term kinds, SCT term kinds) of an exact match, best kind first."""
        matches = exact_match[pair]
        h_kinds = {m[1] for m in matches}
        s_kinds = {m[3] for m in matches}
        h_kind = HPO_LABEL if HPO_LABEL in h_kinds else sorted(h_kinds)[0]
        s_kind = SCT_FSN if SCT_FSN in s_kinds else SCT_SYNONYM
        return h_kind, s_kind

    judged_exact = {p for p in exact_pairs if p[0] in gold_hpo}
    log_precision_by(f"A. Exact lexical pairs of expert-judged classes ({len(judged_exact)}) "
                     f"by matched terms (HPO term, SCT term):", judged_exact, match_kinds)
    log_precision_by("   by semantic tag of the target:", judged_exact, lambda p: sct_tag(p[1]))
    log_precision_by("   by number of exact targets of the class:", judged_exact,
                     lambda p: "1 target" if len(exact_per_hpo[p[0]]) == 1 else "several targets")
    log_precision_by("   by short term or acronym:", judged_exact,
                     lambda p: "short term" if all(is_short(m[0]) for m in exact_match[p])
                     else "longer term")
    log_precision_by("   by case (identical strings vs. equal only after lower-casing):",
                     judged_exact,
                     lambda p: "identical" if any(m[0].strip() == m[2].strip() for m in exact_match[p])
                     else "case differs")
    log_precision_by("   by whether the pair is in the final alignment:", judged_exact,
                     lambda p: "in the final alignment" if p in final_pairs else "not in the final alignment")

    exact_class_cat = Counter()
    for h in exact_per_hpo:
        if h in gold_hpo:
            ours, gold = exact_per_hpo[h], gold_per_hpo[h]
            exact_class_cat["all exact targets are expert targets" if ours <= gold
                            else "some exact targets are expert targets" if ours & gold
                            else "no exact target is an expert target"] += 1
        else:
            exact_class_cat[f"not judged directly: {expert_group(h)}"] += 1
    log_counter(f"B. HPO classes with an exact lexical match ({len(exact_per_hpo)}):",
                exact_class_cat, total=len(exact_per_hpo))
    l.log("   Classes with an exact match that the expert expressed differently "
          "(expression, parent, out of scope) show where the expert deliberately "
          "chose another modelling.")

    exact_rows = []
    exact_reason = Counter()
    for pair in sorted(judged_exact - gold_pairs):
        h, s = pair
        gold = gold_per_hpo[h]
        distance, target, rel = nearest(s, gold)
        h_kind, s_kind = match_kinds(pair)
        flags = dict(
            expert_target_inactive=all(g not in sct_concepts for g in gold),
            homonym=len(exact_per_hpo[h]) > 1,
            expert_target_among_exact=bool(exact_per_hpo[h] & gold),
            same_name_other_tag=any(are_twins(s, g) for g in gold),
            other_semantic_tag=sct_tag(s) not in {sct_tag(g) for g in gold},
            via_hpo_synonym=h_kind != HPO_LABEL,
            via_sct_synonym=s_kind == SCT_SYNONYM,
            short_term=all(is_short(m[0]) for m in exact_match[pair]),
            near_miss=distance is not None and distance <= NEAR_MISS_DISTANCE,
            hpo_label_changed=label_changed(h),
        )
        exact_reason.update(k for k, v in flags.items() if v)
        exact_rows.append(dict(
            hpo_id=h, hpo_label=hpo_label(h), sct_id=s, sct_label=sct_label(s),
            hpo_term=exact_match[pair][0][0], hpo_term_kind=h_kind,
            sct_term=exact_match[pair][0][2], sct_term_kind=s_kind,
            expert_targets="; ".join(sct_label(g) for g in sorted(gold)),
            distance_to_expert=distance, relation_to_expert=rel,
            in_final=pair in final_pairs, **flags,
        ))
    log_counter(f"C. Exact lexical pairs the expert did not choose "
                f"({len(exact_rows)}), flags (a pair can have several):",
                exact_reason, total=len(exact_rows))
    log_counter("   Relation to the nearest expert target (distance, relation):",
                Counter((distance_bucket(r["distance_to_expert"]), r["relation_to_expert"])
                        for r in exact_rows), total=len(exact_rows))
    l.log("   Examples:")
    for r in exact_rows[:EXAMPLE_LIMIT]:
        l.log(f"    {r['hpo_id']} [{r['hpo_label']}] = \"{r['sct_term']}\" ({r['sct_term_kind']}) "
              f"→ {r['sct_label']}; expert: {r['expert_targets']}")
    write_table(exact_rows, "exact_lexical_vs_gold.csv")

    # ======================================================================
    l.printHeader("11. HPO cross-references against exact lexical matches")
    # ======================================================================
    # Possible reasons why a cross-reference and an exact lexical match of
    # the same HPO class point to different SCT concepts (or only one exists):
    #   - the cross-reference points to an inactive concept (outdated xref);
    #   - the cross-reference target is outside the domain (e.g. a body
    #     structure or an observable entity);
    #   - same name, different semantic tag (finding vs. disorder twin);
    #   - the cross-reference is broader or narrower than the exact match
    #     (xrefs are not always equivalences);
    #   - the cross-reference matches a lexical variant or a broad, narrow or
    #     related HPO synonym only, which the exact match does not compare;
    #   - the exact match is a homonym (several targets), uses a synonym or
    #     a short term (acronym);
    #   - the class has no cross-reference at all (curation gap) or several.
    # Each disagreeing pair gets one flag per reason.

    xref_classes  = set(xrefs_all)
    exact_classes = set(exact_per_hpo)
    class_cat = {}
    for h in xref_classes | exact_classes:
        x, e = xrefs_all.get(h, set()), exact_per_hpo.get(h, set())
        if not e:
            class_cat[h] = "cross-reference only"
        elif not x:
            class_cat[h] = "exact lexical match only"
        elif x == e:
            class_cat[h] = "both, same targets"
        elif x & e:
            class_cat[h] = "both, partly the same targets"
        else:
            class_cat[h] = "both, different targets"
    log_counter(f"A. HPO classes with a cross-reference or an exact lexical match "
                f"({len(class_cat)}):", Counter(class_cat.values()), total=len(class_cat))

    xref_rows = []
    xref_flags, exact_flags = Counter(), Counter()
    n_xref_rows = n_exact_rows = 0
    for h in sorted(class_cat):
        x, e = xrefs_all.get(h, set()), exact_per_hpo.get(h, set())
        if x == e:
            continue
        h_terms_all = hpo_terms(h, all_synonyms=True)
        h_keys      = {variant_key(t) for t in h_terms_all}
        h_lower_other = {t.lower().strip() for t, _ in hpo_other_synonyms.get(h, [])}
        # Cross-reference targets without an exact lexical match.
        for s in sorted(x - e):
            s_terms = [t for t, _ in sct_terms(s)]
            distance, target, rel = nearest(s, e) if e else (None, None, "no exact match")
            best = max((jaccard(a, b) for a in h_terms_all for b in s_terms), default=0.0)
            flags = dict(
                xref_target_inactive=s not in sct_concepts,
                xref_out_of_domain=(s in sct_concepts and sct_tag(s) not in acceptableDomains
                                    and sct_tag(s) != "(no tag)"),
                twin_of_exact_target=any(are_twins(s, t) for t in e),
                broader_than_exact=rel == REL_BROADER,
                narrower_than_exact=rel == REL_NARROWER,
                lateral_to_exact=rel == REL_LATERAL,
                lexical_variant=any(variant_key(t) in h_keys for t in s_terms),
                non_exact_synonym_match=any(t.lower().strip() in h_lower_other for t in s_terms),
                several_xrefs=len(x) > 1,
                no_exact_match_for_class=not e,
            )
            n_xref_rows += 1
            xref_flags.update(k for k, v in flags.items() if v)
            xref_rows.append(dict(
                side="cross-reference without exact match", hpo_id=h, hpo_label=hpo_label(h),
                sct_id=s, sct_label=sct_label(s), sct_tag=sct_tag(s),
                class_category=class_cat[h], best_jaccard=round(best, 3),
                jaccard_bin=jaccard_bin(best), distance_to_other=distance,
                relation_to_other=rel, in_gold=(h, s) in gold_pairs,
                in_final=(h, s) in final_pairs, expert_group=expert_group(h), **flags,
            ))
        # Exact lexical targets without a cross-reference.
        for s in sorted(e - x):
            active_x = {t for t in x if t in sct_concepts}
            distance, target, rel = nearest(s, active_x) if active_x else (None, None, "no active cross-reference")
            h_kind, s_kind = match_kinds((h, s))
            flags = dict(
                no_xref_for_class=not x,
                only_inactive_xrefs=bool(x) and not active_x,
                twin_of_xref_target=any(are_twins(s, t) for t in active_x),
                broader_than_xref=rel == REL_BROADER,
                narrower_than_xref=rel == REL_NARROWER,
                lateral_to_xref=rel == REL_LATERAL,
                homonym=len(e) > 1,
                via_hpo_synonym=h_kind != HPO_LABEL,
                via_sct_synonym=s_kind == SCT_SYNONYM,
                short_term=all(is_short(m[0]) for m in exact_match[(h, s)]),
                out_of_domain=(sct_tag(s) not in acceptableDomains and sct_tag(s) != "(no tag)"),
            )
            n_exact_rows += 1
            exact_flags.update(k for k, v in flags.items() if v)
            xref_rows.append(dict(
                side="exact match without cross-reference", hpo_id=h, hpo_label=hpo_label(h),
                sct_id=s, sct_label=sct_label(s), sct_tag=sct_tag(s),
                class_category=class_cat[h], hpo_term=exact_match[(h, s)][0][0],
                hpo_term_kind=h_kind, sct_term_kind=s_kind, distance_to_other=distance,
                relation_to_other=rel, in_gold=(h, s) in gold_pairs,
                in_final=(h, s) in final_pairs, expert_group=expert_group(h), **flags,
            ))

    log_counter(f"B. Cross-reference targets without an exact lexical match ({n_xref_rows}), "
                f"flags (a pair can have several):", xref_flags, total=n_xref_rows)
    xref_only_rows = [r for r in xref_rows if r["side"] == "cross-reference without exact match"]
    log_counter("   Best token Jaccard of all HPO terms × the target's terms:",
                Counter(r["jaccard_bin"] for r in xref_only_rows), total=n_xref_rows)
    log_counter("   Semantic tag of the cross-reference target:",
                Counter(r["sct_tag"] for r in xref_only_rows), total=n_xref_rows, limit=8)
    log_counter(f"C. Exact lexical targets without a cross-reference ({n_exact_rows}), "
                f"flags (a pair can have several):", exact_flags, total=n_exact_rows)

    # Who agrees with the expert where both exist but differ?
    judge = Counter()
    for h, c in class_cat.items():
        if h not in gold_hpo or c not in ("both, different targets", "both, partly the same targets"):
            continue
        gold = gold_per_hpo[h]
        x_ok = bool((xrefs_all[h] - exact_per_hpo[h]) & gold)
        e_ok = bool((exact_per_hpo[h] - xrefs_all[h]) & gold)
        shared_ok = bool(xrefs_all[h] & exact_per_hpo[h] & gold)
        judge["shared target is the expert's" if shared_ok and not (x_ok or e_ok) else
              "both differing targets are the expert's" if x_ok and e_ok else
              "cross-reference target is the expert's" if x_ok else
              "exact lexical target is the expert's" if e_ok else
              "neither is the expert's"] += 1
    log_counter("D. Expert-judged classes where both exist but differ: whose target "
                "did the expert choose?", judge)

    log_precision_by("E. Precision against the expert (expert-judged classes): cross-"
                     "references vs. exact lexical matches:",
                     {(h, s) for h in xref_classes & gold_hpo for s in xrefs_all[h]} |
                     {(h, s) for h in exact_classes & gold_hpo for s in exact_per_hpo[h]},
                     lambda p: ("both" if p[1] in xrefs_all.get(p[0], set()) and p[1] in exact_per_hpo.get(p[0], set())
                                else "cross-reference only" if p[1] in xrefs_all.get(p[0], set())
                                else "exact lexical only"))

    l.log("F. Examples of classes with different targets:")
    shown = 0
    for h in sorted(h for h, c in class_cat.items() if c == "both, different targets"):
        l.log(f"  {h} [{hpo_label(h)}]")
        l.log(f"    cross-reference: " + "; ".join(sct_label(s) for s in sorted(xrefs_all[h])))
        l.log(f"    exact lexical:   " + "; ".join(sct_label(s) for s in sorted(exact_per_hpo[h])))
        if h in gold_hpo:
            l.log(f"    expert:          " + "; ".join(sct_label(s) for s in sorted(gold_per_hpo[h])))
        shown += 1
        if shown >= EXAMPLE_LIMIT:
            break
    write_table(xref_rows, "xref_vs_exact_lexical.csv")

    # ======================================================================
    l.printHeader("12. HPO classes with several cross-references")
    # ======================================================================
    # HPO often cross-references several SCT concepts, while the expert
    # looks for the best one. Does the expert pick one of them, does the
    # pipeline pick the same one, and what distinguishes the chosen one
    # (exact name, narrowest, tag)? Only active cross-references count, as
    # the pipeline sees only these.

    multi_xrefs  = {h: x for h, x in xrefs_active.items() if len(x) > 1}
    judged_multi = sorted(h for h in multi_xrefs if h in gold_hpo)
    l.log(f"HPO classes with several active cross-references: {len(multi_xrefs)} of "
          f"{len(xrefs_active)} with any ({len(judged_multi)} expert-judged).")
    log_counter("   Number of cross-references per class:",
                Counter(f"{min(len(x), 4)}{'+' if len(x) >= 4 else ''} cross-references"
                        for x in multi_xrefs.values()), total=len(multi_xrefs))

    def xref_relation(s: str, others: set) -> str:
        """How one cross-reference relates to the other ones of its class."""
        others = others - {s}
        if any(are_twins(s, o) for o in others):
            return "same-name twin of another"
        if any(o in sct_up(s) for o in others):
            return "narrower than another"
        if any(s in sct_up(o) for o in others):
            return "broader than another"
        return "lateral or unrelated to the others"

    def chosen_from(selected: set, xrefs: set) -> str:
        """Which of the cross-references a selection takes."""
        picked = selected & xrefs
        if not selected:
            return "no target"
        if len(picked) == 1:
            return "one of them"
        if picked:
            return "several of them"
        return "none of them (other target)"

    multi_pattern, multi_rows = Counter(), []
    expert_pick, pipeline_pick = Counter(), Counter()
    for h in judged_multi:
        x, gold, ours = multi_xrefs[h], gold_per_hpo[h], final_per_hpo.get(h, set())
        e_choice, p_choice = chosen_from(gold, x), chosen_from(ours, x)
        gx, fx = gold & x, ours & x
        if gx and fx == gx:
            agreement = "same cross-reference(s)"
        elif gx and gx < fx:
            agreement = "pipeline: the expert's one plus others"
        elif gx & fx:
            agreement = "partly the same"
        elif gx and fx:
            agreement = "different cross-references"
        else:
            agreement = "-"
        multi_pattern[(e_choice, p_choice, agreement)] += 1
        # What distinguishes a single chosen cross-reference from the others?
        for picks, counter in ((gx, expert_pick), (fx, pipeline_pick)):
            if len(picks) == 1:
                s = next(iter(picks))
                counter[(xref_relation(s, x),
                         "exact name" if (h, s) in exact_pairs else "no exact name",
                         sct_tag(s))] += 1
        for s in sorted(x):
            info = review_info.get((h, s), (None,) * 4)
            multi_rows.append(dict(
                hpo_id=h, hpo_label=hpo_label(h), sct_id=s, sct_label=sct_label(s),
                sct_tag=sct_tag(s), n_xrefs=len(x), relation_to_other_xrefs=xref_relation(s, x),
                exact_name=(h, s) in exact_pairs, in_gold=(h, s) in gold_pairs,
                in_final=(h, s) in final_pairs, original_validity=pair_validity.get((h, s)),
                llm_decision=info[0], llm_confidence=llm_confidence((h, s)),
                expert_choice=e_choice, pipeline_choice=p_choice, agreement=agreement,
            ))

    log_counter(f"A. Expert-judged classes with several cross-references ({len(judged_multi)}): "
                f"(expert takes, pipeline takes, agreement):", multi_pattern,
                total=len(judged_multi))
    same_single = multi_pattern[("one of them", "one of them", "same cross-reference(s)")]
    expert_one  = sum(n for (e, _, _), n in multi_pattern.items() if e == "one of them")
    l.log(f"   The expert takes exactly one of the cross-references in {expert_one} classes "
          f"({_safe_div(expert_one, len(judged_multi)):.2%}); both take the same single one "
          f"in {same_single} ({_safe_div(same_single, len(judged_multi)):.2%}).")
    log_counter("B. The cross-reference the expert takes, if exactly one (relation to "
                "the others, exact name, tag):", expert_pick, limit=12)
    log_counter("C. The cross-reference the pipeline takes, if exactly one:",
                pipeline_pick, limit=12)
    judged_xref_pairs = {(h, s) for h in xrefs_active if h in gold_hpo for s in xrefs_active[h]}
    log_precision_by("D. Precision of the cross-references against the expert, by number "
                     "of cross-references of the class:", judged_xref_pairs, xref_count)
    write_table(multi_rows, "multiple_xrefs.csv")

    # ======================================================================
    # Sections 13 and 14 compare the HPO cross-references with a UMLS route.
    # ======================================================================

    def compare_with_xrefs(name: str, other: dict, method: str, xref_flags, other_flags,
                           file_name: str) -> None:
        """
        Compare the HPO cross-references (all, also to inactive concepts)
        with other (HPO class -> SCT concepts): class categories, one flag
        per possible reason for each target found by only one side, the
        relation to the other side's nearest target, whose target the
        expert chose and the precision of each side. xref_flags(h, s) and
        other_flags(h, s) return the route-specific flags.
        """
        pipeline_pairs = {p for p, m in pair_methods.items() if method in m}
        recomputed     = {(h, s) for h, v in other.items() for s in v}
        l.log(f"Recomputed {name} pairs: {len(recomputed)}; agreement with the "
              f"{method} pairs of the pipeline: {len(recomputed & pipeline_pairs)} shared, "
              f"{len(recomputed - pipeline_pairs)} only recomputed, "
              f"{len(pipeline_pairs - recomputed)} only in the pipeline.")

        classes = set(xrefs_all) | set(other)
        cats = {}
        for h in classes:
            x, o = xrefs_all.get(h, set()), other.get(h, set())
            cats[h] = ("both, same targets" if x == o else
                       "cross-reference only" if not o else
                       f"{name} only" if not x else
                       "both, partly the same targets" if x & o else
                       "both, different targets")
        log_counter(f"A. HPO classes with a cross-reference or a {name} target "
                    f"({len(classes)}):", Counter(cats.values()), total=len(classes))
        agree = sum(1 for c in cats.values() if c == "both, same targets")
        both  = sum(1 for c in cats.values() if c.startswith("both"))
        l.log(f"   Same targets in {agree} of the {both} classes with both "
              f"({_safe_div(agree, both):.2%}).")

        rows = []
        sides = [
            ("cross-reference without " + name, xref_flags, lambda h: xrefs_all.get(h, set()),
             lambda h: other.get(h, set())),
            (name + " without cross-reference", other_flags, lambda h: other.get(h, set()),
             lambda h: xrefs_all.get(h, set())),
        ]
        for side, flag_fn, own, opposite in sides:
            flags_counter, relation_counter, n = Counter(), Counter(), 0
            for h in sorted(classes):
                if cats[h] == "both, same targets":
                    continue
                opp = {t for t in opposite(h) if t in sct_concepts}
                for s in sorted(own(h) - opposite(h)):
                    distance, target, rel = (nearest(s, opp) if opp
                                             else (None, None, "other side has no target"))
                    flags = dict(twin_of_other=any(are_twins(s, t) for t in opp),
                                 **flag_fn(h, s))
                    n += 1
                    flags_counter.update(k for k, v in flags.items() if v)
                    relation_counter[(distance_bucket(distance), rel)] += 1
                    rows.append(dict(
                        side=side, hpo_id=h, hpo_label=hpo_label(h), sct_id=s,
                        sct_label=sct_label(s), sct_tag=sct_tag(s), class_category=cats[h],
                        distance_to_other=distance, relation_to_other=rel,
                        nearest_other=target or "", in_gold=(h, s) in gold_pairs,
                        in_final=(h, s) in final_pairs, expert_group=expert_group(h), **flags,
                    ))
            log_counter(f"{'B' if side.startswith('cross') else 'C'}. {side[0].upper() + side[1:]} "
                        f"({n}), flags (a target can have several):", flags_counter, total=n)
            log_counter("   Relation to the nearest target of the other side "
                        "(distance, relation):", relation_counter, total=n, limit=10)

        judge = Counter()
        for h, c in cats.items():
            if h not in gold_hpo or not c.startswith("both") or c == "both, same targets":
                continue
            gold = gold_per_hpo[h]
            x, o = xrefs_all[h], other[h]
            x_ok, o_ok, shared_ok = bool((x - o) & gold), bool((o - x) & gold), bool(x & o & gold)
            judge["shared target is the expert's" if shared_ok and not (x_ok or o_ok) else
                  "both differing targets are the expert's" if x_ok and o_ok else
                  "cross-reference target is the expert's" if x_ok else
                  f"{name} target is the expert's" if o_ok else
                  "neither is the expert's"] += 1
        log_counter("D. Expert-judged classes where both exist but differ: whose target "
                    "did the expert choose?", judge)
        log_precision_by(
            "E. Precision against the expert (expert-judged classes):",
            {(h, s) for h in classes if h in gold_hpo
             for s in xrefs_all.get(h, set()) | other.get(h, set())},
            lambda p: ("both" if p[1] in xrefs_all.get(p[0], set()) and p[1] in other.get(p[0], set())
                       else "cross-reference only" if p[1] in xrefs_all.get(p[0], set())
                       else f"{name} only"))
        l.log("F. Examples of classes with different targets:")
        for h in sorted(h for h, c in cats.items() if c == "both, different targets")[:EXAMPLE_LIMIT // 2]:
            l.log(f"  {h} [{hpo_label(h)}]")
            l.log("    cross-reference: " + "; ".join(sct_label(s) for s in sorted(xrefs_all[h])))
            l.log(f"    {name}: " + "; ".join(sct_label(s) for s in sorted(other[h])))
            if h in gold_hpo:
                l.log("    expert: " + "; ".join(sct_label(s) for s in sorted(gold_per_hpo[h])))
        write_table(rows, file_name)

    def several_sct_in_cui(cuis, s: str) -> bool:
        """One of the CUIs that list s also lists other loaded SCT concepts."""
        return any(len(cui_targets([c])) > 1 for c in set(cuis) & sct_cuis.get(s, set()))

    # ======================================================================
    l.printHeader("13. HPO cross-references against HPO → UMLS → SNOMED CT")
    # ======================================================================
    # How well do the HPO cross-references and the cross-references over
    # UMLS align? Possible reasons for a disagreement:
    #   - the HPO class has no UMLS cross-reference, or its CUI is missing
    #     from the loaded UMLS release;
    #   - the CUI lists no SNOMED CT concept, or only concepts missing from
    #     the loaded SNOMED CT (US extension, or inactive in this release);
    #   - the SCT cross-reference is inactive, or not in UMLS at all (newer
    #     than the UMLS release);
    #   - both cross-references were curated independently: the SCT concept
    #     sits under another CUI than the HPO's UMLS cross-reference (and
    #     that CUI may even list the HPO class itself);
    #   - a CUI groups several SCT concepts (synonyms, same-name twins), so
    #     the bridge adds targets;
    #   - the targets are broader, narrower or siblings of each other;
    #   - the class has several UMLS or SCT cross-references.

    def bridge_xref_flags(h: str, s: str) -> dict:
        h_cuis, s_cuis = hpo_umls_xrefs.get(h, set()), sct_cuis.get(s, set())
        loaded = h_cuis & umls_cuis
        return dict(
            xref_target_inactive=s not in sct_concepts,
            no_umls_xref=not h_cuis,
            umls_xref_not_in_umls=bool(h_cuis) and not loaded,
            umls_xref_without_sct=bool(loaded) and not any(cui_sct.get(c) for c in loaded),
            umls_xref_sct_not_loaded=(any(cui_sct.get(c) for c in loaded)
                                      and not cui_targets(loaded)),
            xref_target_not_in_umls=not s_cuis,
            xref_target_under_other_cui=bool(s_cuis) and bool(h_cuis) and not (s_cuis & h_cuis),
            other_cui_lists_hpo_class=any(h in cui_hpo.get(c, ()) for c in s_cuis - h_cuis),
            several_umls_xrefs=len(h_cuis) > 1,
            several_sct_xrefs=len(xrefs_all.get(h, ())) > 1,
        )

    def bridge_other_flags(h: str, s: str) -> dict:
        h_cuis = hpo_umls_xrefs.get(h, set())
        x = xrefs_all.get(h, set())
        return dict(
            no_sct_xref=not x,
            only_inactive_xrefs=bool(x) and not (x & sct_concepts),
            cui_lists_several_sct=several_sct_in_cui(h_cuis, s),
            several_umls_xrefs=len(h_cuis) > 1,
        )

    compare_with_xrefs("UMLS bridge", umls_bridge, hpo_umls_sct_mappings,
                       bridge_xref_flags, bridge_other_flags, "xref_vs_umls_bridge.csv")

    # ======================================================================
    l.printHeader("14. HPO cross-references against a shared UMLS CUI")
    # ======================================================================
    # How well do the HPO cross-references and the cross-references with a
    # shared CUI align? Here UMLS, not HPO, decides which CUI an HPO class
    # belongs to. Possible reasons for a disagreement:
    #   - the HPO class is not in UMLS (newer than the HPO version of the
    #     UMLS release) or under a CUI without SNOMED CT concepts;
    #   - UMLS puts the HPO class under another CUI than HPO's own UMLS
    #     cross-reference (UMLS integration vs. HPO curation);
    #   - the SCT cross-reference is inactive, not in UMLS, or under another
    #     CUI than the HPO class;
    #   - a CUI groups several SCT concepts (synonyms, same-name twins);
    #   - the HPO class sits under several CUIs;
    #   - the targets are broader, narrower or siblings of each other.

    cui_agreement = Counter()
    for h in hpo_concepts:
        u, x = hpo_cuis.get(h, set()), hpo_umls_xrefs.get(h, set())
        cui_agreement["neither" if not u and not x else
                      "only HPO's UMLS cross-reference" if not u else
                      "only listed in UMLS" if not x else
                      "same CUIs" if u == x else
                      "partly the same CUIs" if u & x else "different CUIs"] += 1
    log_counter("CUIs of the loaded HPO classes: UMLS listing vs. HPO's own UMLS "
                "cross-reference:", cui_agreement, total=len(hpo_concepts))
    not_loaded = sum(1 for h in hpo_cuis if h.startswith("HP:") and h not in hpo_concepts)
    l.log(f"HPO IDs listed in UMLS but not in the loaded HPO (obsolete or replaced): "
          f"{not_loaded}.")

    def shared_xref_flags(h: str, s: str) -> dict:
        h_u, h_x, s_cuis = hpo_cuis.get(h, set()), hpo_umls_xrefs.get(h, set()), sct_cuis.get(s, set())
        return dict(
            xref_target_inactive=s not in sct_concepts,
            hpo_class_not_in_umls=not h_u,
            umls_cui_differs_from_hpo_umls_xref=bool(h_u) and bool(h_x) and not (h_u & h_x),
            cui_without_sct=bool(h_u) and not any(cui_sct.get(c) for c in h_u),
            cui_sct_not_loaded=any(cui_sct.get(c) for c in h_u) and not cui_targets(h_u),
            xref_target_not_in_umls=not s_cuis,
            xref_target_under_other_cui=bool(s_cuis) and bool(h_u) and not (s_cuis & h_u),
            several_cuis_for_class=len(h_u) > 1,
            several_sct_xrefs=len(xrefs_all.get(h, ())) > 1,
        )

    def shared_other_flags(h: str, s: str) -> dict:
        h_u, h_x = hpo_cuis.get(h, set()), hpo_umls_xrefs.get(h, set())
        x = xrefs_all.get(h, set())
        return dict(
            no_sct_xref=not x,
            only_inactive_xrefs=bool(x) and not (x & sct_concepts),
            cui_lists_several_sct=several_sct_in_cui(h_u, s),
            cui_differs_from_hpo_umls_xref=bool(h_x) and not (sct_cuis.get(s, set()) & h_x),
            several_cuis_for_class=len(h_u) > 1,
        )

    compare_with_xrefs("shared CUI", shared_cui, umls_hpo_sct_mappings,
                       shared_xref_flags, shared_other_flags, "xref_vs_shared_cui.csv")

    # ======================================================================
    l.printHeader("15. Auto-accepted pairs against the LLM's decision")
    # ======================================================================
    # The LLM reviews the auto-accepted pairs too (review reason ACCEPTED);
    # pairs it rejects stay in the final alignment. How often does it agree,
    # where does it disagree, who is right according to the expert, and
    # which reasons does it give for its rejections?

    reviewed_auto = sorted(p for p in auto_pairs if p in review_info)
    accepted_auto = [p for p in reviewed_auto if review_info[p][0] == "ACCEPT"]
    rejected_auto = [p for p in reviewed_auto if review_info[p][0] != "ACCEPT"]
    l.log(f"Auto-accepted pairs: {len(auto_pairs)}; reviewed by the LLM: {len(reviewed_auto)}; "
          f"accepted {len(accepted_auto)} ({_safe_div(len(accepted_auto), len(reviewed_auto)):.2%}), "
          f"rejected {len(rejected_auto)} ({_safe_div(len(rejected_auto), len(reviewed_auto)):.2%}).")
    if not reviewed_auto:
        l.log(f"No LLM decisions for auto-accepted pairs ({os.path.basename(llm_review_path)} "
              f"missing or empty): section skipped.")
    else:
        def gold_status(pair) -> str:
            if pair[0] not in gold_hpo:
                return "class not judged directly"
            return "expert's target" if pair in gold_pairs else "not the expert's target"

        log_counter("A. LLM decision × expert's judgment:",
                    Counter((review_info[p][0], gold_status(p)) for p in reviewed_auto),
                    total=len(reviewed_auto))
        judged_acc = [p for p in accepted_auto if p[0] in gold_hpo]
        judged_rej = [p for p in rejected_auto if p[0] in gold_hpo]
        right_rej  = sum(p not in gold_pairs for p in judged_rej)
        l.log(f"   Precision of the auto-accepted pairs the LLM accepted: "
              f"{_safe_div(sum(p in gold_pairs for p in judged_acc), len(judged_acc)):.3f}; "
              f"of those it rejected: {_safe_div(len(judged_rej) - right_rej, len(judged_rej)):.3f}.")
        l.log(f"   The expert confirms {right_rej} of the {len(judged_rej)} judged rejections "
              f"({_safe_div(right_rej, len(judged_rej)):.2%}).")

        step_labels = {step_id: label for step_id, label, _ in FILTER_STEPS}

        def log_rejection_rate_by(title: str, key) -> None:
            """Reviewed, rejected, rejection rate and expert-confirmed rejections per key."""
            totals, rejected, confirmed, judged = Counter(), Counter(), Counter(), Counter()
            for p in reviewed_auto:
                k = key(p)
                totals[k] += 1
                if review_info[p][0] != "ACCEPT":
                    rejected[k] += 1
                    if p[0] in gold_hpo:
                        judged[k] += 1
                        confirmed[k] += p not in gold_pairs
            l.log(title)
            for k, n in totals.most_common(12):
                l.log(f"  {str(k)[:59] + ':':<60} reviewed {n:6} | rejected {rejected[k]:5} "
                      f"({_safe_div(rejected[k], n):6.2%}) | confirmed by the expert "
                      f"{confirmed[k]:4} of {judged[k]:4}")

        log_rejection_rate_by("B. By the filter step that accepted the pair:",
                              lambda p: step_labels.get(str(pair_step.get(p)), "unknown step"))
        log_rejection_rate_by("C. By LLM pass (1 = one candidate, 2 = several together):",
                              lambda p: f"pass {review_info[p][2]}")
        log_rejection_rate_by("D. By number of candidates reviewed together:",
                              lambda p: (lambda n: "unknown" if n is None or pd.isna(n) else
                                         f"{int(n)}" if n < 5 else "5+")(review_n_sct.get(p)))
        log_rejection_rate_by("E. By semantic tag of the target:", lambda p: sct_tag(p[1]))
        log_rejection_rate_by("F. By generation methods:", methods_label)

        def llm_target_relation(pair) -> str:
            """Relation of a rejected target to the targets the LLM accepted for the class."""
            h, s = pair
            accepted = {t for t in final_per_hpo.get(h, set()) if (h, t) in llm_pairs and t != s}
            if not accepted:
                return "no other target accepted by the LLM"
            if any(are_twins(s, t) for t in accepted):
                return "same-name twin of an LLM-accepted target"
            distance, _, rel = nearest(s, accepted)
            return f"{rel} than an LLM-accepted target" if rel in (REL_BROADER, REL_NARROWER) \
                else f"{rel} to the LLM-accepted targets (distance {distance_bucket(distance)})"

        log_counter("G. Rejected auto-accepted pairs: relation to the targets the LLM "
                    "accepted for the class:", Counter(llm_target_relation(p) for p in rejected_auto),
                    total=len(rejected_auto))
        log_counter("   LLM confidence of the rejections (1–5):",
                    Counter(llm_confidence(p) for p in rejected_auto), total=len(rejected_auto))

        reason_flags, reason_primary, reason_confirmed, reason_judged = (Counter(), Counter(),
                                                                         Counter(), Counter())
        rejected_rows = []
        for p in rejected_auto:
            h, s = p
            rationale = review_info[p][3]
            text = rationale.lower() if isinstance(rationale, str) else ""
            flags = {name: bool(re.search(pattern, text))
                     for name, _, pattern in LLM_REASON_PATTERNS}
            primary = next((name for name, _, _ in LLM_REASON_PATTERNS if flags[name]),
                           "no rationale" if not text else "other")
            reason_flags.update(k for k, v in flags.items() if v)
            reason_primary[primary] += 1
            if h in gold_hpo:
                reason_judged[primary] += 1
                reason_confirmed[primary] += p not in gold_pairs
            rejected_rows.append(dict(
                hpo_id=h, hpo_label=hpo_label(h), sct_id=s, sct_label=sct_label(s),
                sct_tag=sct_tag(s), accepted_by_step=step_labels.get(str(pair_step.get(p)), ""),
                methods=methods_label(p), llm_pass=review_info[p][2],
                n_sct=review_n_sct.get(p), llm_confidence=llm_confidence(p),
                relation_to_llm_accepted=llm_target_relation(p), gold_status=gold_status(p),
                primary_reason=primary, llm_rationale=rationale, **flags,
            ))

        l.log(f"H. Reasons in the LLM's rationales for rejecting auto-accepted pairs "
              f"({len(rejected_auto)}): primary reason, all mentions, and how often the "
              f"expert confirms the rejection (pair is not the expert's target):")
        for name, text, _ in LLM_REASON_PATTERNS + [("other", "none of the patterns", ""),
                                                    ("no rationale", "no rationale", "")]:
            if not reason_primary[name] and not reason_flags[name]:
                continue
            l.log(f"  {text[:55] + ':':<56} primary {reason_primary[name]:5} "
                  f"({_safe_div(reason_primary[name], len(rejected_auto)):6.2%}) | mentioned "
                  f"{reason_flags[name]:5} | confirmed {reason_confirmed[name]:4} of "
                  f"{reason_judged[name]:4}")
        l.log("   Examples per primary reason:")
        shown = Counter()
        for row in rejected_rows:
            if shown[row["primary_reason"]] >= 2 or not isinstance(row["llm_rationale"], str):
                continue
            shown[row["primary_reason"]] += 1
            l.log(f"    [{row['primary_reason']}] {row['hpo_label']} → {row['sct_label']} "
                  f"({row['gold_status']})")
            l.log(f"      LLM: {row['llm_rationale'][:240]}")
        write_table(rejected_rows, "llm_rejected_auto.csv")

        # The other direction: pairs the filter did not accept, by review reason.
        l.log("I. Pairs not auto-accepted, by review reason: LLM acceptance and precision:")
        for code, reason in sorted(REVIEW_REASON.items()):
            if code == validity_accepted:
                continue
            pairs = [p for p, v in pair_validity.items() if pd.notna(v) and v == code
                     and p in review_info]
            if not pairs:
                continue
            acc = [p for p in pairs if review_info[p][0] == "ACCEPT"]
            judged = [p for p in acc if p[0] in gold_hpo]
            l.log(f"  {reason + ':':<40} reviewed {len(pairs):6} | accepted {len(acc):6} "
                  f"({_safe_div(len(acc), len(pairs)):6.2%}) | P "
                  f"{_safe_div(sum(p in gold_pairs for p in judged), len(judged)):.3f}")

    l.printHeader("Error analysis completed.")
