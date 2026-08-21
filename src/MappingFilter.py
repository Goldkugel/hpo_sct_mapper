import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from logger                 import Logger
from adapter                import BaseAdapter, writeHugeCSV
from MappingUtils           import loadOntologies, loadGold, isSCTDomainValid, evaluation, confidence_column, extract_snomed_domain, validity_column, narrow_threshold, borad_threshold, hpo_id_column, sct_id_column, min_sources, min_confidence, reference_methods, printCounts, threshold_cosine_similarity, getRepresentativeLabel
from adapter                import labelClass, childrenClass
from collections            import defaultdict
import pandas               as pd

if __name__ == "__main__":

    l = Logger()

    l.printHeader("Filterin generated mappings")

    hpo, sct, umls, hpo_ontology, sct_ontology, umls_ontology = loadOntologies(";")
    hpo = hpo[hpo[hpo_ontology.config.id_column].str.startswith("HP:")].reset_index(drop = True)

    mask = sct[sct_ontology.config.attribute_column] == childrenClass

    # 2. Extract the target column name
    val_col = sct_ontology.config.value_column

    # 3. Vectorize string concatenation on only the masked rows
    #    (We conditionally prepend 'SNOMEDCT_US:' only if it's not already prefixed)
    sct_vals = sct.loc[mask, val_col].astype(str)
    already_prefixed = sct_vals.str.startswith("SNOMEDCT_US:")

    sct.loc[mask, val_col] = (
        "SNOMEDCT_US:" + sct_vals.where(already_prefixed, sct_vals)
    )
            
    graham = loadGold("./data/input/graham/mapping_graham.csv", ";")

    l.log("Loading generated mappings...")

    mappings = pd.read_csv("./data/output/mapped/mapping.csv", sep=";")
    mappings = mappings[mappings[hpo_ontology.config.id_column].str.startswith("HP:")]

    l.log("Loading generated mappings completed.")

    l.log("Aggregating synonym-level duplicate candidates...")
    l.log(f"Row count before aggregation: {len(mappings.index):7}")

    group_cols = [hpo_ontology.config.id_column, hpo_ontology.config.value_column, hpo_ontology.config.attribute_column]

    # Keep max confidence per (HPO id, SCT id, method) - i.e. the best synonym-level match found for that pair
    mappings = (
        mappings
        .sort_values(confidence_column, ascending=False)
        .drop_duplicates(subset=group_cols, keep="first")
        .reset_index(drop=True)
    )

    l.log(f"Row count after aggregation:  {len(mappings.index):7}")
    l.log("Aggregating synonym-level duplicate candidates completed.")

    mappings[validity_column] = 1

    l.log("Status")
    evaluation(
        mappings[mappings[validity_column] == 1], 
        graham, 
        "Generated mappings:",
        hpo_ontology
    )

    printCounts(mappings, hpo_ontology, [1.0, 0.99, 0.95, 0.9, 0.8, threshold_cosine_similarity], graham, True)
    l.log("Confidence above 0.95:")
    evaluation(
        mappings[mappings["confidence"] > 0.95].reset_index(drop = True), 
        graham, 
        "Auto-accepted evaluation:",
        hpo_ontology
    )






    def evaluateRemoved(data: pd.DataFrame) -> None:
        recovered = data.merge(
            graham[[hpo_id_column, sct_id_column]],
            left_on     = [hpo_ontology.config.id_column, hpo_ontology.config.value_column],
            right_on    = [hpo_id_column, sct_id_column],
            how         = "inner"
        ).drop(columns  = [hpo_id_column, sct_id_column])

        recovered = recovered[[hpo_ontology.config.id_column, hpo_ontology.config.value_column]].drop_duplicates().reset_index(drop = True)

        l.log(f"{len(recovered.index)} flagged rows are incorrect.")

        i = 20 if len(recovered.index) > 20 else len(recovered.index)
        for j in range(0, i):
            l.log(f"\"{recovered.loc[recovered.index[j], hpo_ontology.config.id_column]}\" -> \"{recovered.loc[recovered.index[j], hpo_ontology.config.value_column]}\"")

        l.log(f"{len(data.index) - len(recovered.index)} flagged rows are correct.")
        printCounts(data, hpo_ontology, [1.0, 0.99, 0.95, 0.9, 0.8, threshold_cosine_similarity], graham, True)

    def weakMapping(data: pd.DataFrame) -> pd.DataFrame:
        """
        weakMapping removes candidate mappings that disagree with a trusted 
        reference mapping for the same HPO concept, in a way that isn't just a 
        domain-tag variant.

        A row is marked invalid only if all of these hold:

        - it's not itself a reference-method row,
        - it's still currently valid,
        - its HPO id has trusted reference evidence,
        - its target is not the same pair as the reference (i.e. it disagrees on 
            the target concept),
        - its target is in the same domain as the reference target
        - and the confidence is below 0.99
        """

        ret = data.copy()
        
        l.log("Remove weak mappings...")
        l.log(f"Current valid mapping count: {len(ret[ret[validity_column] == 1].index)}.")

        # Preferred term per SNOMED CT concept, to extract its domain tag (reuses the same lookup as the domain-mismatch step)
        sct_labels = (
            sct[sct[sct_ontology.config.attribute_column] == "label"][[sct_ontology.config.id_column, sct_ontology.config.value_column]]
            .drop_duplicates(subset=[sct_ontology.config.id_column])
            .rename(columns={sct_ontology.config.id_column: "_sct_id", sct_ontology.config.value_column: "_pt"})
        )
        # Missing labels come back as NaN after merges; normalize to None
        sct_labels["_pt"] = sct_labels["_pt"].where(sct_labels["_pt"].notna(), None)
        sct_labels["_domain"] = sct_labels["_pt"].apply(
            lambda pt: extract_snomed_domain(pt) if pt is not None else None
        )

        # Only rows that are BOTH a reference method AND still valid count as trusted references
        # (a reference row already invalidated by an earlier step, e.g. domain mismatch, must not act as ground truth here)
        reference_rows = ret[
            ret[hpo_ontology.config.attribute_column].isin(reference_methods)
            & (ret[validity_column] == 1)
        ]
        reference_pairs = reference_rows[[hpo_ontology.config.id_column, hpo_ontology.config.value_column]].drop_duplicates()
        reference_pairs["_is_reference_pair"] = True

        # Attach each reference target's domain tag
        reference_pairs = reference_pairs.merge(
            sct_labels[["_sct_id", "_domain"]],
            left_on=hpo_ontology.config.value_column, right_on="_sct_id", how="left"
        ).drop(columns=["_sct_id"])

        # Per HPO id, the set of domains its trusted reference target(s) occupy
        # (a set, since an HPO id can have multiple reference targets, e.g. xref + UMLS, possibly in different domains)
        reference_domains_by_id = (
            reference_pairs.groupby(hpo_ontology.config.id_column)["_domain"]
            .apply(lambda s: set(s.dropna()))
            .to_dict()
        )

        ids_with_reference = set(reference_rows[hpo_ontology.config.id_column])

        # Flag whether each row's (id, value) pair matches a trusted reference pair for that id
        ret = ret.merge(
            reference_pairs[[hpo_ontology.config.id_column, hpo_ontology.config.value_column, "_is_reference_pair"]],
            on=[hpo_ontology.config.id_column, hpo_ontology.config.value_column],
            how="left"
        )
        ret["_is_reference_pair"] = ret["_is_reference_pair"] == True
        # Flag whether the HPO id has ANY trusted reference evidence at all
        ret["_id_has_reference"] = ret[hpo_ontology.config.id_column].isin(ids_with_reference)

        # Attach each candidate row's own target domain tag
        ret = ret.merge(
            sct_labels[["_sct_id", "_domain"]],
            left_on=hpo_ontology.config.value_column, right_on="_sct_id", how="left"
        ).drop(columns=["_sct_id"]).rename(columns={"_domain": "_candidate_domain"})

        # A candidate "conflicts" with the reference only if it shares the SAME domain tag
        # as one of the reference targets - a different domain (e.g. disorder vs. finding)
        # is treated as a legitimate tag variant, not a weak/contradicting mapping
        def _domain_conflicts_with_reference(row):
            ref_domains = reference_domains_by_id.get(row[hpo_ontology.config.id_column])
            if not ref_domains or row["_candidate_domain"] is None:
                return False
            return row["_candidate_domain"] in ref_domains

        ret["_same_domain_as_reference"] = ret.apply(_domain_conflicts_with_reference, axis=1)

        is_reference_method = ret[hpo_ontology.config.attribute_column].isin(reference_methods)
        is_still_valid = ret[validity_column] == 1

        # Invalid only if: non-reference method, still valid, id has a reference,
        # candidate's value isn't itself a reference pair, AND candidate is in the SAME domain as the reference
        # (i.e. a genuine same-tag disagreement, not a disorder/finding-style tag variant of the same concept)
        invalid_mask = (
            ~is_reference_method
            & is_still_valid
            & ret["_id_has_reference"]
            & ~ret["_is_reference_pair"]
            & ret["_same_domain_as_reference"]
            & (ret[confidence_column] < 0.99)
        )

        # Extract rows newly invalidated by THIS step, for a correctness check against Graham's Mapping
        invalid = ret[invalid_mask].copy()

        still_valid_pairs = set(
            zip(
                ret.loc[is_still_valid & ~invalid_mask, hpo_ontology.config.id_column],
                ret.loc[is_still_valid & ~invalid_mask, val_col],
            )
        )

        pair_keys = list(zip(ret[hpo_ontology.config.id_column], ret[val_col]))
        mapping_actually_lost = pd.Series(
            [key not in still_valid_pairs for key in pair_keys],
            index=ret.index
        )

        genuinely_invalid_mask = invalid_mask & mapping_actually_lost

        invalid = ret[genuinely_invalid_mask].copy()
        evaluateRemoved(invalid)

        ret.loc[invalid_mask, validity_column] = 0
        ret = ret.drop(columns=["_is_reference_pair", "_id_has_reference", "_candidate_domain", "_same_domain_as_reference"])
        l.log(f"Removed {sum(invalid_mask)} mappings.")
        l.log("Remove weak mappings completed.")

        # Overall recall/precision of the mapping pool after this step, against the gold standard
        evaluation(
            ret[ret[validity_column] == 1],
            graham,
            "After removing weak mappings:",
            hpo_ontology
        )
        return ret

    def domainMismatch(data: pd.DataFrame) -> pd.DataFrame:
        ret = data.copy()
        l.log("Removing mappings with a domain mismatch...")
        l.log(f"Current valid mapping count: {len(ret[ret[validity_column] == 1].index)}.")
        valid_before = ret[validity_column] == 1

        # ALL labels per SNOMED CT concept (not just one arbitrary row) - a concept can have
        # multiple label rows with different semantic tags, so validity must check all of them
        sct_labels = (
            sct[sct[sct_ontology.config.attribute_column] == labelClass]
            [[sct_ontology.config.id_column, sct_ontology.config.value_column]]
            .groupby(sct_ontology.config.id_column)[sct_ontology.config.value_column]
            .apply(list)
            .reset_index()
            .rename(columns={sct_ontology.config.value_column: "_pts"})
        )

        ret = ret.merge(
            sct_labels,
            left_on=hpo_ontology.config.value_column,
            right_on=sct_ontology.config.id_column,
            how="left",
            suffixes=("", "_sct")
        )
        # Missing matches come back as NaN (float), not a list - normalize to empty list
        ret["_pts"] = ret["_pts"].apply(lambda x: x if isinstance(x, list) else [])
        ret["_domain_valid"] = ret["_pts"].apply(isSCTDomainValid)
        ret["_pt"] = ret["_pts"].apply(getRepresentativeLabel)

        domain_mismatch = ret["_domain_valid"] == False
        newly_invalidated = domain_mismatch & valid_before

        still_valid_pairs = set(
            zip(
                ret.loc[valid_before & ~newly_invalidated, hpo_ontology.config.id_column],
                ret.loc[valid_before & ~newly_invalidated, hpo_ontology.config.value_column],
            )
        )
        pair_keys = list(zip(ret[hpo_ontology.config.id_column], ret[hpo_ontology.config.value_column]))
        mapping_actually_lost = pd.Series(
            [key not in still_valid_pairs for key in pair_keys],
            index=ret.index
        )
        genuinely_invalidated = newly_invalidated & mapping_actually_lost

        hpo_labels = (
            hpo[hpo[hpo_ontology.config.attribute_column] == labelClass]
            [[hpo_ontology.config.id_column, hpo_ontology.config.value_column]]
            .drop_duplicates(subset=[hpo_ontology.config.id_column])
            .rename(columns={hpo_ontology.config.value_column: "hpo_label"})
        )
        invalid = ret[genuinely_invalidated].copy()
        invalid = invalid.merge(hpo_labels, on=hpo_ontology.config.id_column, how="left")
        invalid = invalid.rename(columns={"_pt": "sct_label"})
        invalid = invalid[[
            hpo_ontology.config.id_column, "hpo_label",
            hpo_ontology.config.value_column, "sct_label",
            hpo_ontology.config.attribute_column, confidence_column
        ]].drop_duplicates()
        l.log(f"{len(invalid.index)} mappings newly flagged invalid by domain-mismatch removal.")
        evaluateRemoved(invalid)

        ret.loc[genuinely_invalidated, validity_column] = 0
        ret = ret.drop(columns=["_pt", "_pts", "_domain_valid"] + ([sct_ontology.config.id_column] if sct_ontology.config.id_column != hpo_ontology.config.id_column else []))
        l.log("Removing mappings with a domain mismatch completed.")
        evaluation(
            ret[ret[validity_column] == 1],
            graham,
            "After removing domain mismatches:",
            hpo_ontology
        )
        return ret

    def build_child_map(data: pd.DataFrame, adapter: BaseAdapter) -> dict:
        rows = data[data[adapter.config.attribute_column] == childrenClass]
        child_map = defaultdict(set)
        for parent, child in zip(rows[adapter.config.id_column], rows[adapter.config.value_column]):
            child_map[child].add(parent)
        return child_map

    # --- Relaxed check (tightened): within one HPO concept's own valid candidates, flag a target A
    #     if some other valid target B for the SAME HPO concept is A's DIRECT child in SCT, AND:
    #       - A is NOT a reference-method mapping (xref/UMLS mappings are never flagged this way), AND
    #       - B's confidence clears a minimum threshold (weak narrower candidates can't veto anything)
    def relaxedDirectHirarchyReversal(
        data: pd.DataFrame, 
    ) -> pd.DataFrame:
        ret = data.copy()

        l.log("Removing mappings with a direct hirarchy reversal issue...")

        # Direct (non-transitive) SCT parent -> children lookup, used to detect
        # immediate broader/narrower relationships between candidate targets
        sct_child_map = build_child_map(sct, sct_ontology)

        # Only consider mappings that are currently valid (survived earlier filters)
        valid_before_relaxed = ret[ret[validity_column] == 1]
        valid_relaxed_count_before = len(valid_before_relaxed.index)
        l.log(f"Current valid mapping count: {valid_relaxed_count_before}.")

        # Reduce to the columns needed for the check and drop duplicates,
        # since multiple synonym-level rows can otherwise repeat the same (id, value, method) triple
        records = valid_before_relaxed[[
            hpo_ontology.config.id_column, hpo_ontology.config.value_column,
            hpo_ontology.config.attribute_column, confidence_column
        ]].drop_duplicates().to_dict("records")

        # Group all valid candidate targets by their HPO concept, so we only
        # compare targets that belong to the SAME HPO concept
        by_hpo_id = defaultdict(list)
        for r in records:
            by_hpo_id[r[hpo_ontology.config.id_column]].append(r)

        ret["narrower_alternative_exists"] = False
        flagged_pairs = []   # (broader_row, narrower_row) pairs that triggered the rule

        # For each HPO concept, check every pair of its candidate targets (A, B):
        # if B is a DIRECT SCT child of A, A is considered "broader" and B "narrower"
        for _, rows in by_hpo_id.items():
            if len(rows) < 2:
                continue  # need at least two candidate targets to compare
            for row_a in rows:                                  # candidate broader target A
                if row_a[hpo_ontology.config.attribute_column] in reference_methods:
                    continue                                     # guard: never flag a reference-method mapping (xref/UMLS are trusted)
                if row_a[confidence_column] >= borad_threshold:
                    continue                                     # guard: high-confidence broader mapping is protected from being flagged
                a = row_a[hpo_ontology.config.value_column]
                direct_children_of_a = sct_child_map.get(a, set())
                if not direct_children_of_a:
                    continue                                     # A has no direct children in SCT, so it can't be "broader" than anything here
                for row_b in rows:                               # candidate narrower target B
                    if row_a is row_b:
                        continue                                 # skip comparing a row to itself
                    b = row_b[hpo_ontology.config.value_column]
                    if b not in direct_children_of_a:
                        continue                                 # B is not a direct child of A, so no broader/narrower relation
                    if row_b[confidence_column] < narrow_threshold:
                        continue                                  # guard: narrower candidate too weak (low confidence) to veto A
                    if row_b[confidence_column] <= row_a[confidence_column]:
                        continue 
                    flagged_pairs.append((row_a, row_b))

        l.log(f"{len(flagged_pairs)} broader/narrower pairs found within same-HPO-concept candidate sets.")

        # 2. Collect all target keys from flagged_pairs using set comprehension
        # (This extracts the unique combinations of broader_row key values)
        target_keys = {
            (
                broader_row[hpo_ontology.config.id_column], 
                broader_row[hpo_ontology.config.value_column], 
                broader_row[hpo_ontology.config.attribute_column])
            for broader_row, _ in flagged_pairs
        }
        # 3. Zip the dataframe columns to build a matching key series or index
        # Using zip() on values is drastically faster than DataFrame.apply()
        mapping_keys = zip(
            ret[hpo_ontology.config.id_column], 
            ret[hpo_ontology.config.value_column], 
            ret[hpo_ontology.config.attribute_column]
        )
        # 4. Create a single boolean mask using set membership
        mask = [key in target_keys for key in mapping_keys]
        # 5. Apply the flag update in ONE vector operation
        ret.loc[mask, "narrower_alternative_exists"] = True

        flagged_count = int(ret["narrower_alternative_exists"].sum())
        l.log(f"{flagged_count} distinct mapping rows flagged.")

        # Check how many of the flagged (would-be-invalidated) rows are actually correct
        # per Graham's Mapping, before committing to invalidating them
        flagged_mappings = ret[ret["narrower_alternative_exists"] == True].reset_index(drop=True).copy()
        evaluateRemoved(flagged_mappings)

        # Invalidate the flagged broader mappings that are still currently valid
        would_invalidate_mask = (ret["narrower_alternative_exists"] == True) & (ret[validity_column] == 1)
        ret.loc[would_invalidate_mask, validity_column] = 0

        valid_relaxed_count_after = len(ret[ret[validity_column] == 1].index)
        l.log(f"{valid_relaxed_count_before - valid_relaxed_count_after} mappings newly invalidated by tightened relaxed hierarchy check.")
        l.log(f"Valid mapping count after tightened relaxed hierarchy check: {valid_relaxed_count_after}")

        # Overall recall/precision of the mapping pool after this step, against the gold standard
        evaluation(
            ret[ret[validity_column] == 1], 
            graham, 
            "After relaxed hierarchy check:",
            hpo_ontology
        )

        l.log("Removing mappings with a direct hirarchy reversal issue completed.")

        return ret

    def autoAccept(    
        data: pd.DataFrame, 
    ) -> pd.DataFrame:

        ret = data.copy()

        l.log("Computing auto-accept tier...")

        valid = ret[ret[validity_column] == 1]

        # --- Recompute source agreement on the CURRENT valid pool.
        support = (
            valid.groupby([
                hpo_ontology.config.id_column, 
                hpo_ontology.config.value_column
            ]).agg(
                n_supporting_sources = (
                    hpo_ontology.config.attribute_column, 
                    "nunique"
                ),
                max_confidence       = (
                    confidence_column, 
                    "max"
                ),
                methods              = (
                    hpo_ontology.config.attribute_column, 
                    lambda s: set(s)
                )
            )
            .reset_index()
        )

        is_reference_pair = support["methods"].apply(
            lambda methods: bool(methods & set(reference_methods))
        )

        is_corroborated = (
            (support["n_supporting_sources"]    >= min_sources) &
            (support["max_confidence"]          >= min_confidence)
        )

        support["auto_accept"] = is_reference_pair | is_corroborated

        l.log(f"{int(is_reference_pair.sum())} mappings auto-accepted via reference method.")
        l.log(f"{int((~is_reference_pair & is_corroborated).sum())} additional mappings auto-accepted via source corroboration.")
        l.log(f"{int(support['auto_accept'].sum())} total mappings auto-accepted.")

        # --- Attach n_supporting_sources back onto mappings (auto_accept is derived separately below) ---
        ret = ret.merge(
            support[[
                hpo_ontology.config.id_column, 
                hpo_ontology.config.value_column, 
                "n_supporting_sources"]
            ],
            on      = [
                hpo_ontology.config.id_column, 
                hpo_ontology.config.value_column
            ],
            how     = "left"
        )

        ret["n_supporting_sources"] = ret["n_supporting_sources"].fillna(0).astype(int)

        accepted_pairs = set(
            zip(
                support.loc[support["auto_accept"], 
                hpo_ontology.config.id_column], 
                support.loc[support["auto_accept"], 
                hpo_ontology.config.value_column]
            )
        )

        ret["auto_accept"] = list(zip(
            ret[hpo_ontology.config.id_column], 
            ret[hpo_ontology.config.value_column])
        )
        ret["auto_accept"] = ret["auto_accept"].isin(accepted_pairs)

        l.log("Auto-accept tiering completed.")
        l.log(f"Auto-accepted mappings: {sum((ret[validity_column] == 1) & ret['auto_accept'] == True)}")

        auto = ret[
            (ret[validity_column] == 1) & (ret["auto_accept"] == True)
        ].reset_index(drop=True).copy()
        
        evaluation(
            auto,
            graham,
            "Auto-accepted evaluation:",
            hpo_ontology
        )

        #l.log("Not auto-accepted evaluation:")
        #evaluateRemoved(ret[left].reset_index(drop = True).copy())

        return ret





    afterWeakMappings       = weakMapping(mappings)
    afterDomainMismatch     = domainMismatch(afterWeakMappings)
    afterHirarchyReversal   = relaxedDirectHirarchyReversal(afterDomainMismatch)
    afterAutoAccept         = autoAccept(afterHirarchyReversal)

    to_test = afterAutoAccept[(afterAutoAccept[validity_column] == 1) & (afterAutoAccept["auto_accept"] == 0)].reset_index(drop = True).copy()
    evaluation(
        to_test, 
        graham, 
        "Auto-accepted evaluation:",
        hpo_ontology
    )
    l.log("Still possible to save with an LLM:")
    printCounts(to_test, hpo_ontology, [1.0, 0.99, 0.95, 0.9, 0.8, threshold_cosine_similarity], graham, False)

    afterAutoAccept = afterAutoAccept.drop(["id_sct", "narrower_alternative_exists", "n_supporting_sources"], axis = 1)

    writeHugeCSV(afterAutoAccept, "./data/output/filtered/filtered.csv")

    l.printHeader("Filtering generated mappings completed")