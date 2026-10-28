# Round 1 review: data (curation, provenance, circularity)

Data reviewer, 2026-10-28. Checks: `reviews/scratch/data_review_checks.py`, `reviews/scratch/id_checks.py` (`.venv/bin/python -I <script> <project dir>`).

## 1. Verdict

**Sound, with fixes needed.** S2 < −2.3 is the right primary label (the authors' per-pair matrix and the paper's scale); feature sources follow the brief. FYPO-derived deletion viability is acceptable for **defining the universe and evaluation masks**: it is a single-mutant phenotype, not GO, GI or text mining, but must never be a feature. Gaps:
- DATA.md does not yet document the auxiliary data (two downloads have no script at all).
- One concrete circularity route is open: BioGRID low-throughput physical rows.
- Auxiliary IDs are lost silently.
- Edge-case label rules (DAmP, merged IDs, condition-dependent genes) must be fixed before results exist.

Evidence from my checks (viable×viable S2 pairs, n = 1,036,718, 20,631 positives, 2.0%):
- **BioGRID edges.** Pairs with a low-throughput (LTP) physical edge: 593, positive rate 6.7%. Pairs with a high-throughput (HTP) edge: 649, positive rate 4.0%. 9,750 of 19,416 rows are LTP (1,665 PMIDs).
- **DAmP alleles.** S2 for 10 viable genes includes a DAmP strain. For rap1, sep1 and ten1 it is DAmP only. For taz1, poz1, rif1, trt1, est1, ccq1 and cwf18 it is deletion + DAmP. These genes are in 19,260 viable×viable pairs (232 positives).
- **Merged IDs.** 4,712 viable×viable pairs (80 positives) carry an S2 value averaged across two strains (fhl1, tco89, SPAC27D7.09c).
- **Condition-dependent genes.** 51,799 S2 pairs between viable or condition-dependent genes that involve at least one condition-dependent gene. They have 2,020 positives (3.9%) and include the hubs fft3 and prw1. All are excluded from evaluation.
- **Orientation.** Positive rate is 3.37% for reciprocal pairs and 1.91% for single-orientation pairs. This follows from the shrinkage: −2.3 corresponds to raw −2.25 for reciprocal pairs but raw −3.76 for single-orientation pairs.
- **Matsuyama IDs.** 271 IDs are not current; 223 of these are PomBase synonyms (old systematic IDs) and can be recovered. Kim 2010: 47 IDs are not current.
- **PANTHER.** The domain file has family-level IDs only: 3,088 PTHR IDs and 0 `:SF` subfamilies.
- **COXPRESdb.** 0 of 720 sample titles mention double mutants or genetic interactions.
- **Undocumented downloads.** `data/raw/esm2_weights/`, `data/raw/sgd_protein/`: no script, no DATA.md entry.

## 2. Concrete changes

1. **DATA.md must cover every dataset. [must]**
   - **What:** Replace the "Auxiliary datasets" stub with one entry per dataset: URL, release/version, retrieval date, sha256, licence, allowed use and excluded parts. That means moving the content of docs/data_aux_inventory.md Appendix A into DATA.md.
   - Add **ESM-2 weights** (650M and 150M; fair-esm; MIT licence; sha256) and the **SGD proteome** (R64-5-1 `orf_trans_all`; SGD CC BY 4.0) to `download_aux_data.py` and to DATA.md.
   - Pin `pombase_quant_expression` and `pombase_protein` to `releases/pombase-2026-10-01/` instead of `latest` or the rolling `/data`, so they share the release used for IDs.
   - Mark Costanzo, Kim 2010 and Matsuyama as "licence unclear – never redistribute". The same applies to their derived feature tables.
   - **Why:** required by PROMPT.md; two files are one-off downloads.

2. **Make the FYPO carve-out explicit and switch the manifest test to an allowlist. [must]**
   - **What:** DESIGN §1/§4(iii): `FYPOviability.tsv` may be used only for the universe, evaluation masks and split stratification, and never as a model feature. The manifest test should require every feature source to appear in DATA.md with `use = feature`, instead of grepping for forbidden names.
   - **Why:** DESIGN both uses and forbids FYPO; blocklists miss unknown sources; Compara's `goc_score` (gene-order conservation) would trip a "GO" grep.

3. **Add a BioGRID PMID blocklist and an HTP-only ablation. [must]**
   - **What:** At download time, derive the set of PMIDs that contribute any *S. pombe* genetic-interaction row in BioGRID 5.0.262. Store only the PMID list and counts. Drop physical rows from those PMIDs and report how many rows are lost. Also add a PPI-HTP-only ablation of the PPI group.
   - **Why:** papers testing genetic and physical interaction on the same pair are the most direct route for pombe GI knowledge into features; LTP edges are enriched (6.7% vs 4.0% HTP, base 2.0%). Only Ryan, Roguev and Frost were checked.

4. **Use one shared ID mapper for all sources. [must]**
   - **What:** Map every auxiliary table through the Ryan resolver (exact, then PomBase synonyms/old IDs, ambiguous → drop); log unmapped counts per source; fix the inventory's "no mapping needed" claim.
   - **Why:** 223 Matsuyama genes (plus some Kim) are dropped silently.

5. **Pre-register the DAmP rule for viable genes. [should]**
   - **What:** Add a per-gene `s2_uses_damp` flag.
     - Exclude DAmP-only viable genes (rap1, sep1, ten1) from evaluation sets. Their label is a hypomorph phenotype, not a deletion phenotype. They can stay in training, under E-ALLELE.
     - Keep the mixed deletion + DAmP genes, and report a sensitivity analysis that excludes them.
   - **Why:** DESIGN §1 assumes DAmP only affects inviable genes.

6. **Exclude merged-ID pairs from evaluation. [should]**
   - **What:** Keep `involves_merged_id` pairs (4,712 viable×viable) for training only.
   - **Why:** the label averages two separately shrunk S2 values; the tco89 merge creates artificial reciprocal pairs.

7. **Decide on condition-dependent genes now. [should]**
   - **What:** Keep viable×viable as the primary evaluation set. Pre-declare a secondary evaluation that adds condition-dependent genes whose Ryan strain is a deletion. Make §1 consistent about whether the 177 genes are in the deliverable.
   - Also state how Ryan deletion strains with FYPO `unknown` viability are treated (30,589 S2 pairs). The deletion strain grew in the screen, which is single-mutant evidence of viability.
   - **Why:** the excluded set is hub-rich (3.9% positive), so the choice moves AUPRC.

8. **Stratify reporting by orientation. [should]**
   - **What:** Report C1 AUPRC separately for reciprocal and single-orientation pairs. If S2 is used as a regression target, document that it is the shrunk value.
   - **Why:** effective thresholds differ (raw −2.25 vs −3.76). Primary label unchanged.

9. **Define the data M4 needs, or cut M4's scope. [should]**
   - **What:** Specify the *S. cerevisiae* feature set for M4. The Compara file was filtered to pombe rows only, so cerevisiae phylogenetic profiles need a re-stream that keeps S. cerevisiae rows. There is no cerevisiae expression or localisation data. Either acquire and document these, or restrict M4 to PPI + ESM + profile features.
   - **Why:** DESIGN §5 assumes data that is not on disk.

10. **Document the source of "essential-in-cerevisiae". [should]**
    - **What:** Derive it from the Costanzo strain classes (TSA/DAmP alleles = essential), not from SGD phenotype annotations, which are literature-curated (APO).

11. **Add an ortholog-source sensitivity check. [could]**
    - **What:** Recompute CER features with Compara 116 (sequence-only) orthologs.
    - **Why:** PomBase curation is literature-informed; low risk, untested.

12. **Handle missingness that encodes essentiality. [could]**
    - **What:** Inviable training genes always lack deletion-based features (PHEN, KOPROT). In E-ALLELE, check that missingness is not an essentiality proxy (mask/impute for inviable genes).
    - **Why:** the shortcut cannot transfer to the all-viable universe.

13. **No change needed for these sources:**
    - **PANTHER:** family-level IDs only. Never use PANTHER names or PAINT GO annotations.
    - **Ensembl Compara:** sequence-based gene trees.
    - **COXPRESdb:** expression matrices only. Record the study accessions in DATA.md.
    - **Rodríguez-López** fitness/discretized sheets: single-mutant data.
    - **Öztürk:** single KO strains, but verify that `strainType` contains only KO and WT.
    - **Kim 2010:** excluding "Published phenotype".
    - **Matsuyama:** raw calls.
