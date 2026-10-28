# Round 2 review: data provenance + yeast genetics

Checks: `reviews/scratch/round2_data_domain_checks.py`.

## Round-1 disposition
Mostly addressed correctly. Still open:
- **DA3:** the HTP-only ablation is missing from §4.
- **DA1:** PomBase expression and protein files are still fetched from `latest`/rolling.
- **DA9:** M4 uses "conservation", but Compara holds pombe rows only.
- **DA7:** primary evaluation now includes condition-dependent (CD) and unknown genes. Acceptable, but see item 1.

## Changes

**1. [must] Exclude DAmP data per pair, by orientation, not per gene.**
- **The gene list is wrong both ways.**
  - rap1 has an array deletion: 649 of its 2,003 pairs have deletion data.
  - CD and unknown genes are missing from the list.
- **Universe S2 pairs that touch DAmP-queried genes** (36,892 pairs, 439 positives):

  | S2 comes from | Pairs | Positives |
  |---|---|---|
  | DAmP only | 18,345 | 177 |
  | Mixed DAmP + deletion | 17,073 | 143 |
  | Deletion only | 1,474 | 119 |

- **The DAmP-only set covers** all pairs of pot1, sod1, bud32, rhb1, hsk1, cwf20 and sap10, and 1,309 pairs of mcl1 (36 positives).
- **Rule:**
  - Build a per-pair `s2_damp_only` flag from `ryan_measurements.parquet` and exclude those pairs.
  - Keep mixed pairs, with a sensitivity check.
  - Genes never measured as deletions count as **unseen** for the class flags and the `seen_*` features.

**2. [must] Make the unknown-viability rule operational.**
- **Rule:** `unknown` AND a main-allele deletion strain (array or unlabelled query). This gives 21 genes: 19 array-only, plus dad4, urm1 and SPAC222.18.
- **Exclude sap10.** It was screened as DAmP only, which implies the deletion was presumed inviable.
- **143 unknown genes not in Ryan:** score with a flag or exclude; state which.
- **Flag "Ryan used DAmP"** for CD genes in the deliverable.
- **12 inviable genes with Ryan "deletion" arrays** (sld3, rps3…): likely wrong strains; drop from training by default (or an E-ALLELE arm).

**3. [must] BioGRID blocklist re-fetch.**
- **Steps:**
  - Re-download `BIOGRID-ORGANISM-5.0.262.tab3.zip`.
  - Verify its sha256 (`096a9e4d…`).
  - Stream only the S. pombe member.
  - For genetic rows, write only PMID → row count to a TSV.
  - Delete the zip; record the list's sha256 in DATA.md (in `download_aux_data.py`).
- **Block by PMID, never by pair.** Pair-level removal would encode the GI label.
- **Report rows lost.** Vo 2016 (PMID 26771498) is 12% of the 19,416 rows.
- **Drop non-protein–protein systems:** Affinity Capture-RNA (450), Protein-RNA (24), Co-localization (407).
- **Restore the HTP-only ablation.** It also guards against PomBase GIs that are missing from BioGRID.

**4. [should] Specify how Costanzo modules are built.**
- Own clustering, fixed seed; never the GO-derived SAFE/bioprocess labels.
- Tune the number of modules on inner splits.
- 1,074 of 3,757 universe genes (29%) lack a measured ortholog. Write the soft NBR assignment (GE5) into §4.
- 25% of genes with orthologs are 1:many. Use fractional membership.
- The "Ryan module-density" variant must be computed out-of-fold.

**5. [should] Make the allowlist granular.** List it by file **and** sheet/column:
- Rodríguez-López: exclude the GO and KEGG sheets.
- Ryan mmc1: exclude the process annotations.
- Kim: exclude "Published phenotype".

**6. [should] Complete DATA.md before freezing the protocol.**
- Complete the auxiliary entries. The allowlist test reads DATA.md.
- Pin the PomBase files to `pombase-2026-10-01`.
- For M4: drop "conservation", or re-stream Compara with S. cerevisiae rows.

**7. [could] PomBase vs Compara 116 ortholog agreement table** (PomBase orthologs are literature-informed and feed CER/NBR/modules).

**Missing biology:** no unused high-payoff source found; the gap is orphan genes (29%), served by NBR, ESM and phenomics.
