# Round 1 review: yeast genetics / functional-genomics validity

Reviewer: domain-genetics agent · 2026-10-28

## 1. Verdict

**Sound with revisions.** Labels, gene-disjoint splits, viable×viable evaluation and the feature groups cover the main GO-free proxies (SMF, PPI, co-expression, phenomics, KO-proteome, paralogy, orthology). Gaps:

- Linkage/marker artefacts are unhandled.
- Single-mutant fitness (SMF), the top degree predictor, is not first-class; one source is mis-specified.
- Within-complex vs between-pathway logic is not separated.
- Ortholog aggregation is biased for 1:many (WGD) cases.
- Nothing carries Ryan's module structure to unmeasured genes.

No change below alters labels, metrics or the lockbox.

Quick check (chromosome from systematic-ID letter): same-chromosome S2 pairs have mean S2 −0.063 vs +0.021 otherwise (negatives 1.94% vs 1.85%; n = 433k/904k). Only 15 same-cosmid pairs survive, so close pairs were filtered; residual linkage at longer range is plausible.

## 2. Concrete changes

**1. Gene coordinates and a linkage diagnostic (must).**
- Add PomBase chromosome coordinates (sequence annotation, CC BY) to DATA.md; compute `same_chr`, `distance_kb`; plot P(S2 < −2.3) and mean S2 vs distance before modelling.
- Distance is a **nuisance covariate, never a deliverable feature**. If a residual-linkage band exists (roughly <100–200 kb), drop those pairs from training or train with the covariate and score the universe with it set to "unlinked". Primary metric unchanged; secondary metric excludes linked pairs. Flag universe pairs inside the filter distance as "untestable by E-MAP".
- Why: linkage depresses double-mutant recovery and mimics a negative GI. Tandem paralogs are linked, so Ryan under-samples the strongest redundancy pairs (state in limitations).

**2. Haploid SMF as a first-class gene feature (must).**
- Sources: Rodríguez-López 2023 standard condition (YES, 32 °C); Koch 2012 Additional file 2, SMF column only. Kim 2010 S-Table 15 is **heterozygous-diploid** fitness (haploinsufficiency, not SMF); relabel it in §4.
- Pair features: min, max, product of SMFs.
- Why: SMF is the top correlate of negative degree in S. pombe (r = 0.48, Koch 2012), and sick strains score differently (compressed range, variance floor).

**3. Make missingness honest (must).** Inviable/DAmP training genes lack deletion SMF, phenomics and KO-proteome data, so trees learn "phenomics missing → hub" and apply it to ~80 viable genes merely absent from those screens. Add an `essential_hypomorph` indicator (training rows only; 0 at inference, constant over the universe, so allowed). Impute their SMF as "severe" rather than NaN. In E-ALLELE, compare feature importances with and without these rows.

**4. Use non-GI neighbours to carry Ryan structure to unseen genes (must, the main C2/C3 lever).**
- Current neighbour features use GI profiles, absent for unseen genes. Add fold-safe features: mean training S2 and hit rate between b and a's top-k neighbours, with neighbours from PPI, co-expression, KO-proteome, phenomics or ortholog Costanzo profile (per source, symmetrised). Add a block version: mean training S2 between the neighbourhoods of a and b.
- Why: negative GIs form complex–complex and pathway–pathway blocks (Kelley & Ideker; Costanzo 2016; Ryan Fig. 6). This is the GO-free way to say "a's complex is synthetic-sick with b's pathway".

**5. Costanzo-profile modules as a GO substitute (should, high).**
Cluster cerevisiae genes on Costanzo profile similarity into ~100–300 modules; map pombe genes through orthologs, and orphans softly via nearest non-GI neighbours. Pair features: module×module negative density in Costanzo and in fold-training Ryan pairs. Gene features: module degree and "pleiotropy" (entropy of the ortholog's negative GIs across modules), a GO-free multifunctionality proxy. Why: Ryan concludes that process cross-talk is the transferable unit.

**6. Fix ortholog aggregation for 1:many and multi-allele cases (must).**
- Min ε over k ortholog/allele pairs grows more extreme with k, inflating false positives for genes with WGD ohnologs and for essential orthologs with several TS alleles.
- Collapse alleles to one ORF value first (median ε or most reproducible allele). Then give mean, min and k-corrected min, plus `n_ortholog_pairs` and ortholog class.
- Gene features: "any ortholog essential"; "ohnolog pair is itself negative/SL in Costanzo" (buffered essential function → likely pombe hub, as ortho-essential genes have ~2.5× more GIs); an ohnolog-union profile (element-wise min ε) for profile correlation.
- Compute ortholog degree as a **rate per tested pair**, separately for NxN and ExN/ExE (essential genes have ~5× more GIs).

**7. Separate within-complex and between-pathway signal (should).**
Nonessential intra-complex GIs are mostly *positive* (Costanzo: 21% vs 5%), so a direct PPI edge is not a monotone SL signal. Add "shared PPI neighbours, no direct edge" and change 4's PPI-neighbourhood S2. Use positive GIs (S2 > 1.8) internally as co-module evidence in RYAN-fold features: alleviating partners share negative partners.

**8. Tighten the paralog definition (should).**
Separate recent, high-identity `within_species_paralog` pairs (identity bins) from ancient `other_paralog` superfamily members (e.g. kinases), which carry little redundancy. Make KO-proteome compensation directed (is protein b *up* in KO-a, and vice versa). Report paralog-pair AUPRC separately (few in pombe, which lacks the WGD).

**9. Stratify by ortholog class and lineage specificity (should).**
Report AUPRC by ortholog class (1:1 / 1:many / none-in-cerevisiae / pombe-specific in Compara). Add S. japonicus ortholog presence (PomBase file) to CONS. Why: orphans have fewer GIs, and CER features are missing for ~25% of genes.

**10. Check the Ryan-to-universe shift (should).**
Compare Ryan genes with the 1,406 never-measured viable genes on SMF, expression, conservation, nuclear localisation and phenomics hits; they may be enriched for slow growers, meiotic or lineage-specific genes. Every Ryan pair contains a (chromatin-biased) query gene, so evaluation over-represents query-type pairs. If the shift is large, add a covariate-reweighted secondary AUPRC.

**11. Universe definition (should).**
Treat condition-dependent genes that exist as Bioneer haploid deletions as full universe members (hubs fft3 and prw1 were screened as deletions). Add a secondary evaluation on viable ∪ condition-dependent pairs; keep the primary as declared. List the 166 "unknown"-viability exclusions with the reason.

**12. Hub-class gene features (should).**
Add the per-condition phenomics sensitivity vector (PCA) as gene features, not only hit count and pair correlation; DNA-damage, cell-wall and drug sensitivities mark hub classes. Top hubs include calcineurin/cell-wall genes (ptc1, prz1, pef1, kin1) besides chromatin genes. Some of that hubness may be sensitivity to PEM selection drugs or media: still predictable label, but run the diagnostic (degree vs drug sensitivity).

**13. Assay-specific artefacts from germination and sporulation (could).**
Double mutants pass through spore germination, so spore or germination genes can give assay-specific negatives. Add meiosis/sporulation induction (COXPRESdb time courses) and a "not expressed in vegetative growth" flag (such genes have low degree).

**14. Mixed-allele S2 (could).** For the 22 viable genes that also have a DAmP allele, count viable×viable pairs whose S2 averages non-deletion alleles; sensitivity analysis only.

**15. B1 form (could).** Combine propensities on the log-odds scale (logit P = α_a + α_b), not as a sum of probabilities, like degree-corrected GI null models.

**16. Kuzmin 2018/2020 trigenic/ohnolog data (could).** Allowed (not pombe GIs) as a source for ohnolog-masked interactions (change 6). Defer unless the 1:many stratum underperforms.
