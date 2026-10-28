# Review round 1: prior work (DESIGN.md v1)

Reviewer: prior-work agent · 2026-10-28. All proposals respect PROMPT.md's data rules.

## 1. Verdict

The design follows the evidence well: gene-disjoint C1/C2/C3 splits (Park & Marcotte 2012), a degree-only baseline, per-fold Ryan features and three uses of Costanzo 2016. Three gaps matter.

1. **No cold-start training:** M2's Ryan features are never missing in training but always missing for unseen genes, and training rows get in-sample encodings.
2. **No similarity transfer:** unseen genes do not borrow Ryan information from their non-GI nearest neighbours.
3. **Edge-level transfer only:** Costanzo is used mostly edge by edge, although Ryan 2012 found module-level cross-talk transfers far better.

LITERATURE.md misses the closest prior work on this data (EMF/XSMF, Fan et al. 2020, which imputed **Ryan 2012** using S. cerevisiae data) and cross-species SL transfer to S. pombe (MUNK 2019; ETNA 2023). Its prevalence anchor (0.04) is stale; the relevant value is 0.020. None of this is fatal, but items 1–2 are needed for the C2/C3 numbers to mean anything.

## 2. Proposed changes

**1. Cold-start-aware training for every model that uses RYAN-fold features (M2, M3, M5). Priority: MUST.**
- *What:*
  - (a) Inside each training fold, compute target encodings, hit rates and neighbour scores for the training rows out-of-fold. Use a nested gene-split K=5, so that no row's features include its own label.
  - (b) For each training row, randomly mask the RYAN-fold features of one or both genes at rates matching the deployment C1/C2/C3 mix (0.37/0.48/0.15). Alternatively, train a separate model per class. Treat the choice as a pre-declared ablation.
- *Why:* As written, LightGBM never sees NaN Ryan features in training, so its C2/C3 behaviour is arbitrary, and in-sample encodings teach it to over-trust them. Cold-start pair prediction needs training that matches the test setting (Pahikkala et al., Brief Bioinform 16:325, 2015). Feature dropout is the recommender-system fix (Volkovs et al., *DropoutNet*, NeurIPS 2017); out-of-fold target statistics prevent target leakage (Prokhorenkova et al., *CatBoost*, NeurIPS 2018).

**2. Add similarity-transfer ("cold-start KNN") pair features for C2/C3. Priority: MUST.**
- *What:* Let gene a be unseen. Define its neighbours as its top-k **training** genes by similarity in each non-GI space: phenomics, KO-proteome, co-expression, ESM, and the Costanzo ortholog profile.
  - C2 feature: the mean Ryan S (and hit rate) between seen partner b and a's neighbours.
  - C3 feature: the mean over (neighbours of a) × (neighbours of b).
  - Compute per fold from training pairs only, one feature per space. Add the same to B1 (unseen-gene propensity = similarity-weighted neighbour propensity).
- *Why:* This extends profile-neighbour imputation (Ulitsky et al. 2009; Ryan et al. 2011) to genes without profiles, the standard new-drug/new-target baseline (Pahikkala 2015). The design's only neighbour feature needs a Ryan profile, so it is missing for the 63% of universe pairs that involve an unseen gene (data_inventory §8).

**3. Add module-level cross-species features. Priority: SHOULD.**
- *What:*
  - (i) A gene embedding from orthologs' Costanzo ε profiles (PCA; averaged over 1:many), available for unseen genes.
  - (ii) A neighbourhood-smoothed ε: the mean ε between the Costanzo profile neighbours of ortholog(a) and ortholog(b).
  - (iii) Module–module negative-GI density: cluster Costanzo genes by profile similarity, then take the density between the modules of ortholog(a) and ortholog(b).
- *Why:* Edge transfer is weak (different-process r = 0.03; our EDA: Pearson(S2, ε) = 0.027), whereas "functional module definitions and the level of cross-talk between different processes" are conserved (Ryan 2012; Roguev 2008).

**4. Make M4 a real transfer test. Priority: SHOULD.**
- *What:*
  - (a) Build cerevisiae counterparts of the strongest pombe features: KO proteomes (Messner et al., Cell 186:2018, 2023), chemogenomic and condition fitness (Hillenmeyer et al., Science 320:362, 2008), GFP localisation (Huh et al., Nature 425:686, 2003), and expression.
  - (b) Rank-normalise every feature within each species (SINaTRA, which raised AUC from 0.67 to 0.86).
  - (c) Add **B4**: a model trained on cerevisiae only and applied zero-shot to pombe, mirroring Koch et al. 2012 (degree r = 0.51 across species).
  - (d) Optional: ortholog-anchored aligned PPI embeddings as a feature group (MUNK, Fan et al., NAR 47:e51, 2019; ETNA, Li et al., Bioinformatics 39:btad529, 2023; Sce→Spo SL AUROC 0.74, balanced).
- *Why:* As specified, M4 pre-trains only on PPI, ESM and conservation features, so it would test only whether those few features transfer.

**5. Report results under the shift between Ryan genes and the universe. Priority: SHOULD.**
- *What:*
  - Every Ryan evaluation pair contains a hub-enriched query gene; most universe pairs contain no query gene and no Ryan gene.
  - Report AUPRC stratified by the seen partner's role and degree, and an importance-weighted AUPRC with test genes weighted by P(non-Ryan)/P(Ryan) estimated from gene features (Sugiyama et al., JMLR 8:985, 2007).
  - Report-only; neither replaces the selection score.
- *Why:* Evaluate on the population's distribution (Park & Marcotte 2012); LITERATURE §7.2 flags query-selection bias but the design does not act on it.

**6. Control for homology leakage. Priority: SHOULD.**
- *What:* Stratify C2/C3 results by each test gene's maximum sequence identity to a training gene, and run one sensitivity split keeping paralog families in one fold.
- *Why:* Near-duplicates turn "unseen" into "seen" (Park & Marcotte 2012; Hamp & Rost 2015); item 2 and ESM features would exploit this.

**7. Fit B1 as a joint additive gene-effect model. Priority: SHOULD.**
- *What:* Fit logit p(a,b) = α_a + α_b (+ β·features for unseen genes), estimated jointly with shrinkage, instead of using per-gene raw hit rates.
- *Why:* Array-only genes are tested only against hub-enriched queries (hit rate 1.74% vs 2.19% for query+array genes), so raw rates confound propensity with partner set. A degree audit must be a strong baseline (Eid et al. 2021).

**8. Use data already on disk to widen coverage. Priority: COULD.**
- *What:*
  - (a) Flagged "soft orthologs" (best Compara/ESM hit) for the ~25% of Ryan genes with no curated cerevisiae ortholog, to extend CER features.
  - (b) Disorder fraction from AlphaFold pLDDT in `data/raw/alphafold_spombe` (disorder vs degree r = 0.13 in pombe, Koch 2012).
- *Why:* More genes get the strongest gene-level priors.

**9. Add EMF/XSMF as a comparator for C1 and E-FEWSHOT. Priority: COULD.**
- *What:* Cross-species matrix factorisation with Costanzo plus a PPI kernel, using the public code lrgr/EMF.
- *Why:* Fan et al. (Bioinformatics 36:i866, 2020) imputed **Ryan 2012** with Costanzo 2010; gains were concentrated in data-scarce settings (R² 0.049 → 0.067 at 10% training), with pair-random splits only. It is the direct precedent for our C1 and few-shot setting.

**10. Neural rungs (M3). No change needed.**
The ladder and decision rule already cover the risk that additive or linear baselines match deep models for double perturbations (Ahlmann-Eltze et al., Nat Methods 22:1657, 2025).

**11. Fix LITERATURE.md. Priority: SHOULD.**
- *What:*
  - (a) Update the prevalence in §1.2, §5 and §7.3 from 0.04 to 0.020 (S2, viable×viable), and rescale the expected-AUPRC ranges.
  - (b) Add Fan 2020 (EMF), MUNK 2019, ETNA 2023, ELISL (Tepeli et al., Bioinformatics 40:btad764, 2024), Messner 2023 and Ahlmann-Eltze 2025.
  - (c) Phenomics is 129 usable conditions, not 131.
  - (d) Note that the "brief" attributions (Genes Dev, 2023, "Slorth") do not appear in PROMPT.md.
- *Why:* Later agents anchor on it.
