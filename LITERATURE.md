# LITERATURE.md: evidence base for S. pombe synthetic-lethality prediction

Compiled 2026-10-28 by the literature-review agent. This is the evidence base for DESIGN.md.

**Verification legend** (applies to every factual claim below):
- **[V]**: verified this session from the source's full text or supplementary text (PMC/BioC full text, the publisher supplement, or the official data page).
- **[A]**: verified from the abstract or bibliographic record only (Europe PMC).
- **[R]**: recalled from background knowledge and **not verified** this session. Treat as unverified.
- **[P]**: preliminary number from the in-progress data-discovery outputs in `data/processed/` (not produced by me and not yet reviewed). Use only as orientation.

**Provenance note (corrections applied 2026-10-28, see changelog at the end).** Statements below that earlier said "the brief" now cite `PROMPT.md` only where PROMPT.md really says it. PROMPT.md fixes: Ryan 2012 as training data; S < −2.3 and AUPRC as the metric; Costanzo 2016 may be used; no fission-yeast GI data from any source (other than Ryan 2012); no text mining or derived resources such as STRING; no Gene Ontology. **PROMPT.md names no other paper or tool** (no Koch 2012, Feng 2024, "Slorth", Genes Dev or 2023 citation). Any further exclusion (BioGRID genetic rows, FYPO multi-allele phenotypes, Complex Portal, NET-FF, GeneMANIA/HumanNet) is a project **interpretation** of those rules and is labelled as such.

Project constraints that shape this review (from PROMPT.md): the only fission-yeast genetic-interaction (GI) data allowed is Ryan et al. 2012; no text mining or text-mining-derived resources (e.g. STRING); no Gene Ontology (GO). The budding-yeast Costanzo et al. 2016 map is explicitly allowed. The target is S-score < −2.3, evaluated by AUPRC, including on genes unseen or rarely seen in training.

---

## 1. The training data: Ryan et al. 2012 (S. pombe E-MAP)

**Citation.** Ryan CJ, Roguev A, Patrick K, Xu J, Jahari H, Tong Z, Beltrao P, Shales M, Qu H, Collins SR, Kliegman JI, Jiang L, Kuo D, Tosti E, Kim HS, Edelmann W, Keogh MC, Greene D, Tang C, Cunningham P, Shokat KM, Cagney G, Svensson JP, Guthrie C, Espenshade PJ, Ideker T, Krogan NJ. *Hierarchical modularity and the evolution of genetic interactomes across species.* Mol Cell 46(5):691–704 (2012). DOI 10.1016/j.molcel.2012.05.028; PMID 22681890; PMCID PMC3380636. [A for the bibliographic record. The author list is partly [V] from the supplement header and the middle authors are [R].]

### 1.1 Design and scale [V, main text and supplement]
- The **PEM (Pombe Epistasis Mapper)** system was used. "We screened **953 alleles** (Table S1) of **876 genes** against a fission yeast mutant library containing **more than 2000 deletions**", giving "an E-MAP containing **~1.6 million pairwise measurements** (Datasets S1, S2)."
- **Query vs array.** Queries are NAT-marked. The array is the **Bioneer** KAN-marked deletion library (Fig. 1B legend distinguishes genes that are "array only", "query only" and "both"). Query strains for genes in the Bioneer library were made by marker switching (G418→NAT). The supplement states: "**Many of the genes in our E-MAP are only on one axis (Array or Query)**, and thus, we do not have a measurement for the direct interaction of many pairs."
- **Coverage.** The abstract says "phenotypic signatures for **~60% of the non-essential genome**". The results text says "genetic interaction profiles for **~50% of the genome**". Both statements are verbatim. The genes are "broadly conserved across eukaryotes, with subsets that are fungal- and fission yeast-specific".
- **Gene classes are not only deletions.** The supplement says that for the cross-species conservation estimates "**only deletion alleles were used** … to prevent errors resulting from comparing **hypomorphic alleles of an essential gene** to a deletion". So the dataset contains hypomorphic alleles of essential genes, e.g. a *prp43-DAmP* strain appears in the main text. That there are more alleles (953) than genes (876) implies some genes carry several alleles.
  - [P] The data agent's in-progress parse classifies about 230k of 1.55M measurements as DAmP-allele queries, with smaller sets of point mutants, ts-like alleles and truncations. The gene table lists about 151 genes as inviable (essential). The same parse gives 876 query alleles, 873 query genes and about 1,950 array genes, about 408 genes on both axes, about 468 query-only and about 1,547 array-only. These counts differ from the paper's 953/876 and must be reconciled before use.
- **Replicates and quality control.** Data were collected "in batches of 25–35 queries", and each batch was "normalized and scored separately". Strains with mis-annotated genotype were removed using linkage curves. Mutants "with very noisy profiles and/or poor reproducibility between replicate screens" were pruned. "**Data points caused by genetic linkage were removed** from each profile using an adaptive threshold procedure". Consistency was checked with "marker-swap and replicate experiments" (A×B vs B×A). **Fig. S1A gives a replicate / marker-swap correlation of r = 0.4** over all pairs. This is a key noise-ceiling number. The exact number of technical replicates per pair is **not stated** in the text I could read [unverified].
- **Data and web resources.** Datasets S1/S2 hold the scores and Dataset S3 the profile similarities. The original web site `interactomecmp.ucsf.edu/pombe2012` is probably dead [R]. The data agent retrieved the Cell Press supplementary files (mmc1–mmc8).

### 1.2 S-score definition and thresholds
- The scores come from the published E-MAP MATLAB toolbox (Collins et al. 2006; the S-score). **Collins SR, Schuldiner M, Krogan NJ, Weissman JS. Genome Biol 7:R63 (2006).** DOI 10.1186/gb-2006-7-7-r63; PMID 16859555. [V]:
  - S = (μ_Exp − μ_Cont) / sqrt(s_Var/n_Exp + s_Var/n_Cont). It is a t-statistic-like score. μ_Cont is the **median normalized colony size of all double mutants containing that array mutant**, so the expectation is set empirically per array strain rather than from multiplicative single-mutant fitness. Variances have minimum bounds, and n_Cont is fixed (6 in that paper).
  - **Consequence:** an S-score mixes effect size with measurement confidence. Pairs with more replicates or lower variance get larger |S| for the same effect, and the score is not symmetric in query and array.
- **Thresholds in Ryan 2012** [V]: the hierarchical conservation analysis uses **S < −2.3 for negative** and **S > 1.8 for positive** interactions. So the project's −2.3 cut-off matches the paper.
- **Earlier conventions:** Roguev 2008 used **S ≤ −2.5 / S ≥ 2.0** [V]. Collins 2006 used −2.5 (and −6 for "strong") [V]. Koch 2012 counted S. pombe negative degree at S ≤ −2.5 [V].
- **Prevalence (corrected).** The project's primary label is **Dataset S2 `s_score` < −2.3**, one value per gene pair. Its prevalence is **0.0188** (25,084 of 1,337,033 pairs with S2; `docs/data_inventory.md`), and about **2.0% in viable×viable pairs** (the target universe). **Random-classifier AUPRC ≈ 0.019–0.020.** [P, from the data agent's tables.]
  - The earlier figure of 0.04 (4.4% of raw S1 measurements, 4.31% of raw-mean pairs) is the **raw-score prevalence**. It is **not** the label prevalence. S2 is pseudo-averaged and shrunk for single-orientation pairs (95% of pairs), so S2 = −2.3 corresponds to raw ≈ −3.76 (DATA.md, decision 1).
  - Do not compare AUPRC values across the two label definitions.

### 1.3 Main findings relevant to prediction [V unless marked]
- **Modules.** Profile similarity (Pearson correlation transformed by logistic regression into a co-complex probability, assuming a 1:300 true:false ratio) recovers complexes. The paper reports 297 modules (sizes 2–26) and assigned function to 144 uncharacterised genes.
- **Network features conserved in both species** (Fig. 5):
  - Interacting pairs are enriched for shared deletion phenotype, same process and same complex.
  - **Members of protein complexes have higher GI degree.**
  - **Sequence orphans have fewer GIs.**
  - **"Ortho-essential" genes** (non-essential in one species, essential ortholog in the other) have **~2.5× more GIs**. 83% of 1:1 orthologs have conserved dispensability.
  - Each of these is a **gene-level prior that transfers to unseen genes**.
- **Hierarchical conservation.** Quantified on 1:1 orthologs and deletion alleles only, with the orthology list manually curated by Wood 2006. The raw cross-species agreement was divided by the within-*S. cerevisiae* SGA-vs-E-MAP reproducibility:

  | Category (negative GIs, S < −2.3) | Raw between-species % | Within-species reproducibility % | Normalized conservation % |
  |---|---|---|---|
  | All pairs | 9 | 37 | 24 |
  | Different process | 6 | 31 | 19 (main text: 15) |
  | Same process | 19 | 50 | 38 |
  | Same complex | 42 | 62 | 68 |

  - Positive GIs (S > 1.8): same complex 70%, same process 58%, different process 19% (main text). The supplement table gives an "all" value of 27%.
  - There is a small discrepancy for "different process": the main text says 15% and the supplement table appears to say 19%. The PDF table extraction may be garbled, so do not rely on the second digit.
  - **Score correlation across species** (Fig. 6B): same complex r = 0.46, same process r = 0.16, **different process r = 0.03**.
  - Chromatin/Transcription and Mitosis are each >1.4-fold enriched for GIs in both species, but only **~25% of the individual interactions between them are conserved**.
  - **Implication:** direct transfer of *S. cerevisiae* pair scores helps mainly for intra-complex and intra-process pairs, which are a small fraction of all pairs. Process-level cross-talk is conserved better than individual edges. The authors conclude that "information collected from model systems about connections between individual genes may not be as useful as inferences derived from functional module definitions and the level of cross-talk between different processes."
- **Hub processes.** In S. pombe, Chromatin/Transcription, Mitosis and Mitochondrion organization are "hub processes". Amino-acid metabolism and transmembrane transport have very few GIs.
- **The paper's own S. cerevisiae comparator** was an integrated map (Costanzo 2010 SGA plus E-MAPs), with the SGA scores **non-linearly rescaled to the S-score range** (500 overlapping bins plus a spline fit; the distributions matched in −10…5). If we map ε to S-scale we can reuse this approach. Rank-based features avoid the issue.

### 1.4 Known biases of the Ryan 2012 data
1. **Gene selection.** It builds on the chromosome-function E-MAP (Roguev 2008, 550 genes chosen from a budding-yeast chromatin E-MAP). Whether the Roguev 2008 data are physically merged into the 2012 matrix is **unverified**. The 2012 Perspective calls the screen broadly unbiased, but queries are enriched for conserved genes. Expect an over-representation of nuclear/chromatin/transcription genes among queries [inference from Fig. 1 and the screen history; verify against Table S1].
2. **Query/array asymmetry.** Query genes have full profiles of about 2,000 measurements. Array-only genes have about 876 measurements, all against queries. A gene's "seen-ness" therefore differs between axes.
3. **Missing not at random.** Linked (nearby) gene pairs were removed, noisy mutants were pruned, and essential genes enter only as hypomorphs.
4. **Noise.** Replicate r = 0.4, and the S-score depends on the replicate count. Even for *S. cerevisiae*, only 37% of E-MAP negatives (all pairs) reproduce in SGA at the same threshold (Ryan 2012 supplement). **Label noise caps attainable precision.**
5. **Batches.** Queries were run in batches of 25–35. There may be residual batch effects in query-level negative-interaction rates.

## 2. Earlier S. pombe GI maps (context only; their data will NOT be used)
- **Roguev A, et al.** *Conservation and rewiring of functional modules revealed by an epistasis map in fission yeast.* Science 322:405–410 (2008). DOI 10.1126/science.1162609; PMID 18818364. [V]
  - About 118,000 double mutants among **550 chromosome-function genes**. 239 one-to-one orthologs were compared with *S. cerevisiae*.
  - Negative GIs (S ≤ −2.5) were 17.3% conserved overall and 33% among functionally co-annotated pairs. Positive GIs between physically associated proteins were >50% conserved.
  - Profile correlation of orthologous pairs: r = 0.14 overall and **r = 0.60 for PPI pairs**.
  - Conclusion: conservation is at the level of the module/complex.
- **Dixon SJ, et al.** *Significant conservation of synthetic lethal genetic interaction networks between distantly related eukaryotes.* PNAS 105:16653–16658 (2008). DOI 10.1073/pnas.0806261105; PMID 18931302. [A]
  - SpSGA on about 220 genes. About 29% of GIs were common to both species, and 23% of literature-curated S. pombe interactions were conserved. They defined a "conserved yeast network" of 106 genes and 144 interactions.
- **Frost A, et al.** *Functional repurposing revealed by comparing S. pombe and S. cerevisiae genetic interactions.* Cell 149:1339–1352 (2012). DOI 10.1016/j.cell.2012.04.028; PMID 22682253. [A]
  - An E-MAP covering about 40% of nonessential S. pombe genes. It confirmed widespread conservation within and between complexes, but found "functional repurposing" of some orthologous complexes.
  - These data are **excluded** by the project rules. They are also the most likely source of hidden leakage, through curated resources that ingest them (BioGRID genetic, PomBase genetic interactions, FYPO multi-allele phenotypes); see §6.

## 3. The budding-yeast reference: Costanzo 2016 and related work

### 3.1 Costanzo et al. 2016
**Costanzo M, VanderSluis B, Koch EN, Baryshnikova A, et al.** *A global genetic interaction network maps a wiring diagram of cellular function.* Science 353:aaf1420 (2016). DOI 10.1126/science.aaf1420; PMID 27708008; PMCID PMC5661885. [V]
- About 23 million double mutants covering 5,416 genes (about 90% of yeast genes as query and/or array). About 550,000 negative and 350,000 positive GIs. About 120,000 of these are between essential-gene pairs.
- **Three maps:**
  - **NxN**: nonessential deletion queries × the deletion mutant array (DMA).
  - **ExN/NxE**: TS-allele queries × DMA, and deletion queries × the TS array (TSA).
  - **ExE**: TS × TS.
  - There are 2,001 TS strains for 868 essential genes, plus DAmP alleles.
- **ε score:** ε = f_ij − f_i·f_j (multiplicative model; Baryshnikova A, et al. Nat Methods 7:1017–1024 (2010), DOI 10.1038/nmeth.1534, PMID 21076421 [A]. The formula itself is [R]).
- **Data files (official supplement page, boonelab.ccbr.utoronto.ca/supplement/costanzo2016) [V]:**
  - **S1**: pairwise format, 430 MB. Columns: query ORF, query gene name, array ORF, array gene name, array type (DMA/TSA) and temperature (26/30 °C), ε, P-value, **query SMF, array SMF**, DMF, DMF s.d.
  - **S2**: matrix format.
  - **S3**: **profile-similarity (Pearson) matrices** for ExE, NxN and the global network, with allele-level and ORF-level headers.
  - **S9**: per-strain degree. **S12**: protein complex standard.
  - The S1/S2 files are raw and "should be filtered for specific applications". Strains are identified at **allele level**, so one ORF can have several strains.
- **Thresholds (verbatim from the supplement page) [V]:**
  - **lenient**: P < 0.05
  - **intermediate**: P < 0.05 and |ε| > 0.08
  - **stringent**: P < 0.05 and "0.16 < ε < −0.12", which should be read as ε < −0.12 (negative) or ε > 0.16 (positive)
  - On average a gene has about 100 negative interactions (2% of genes tested) and about 65 positive (1%) at the intermediate threshold.
- **Profile similarity:** PCC > 0.4 corresponds roughly to complex/pathway, 0.2–0.4 to bioprocess, and 0.05–0.2 to compartment. At PCC ≥ 0.2, 3.12% of ExE pairs but only 0.12% of NxN pairs are connected.
- **Hubs and degree:**
  - **Single-mutant fitness is highly correlated with GI degree.**
  - Hubs tend to encode conserved, multifunctional, highly expressed, abundant proteins with many physical interactions.
  - Low-degree genes have higher dN/dS and more expression variance. About 1,000 genes (about 20%) show few GIs.
  - Essential genes have about 5× more interactions than nonessential genes.
- **Negative interactions are functionally coherent.**
  - They are enriched within bioprocesses. For nonessential genes they are also enriched between processes.
  - Nonessential complexes are more often enriched for *positive* (21%) than negative (5%) intra-complex GIs.
  - 50% of physically interacting essential pairs share a negative GI (about 10-fold enrichment).
  - Negative GIs often occur as coherent complex–complex blocks: 4% of nonessential complex pairs are connected by negative interactions.

**Relevance:**
1. Ortholog-mapped ε, and ortholog-mapped **profile similarity**, are allowed cross-species pair features.
2. *S. cerevisiae* SMF and degree are allowed gene features.
3. The ExN/TS data matter because S. pombe-nonessential genes can have *S. cerevisiae*-essential orthologs (ortho-essential genes). Their cerevisiae information is only in the TS/DAmP screens.

### 3.2 Koch et al. 2012
(An earlier draft attributed this paper to Genes Dev and to "the brief". PROMPT.md cites no paper by name; the journal is **Genome Biology**.)
**Koch EN, Costanzo M, Bellay J, Deshpande R, Chatfield-Reed K, Chua G, D'Urso G, Andrews BJ, Boone C, Myers CL.** *Conserved rules govern genetic interaction degree across species.* Genome Biol 13:R57 (2012). DOI 10.1186/gb-2012-13-7-r57; PMID 22747640; PMCID PMC3491379. [V]
- Regression trees on 16 gene features predict negative GI degree.
  - *S. cerevisiae*, held-out genes: r = 0.80.
  - Trained on *S. cerevisiae* and applied to *S. pombe* (548 genes, Roguev 2008 degrees): **r = 0.51**.
  - The *S. pombe*-trained model was **not** significantly better than the cross-species model.
  - Performance held for S. pombe-specific genes.
  - Validation used 8 new genome-wide S. pombe SGA queries (about 23,000 pairs): r = 0.40.
- **Per-feature correlation with negative degree, S. pombe vs S. cerevisiae (Table 1):**

  | Feature | S. pombe r | S. cerevisiae r |
  |---|---|---|
  | **Single-mutant fitness defect** | **0.48** | 0.75 |
  | Multifunctionality (GO count; **forbidden** for us) | 0.30 | 0.26 |
  | PPI degree | 0.20 | 0.15 |
  | Expression variation | −0.15 | −0.18 |
  | Disorder | 0.13 | 0.11 |
  | Copy number (paralogs) | −0.08 | −0.12 |
  | dN/dS | −0.05 | −0.11 |
  | Conservation | 0.07 | 0.16 |
  | Expression level, CAI, length, domain counts, co-expression degree | ≈0 | — |

  SMF is the dominant predictor in both species, and SMF itself is poorly conserved between orthologs.
- **Resource:** the paper reports genome-wide S. pombe single-mutant fitness from an SGA control screen (neutral NAT-marked query × deletion collection), in Additional file 2. **Caution:** that file also contains Roguev-2008-derived S. pombe GI degree (forbidden) and GO multifunctionality (forbidden). Only the fitness and sequence-derived columns are usable.

### 3.3 Higher-order and duplicate-gene work (context)
- **Kuzmin E, et al.** *Systematic analysis of complex genetic interactions.* Science 360:eaao1729 (2018). DOI 10.1126/science.aao1729; PMID 29674565. [A] About 200,000 triple mutants. Trigenic interactions are weaker and link more distant bioprocesses than digenic ones. This is not directly useful for us.
- **Kuzmin E, et al.** *Exploring whole-genome duplicate gene retention with complex genetic interaction analysis.* Science 368:eaaz5667 (2020). DOI 10.1126/science.aaz5667; PMID 32586993. [A] The *S. cerevisiae* paralog redundancy it studies is largely WGD-derived. S. pombe did not undergo that WGD, so paralog-based SL priors will transfer only partially [R/inference].

## 4. Methods for predicting SL / negative GIs

### 4.1 Classic yeast predictors (feature integration)
- **Wong SL, et al.** *Combining biological networks to predict genetic interactions.* PNAS 101:15682–15687 (2004). DOI 10.1073/pnas.0406614101; PMID 15496468. [A] Probabilistic decision trees ("MNDT") integrating localization, mRNA expression, physical interaction, protein function and network topology. Pandey 2010 states [V] that MNDT used "SSL-dependent features (referred to as 2-hop features)". These features use the known SL network itself, for example "A is SL with C and C interacts physically with B". They were the most informative but are only available for well-studied genes [R for the ranking].
- **Qi Y, Suhail Y, Lin YY, Boeke JD, Bader JS.** *Finding friends and enemies in an enemies-only network: a graph diffusion kernel…* Genome Res 18:1991–2004 (2008). DOI 10.1101/gr.077693.108; PMID 18832443. [A] Diffusion kernel on the SL network alone gives about 50% precision at 20–50% recall genome-wide. The method is transductive and needs both genes to be in the network (C1 only).
- **Paladugu SR, Zhao S, Ray A, Raval A.** *Mining protein networks for synthetic genetic interactions.* BMC Bioinformatics 9:426 (2008). DOI 10.1186/1471-2105-9-426; PMID 18844977. [A] SVM on PPI graph-topology features. Reports >85% sensitivity and specificity on a balanced evaluation.
- **Chipman KC, Singh AK.** *Predicting genetic interactions with random walks on biological networks.* BMC Bioinformatics 10:17 (2009). DOI 10.1186/1471-2105-10-17; PMID 19138426. [A] Random-walk proximity plus a decision tree gives TPR 95% at FPR 10% in *S. cerevisiae*. Including known non-interactions helped. The evaluation is balanced, ROC-based and pair-random.
- **Pandey G, Zhang B, Chang AN, Myers CL, Zhu J, Kumar V, Schadt EE.** *An integrative multi-network and multi-classifier approach to predict genetic interactions.* PLoS Comput Biol 6:e1000928 (2010). DOI 10.1371/journal.pcbi.1000928; PMID 20838583. [V]
  - 152 "SL-independent" features (expression, PPI, GO, pathways, phenotype, localization) and a classifier ensemble.
  - The most discriminative single features were pathway co-membership and GO semantic similarity. Both are **forbidden or curated for us**.
  - SL-independent features: AUC 0.741. All features including SL-dependent ones: AUC 0.837.
  - **On the 337 least-connected SL pairs, with training pairs sharing no gene, SL-independent features won (AUC 0.851 vs 0.819).** This is early evidence that SL-network features do not generalize to poorly studied genes.
  - Precision was 49% at 14% recall (10-fold CV) and 41% on 730 newly deposited SL pairs. The base rate was about 7% (SGD SL vs non-SL), so these numbers are prevalence-specific.
  - Non-PPI features outperformed PPI-only features (precision 36.7% vs 17.8% at 20% recall).
- **Ulitsky I, Krogan NJ, Shamir R.** *Towards accurate imputation of quantitative genetic interactions.* Genome Biol 10:R140 (2009). DOI 10.1186/gb-2009-10-12-r140; PMID 20003301 [A]. **Ryan C, Cagney G, Krogan N, Cunningham P, Greene D.** *Imputing and predicting quantitative genetic interactions in epistatic MAPs.* Methods Mol Biol 781:353–361 (2011). DOI 10.1007/978-1-61779-276-2_17; PMID 21877290 [A].
  - These are E-MAP matrix-completion methods (KNN and integrated imputation). They are strong for C1 (both genes profiled) and inapplicable to C3.
  - Imputing an entry from same-matrix neighbours is legitimate for seen genes, provided neighbour profiles are computed from **training entries only**.
- **Ma J, et al.** *Using deep learning to model the hierarchical structure and function of a cell* (DCell). Nat Methods 15:290–298 (2018). DOI 10.1038/nmeth.4627; PMID 29505029. [A] A GO-structured network predicting double-mutant growth. **Not usable** (GO).

### 4.2 Cross-species transfer
- **Jacunski A, Dixon SJ, Tatonetti NP.** *Connectivity homology enables inter-species network models of synthetic lethality* (SINaTRA). PLoS Comput Biol 11:e1004506 (2015). DOI 10.1371/journal.pcbi.1004506; PMID 26451775. [V] **Directly relevant: they predicted S. pombe SL from S. cerevisiae.**
  - Gene-pair features: 20 PPI-topology parameters (degree, betweenness, closeness, shared neighbours, communicability, …), **rank-normalized within each species**. Random forest; labels are BioGRID SL (13,196 cerevisiae, 628 pombe).
  - *S. cerevisiae*→*S. pombe*: AUC 0.67 with raw parameters and **0.86 with rank normalization**.
  - Baselines: GO functional similarity AUC 0.81; **ortholog-pair homology transfer AUC 0.60** on homologous pairs (OR = 145) and 0.52 genome-wide.
  - Within-species CV AUC was 0.92–0.93.
  - **Caveats:**
    - The evaluation used 1:1 balanced random negatives (untested pairs treated as non-SL).
    - Labels are literature-curated, so there is strong study/degree bias. The authors show the score correlates with degree and "popularity".
    - At a realistic 1:1000 SL ratio, median PPV is about 17% at score ≥ 0.95.
    - Within-complex pairs reach PPV 0.27.
  - **Lesson:** rank-normalizing gene-level network features across species is a cheap, useful trick, and **naive ortholog transfer of individual SL edges is weak**, consistent with Ryan 2012.
- **Benstead-Hume G, Chen X, Hopkins SR, Lane KA, Downs JA, Pearl FMG.** *Predicting synthetic lethal interactions using conserved patterns in protein interaction networks* (SLant). PLoS Comput Biol 15:e1006888 (2019). DOI 10.1371/journal.pcbi.1006888; PMID 30995217. [V]
  - Random forest on PPI-topology plus GO features across five species. **The PPIs come from STRING and the features include GO, so both are forbidden for us.**
  - In-species AUC was 0.889 for S. pombe. **Cross-species *S. cerevisiae*→*S. pombe* AUC was 0.717.** A consensus model gave 0.92 (balanced sets).
  - Pairwise features (shortest path, cohesion, shared community) were more discriminative than node-wise features. A tool or paper named "Slorth" is **not mentioned in PROMPT.md** and I could not find or verify one; do not cite it.
- **Conde-Pueyo N, Munteanu A, Solé RV, Rodríguez-Caso C.** *Human synthetic lethal inference as potential anti-cancer target gene detection.* BMC Syst Biol 3:116 (2009). DOI 10.1186/1752-0509-3-116; PMID 20015360. [A] Phylogenetic (ortholog) transfer of yeast SL edges to human, using BioGRID, Ensembl/NCBI homology and GO.
- **Srivas R, et al.** *A network of conserved synthetic lethal interactions for exploration of precision cancer therapy.* Mol Cell 63:514–525 (2016). DOI 10.1016/j.molcel.2016.06.022; PMID 27453043. [A] About 169,000 yeast interactions among orthologs of tumour suppressors and drug targets were screened, then tested in HeLa. **Interaction stability across environments and shared gene function increased the chance that an SL interaction transferred across species.**
- **Dixon 2008, Roguev 2008 and Ryan 2012** (§1–2) give the empirical transfer rates: about 17–29% of individual negative GIs are conserved overall, and 68% within complexes after normalization.

### 4.3 Deep-learning SL predictors (mostly human/cancer)
All bibliographic details are [A]. All rely on a dense known-SL graph and most use GO/PPI.
- **SL²MF**: Liu Y, Wu M, Liu C, Li XL, Zheng J. IEEE/ACM TCBB 17:748–757 (2020). DOI 10.1109/TCBB.2019.2909908; PMID 30969932. Logistic matrix factorization plus PPI/GO.
- **DDGCN**: Cai R, Chen X, Fang Y, Wu M, Hao Y. Bioinformatics 36:4458–4465 (2020). DOI 10.1093/bioinformatics/btaa211; PMID 32221609. Dual-dropout GCN on the SL graph only.
- **GCATSL**: Long Y, Wu M, Liu Y, Zheng J, Kwoh CK, Luo J, Li X. Bioinformatics 37:2432–2440 (2021). DOI 10.1093/bioinformatics/btab110; PMID 33609108.
- **KG4SL**: Wang S, Xu F, Li Y, et al. Bioinformatics 37:i418–i425 (2021). DOI 10.1093/bioinformatics/btab271; PMID 34252965. Knowledge-graph GNN.
- **SLMGAE**: Hao Z, Wu D, Fang Y, Wu M, Cai R, Li X. IEEE J Biomed Health Inform 25:4041–4051 (2021). DOI 10.1109/JBHI.2021.3079302; PMID 33974548. Multi-view graph auto-encoder (SL main view, with PPI and GO as support views).
- **PTGNN**: Long Y, Wu M, Liu Y, et al. Bioinformatics 38:2254–2262 (2022). DOI 10.1093/bioinformatics/btac100; PMID 35171981. Pre-trained GNN with sequence features.
- **NSF4SL**: Wang S, Feng Y, Liu X, et al. Bioinformatics 38:ii13–ii19 (2022). DOI 10.1093/bioinformatics/btac462; PMID 36124790. Negative-sample-free contrastive learning.
- **Recent examples:** KR4SL-type high-order KG GNN (Chen X, et al. Brief Bioinform 26:bbaf142, 2025; PMID 40194555) and GiGCN (Wang Q, et al. Brief Bioinform 27:bbag470, 2026; PMID 42704265). GiGCN reports AUROC 0.978 / AUPRC 0.944 for a joint SL/SV signed-graph model that uses GO-derived similarity. These headline numbers come from balanced or near-balanced pair-random settings and **should not be compared with our gene-disjoint AUPRC**.

### 4.4 The key benchmark: Feng et al. 2024
**Feng Y, Long Y, Wang H, Ouyang Y, Li Q, Wu M, Zheng J.** *Benchmarking machine learning methods for synthetic lethality prediction in cancer.* Nat Commun 15:9058 (2024). DOI 10.1038/s41467-024-52900-7; PMID 39428397. Author correction: Nat Commun 16:1619 (2025), PMID 39948334. [V] (An earlier draft said "2023, per the brief"; PROMPT.md does not cite this paper, and the publication year is 2024.) Feng et al. call the splits **CV1/CV2/CV3**, not C1/C2/C3.
- **Setup:** 12 methods (3 matrix-factorization and 9 GNN) on SynLethDB 2.0 (9,845 genes, 35,913 SL pairs).
- **Splits:** CV1 = both genes seen; CV2 = exactly one gene unseen ("semi-cold start"); CV3 = both unseen.
- **Results (F1 at 1:1 random negatives):**
  - Best CV1 F1 ≈ 0.88 (SLMGAE). This drops to about 0.78 in CV2 and **0.67–0.74 in CV3**.
  - At 1:1 balance, **F1 = 0.667 is exactly what predicting "all positive" achieves**. Several methods (SL2MF, CMFW) sit at that value in CV3, so they are at chance.
  - **NDCG@10 collapses to ≈0.00–0.04 in CV3** for every method.
- **Imbalance:** raising the negative ratio from 1:1 to 1:5/1:20/1:50 lowers F1. Some methods (CMFW, MGE4SL) degrade sharply; SLMGAE is the most robust. Most published methods were evaluated only at 1:1 in CV1.
- **Negative sampling:** sampling negatives with negative co-expression inflates classification scores, because known SL pairs are biased toward positive co-expression. That is an artifact of the negative sampler, not real signal.
- **Label quality:** excluding computationally predicted SL labels improved all methods.
- **Model choice:** DDGCN, which uses only SL labels, matched multi-source models in CV1. This suggests the models mostly learn graph topology and degree.
- **Overall message for us:** published SL deep learning is largely **transductive** and has **near-zero skill on unseen genes**. Since our task requires unseen-gene predictions, we must rely on gene-intrinsic and cross-species features, not SL-graph embeddings alone.

### 4.5 Bias-aware and feature-driven SL prediction
- **Seale C, Tepeli Y, Gonçalves JP.** *Overcoming selection bias in synthetic lethality prediction.* Bioinformatics 38:4360–4368 (2022). DOI 10.1093/bioinformatics/btac523; PMID 35876858. [A] SL data show strong gene-selection bias. **Topology-driven models (graph, matrix factorization) are especially sensitive to it.** Simple RF/logistic models on 27 molecular features were more robust. Gene dependency (fitness) was the top feature.
- **De Kegel B, Quinn N, Thompson NA, Adams DJ, Ryan CJ.** *Comprehensive prediction of robust synthetic lethality between paralog pairs in cancer cell lines.* Cell Syst 12:1144–1159 (2021). DOI 10.1016/j.cels.2021.08.006; PMID 34529928. [A] Features predictive of paralog SL include **shared PPIs and evolutionary conservation**. A random-forest classifier on these features predicted results of combinatorial screens.

### 4.6 Synthesis: which features predict negative GIs
Ranked by how consistently they appear across the sources above:
1. **Gene-level propensity, i.e. GI degree.** Its drivers are single-mutant fitness (strongest in Koch 2012 and Costanzo 2016), complex membership, ortho-essentiality (×2.5), PPI degree, disorder, low expression variability, and conservation versus orphan status. **Most of a pair's probability of being negative is explained by the two genes' individual propensities** [inference from Koch 2012 and Costanzo 2016, to be tested with a degree-only baseline].
2. **Functional relatedness of the pair.** This means same complex or pathway, same process, and profile similarity. In budding yeast, negative GIs concentrate within and between functionally related modules. Without GO we need non-GO proxies: physical PPI (co-complex), co-localization, co-expression, phenotype-profile correlation, **ortholog profile similarity from Costanzo 2016**, and protein-embedding similarity.
3. **Physical interaction.** Co-complexed pairs are enriched for GIs, but in nonessential complexes those GIs are more often **positive** than negative (Costanzo 2016: 21% vs 5% of complexes). So PPI is a "related" signal, not a "negative" signal. Short PPI path, shared neighbours and cohesion are informative (SLant, SINaTRA, Paladugu).
4. **Paralogy.** This is a classic SL mechanism in *S. cerevisiae* and in human cancer cells. Copy number correlates negatively with degree (Koch). S. pombe has fewer paralogs, so this affects a small but high-precision subset [inference].
5. **Ortholog-pair transfer from *S. cerevisiae*.** It is strong within complexes but weak overall (r = 0.03 for different-process pairs; homology-transfer AUC 0.52–0.60 in SINaTRA).

### 4.7 Additional prior work (added after review rounds 1 and 2)
Bibliographic records below were checked against Europe PMC on 2026-10-28 (**[A]**); where full text was read, it is marked **[V]**. None of these papers' training labels may be used here if they derive from fission-yeast GI data (PROMPT.md rule); the entries are for method precedent only.
- **EMF / XSMF (Fan et al. 2020).** Fan J, Li XC, Crovella M, Leiserson MDM. *Matrix (factorization) reloaded: flexible methods for imputing genetic interactions with cross-species and side information.* Bioinformatics 36(Suppl_2):i866–i874 (2020). DOI 10.1093/bioinformatics/btaa818; PMID 33381837. [A]
  - **Extensible Matrix Factorization (EMF)** is a framework of composable models that use GI data from several species plus arbitrary side information as kernels (e.g. from PPI networks), tested on matched baker's- and fission-yeast GI data. It improves imputation, especially in data-scarce settings, and beats a state-of-the-art deep-learning imputer at far lower cost. [A, abstract]
  - The code is at github.com/lrgr/EMF; its configs include `costanzo-ryan` and XSMF (cross-species MF) variants, evaluated by 10-fold Monte-Carlo cross-validation, i.e. **pair-random splits**. [V, repository file listing]
  - Reported by the prior-work reviewer, **not verified**: it imputes Ryan 2012 using Costanzo-based data, and R² rises from 0.049 to 0.067 at 10% training data.
  - **Relevance:** the closest precedent for C1 and few-shot imputation on this exact matrix. It gives no evidence for unseen genes (C2/C3).
- **MUNK (Fan et al. 2019).** Fan J, Cannistra A, Fried I, Lim T, Schaffner T, Crovella M, Hescott B, Leiserson MDM. *Functional protein representations from biological networks enable diverse cross-species inference.* Nucleic Acids Res 47(9):e51 (2019). DOI 10.1093/nar/gkz132; PMID 30847485; PMC6511848. [V]
  - A kernel method that embeds proteins of different species in one vector space using sequence and PPI network structure (PPI from BioGRID, STRING for human/mouse). Used to share synthetic-lethal knowledge across species. Random forests on MUNK features reach AUROC 0.876 (S. pombe) / 0.933 (S. cerevisiae) on balanced BioGRID SL sets, compared with SINaTRA 0.880 / 0.908. [V]
  - **Caveat:** balanced, pair-random evaluation, and SL labels from BioGRID (forbidden for us for S. pombe). Method precedent only.
- **ETNA (Li et al. 2023).** Li L, Dannenfelser R, Zhu Y, Hejduk N, Segarra S, Yao V. *Joint embedding of biological networks for cross-species functional alignment.* Bioinformatics 39(9):btad529 (2023). DOI 10.1093/bioinformatics/btad529; PMID 37632792; PMC10477935. [V]
  - Embeddings to Network Alignment: per-species network embeddings aligned by a cross-training step using sequence orthologs. Its embeddings transfer GIs across species. Table 3 of the paper gives **Sce→Spo SL AUROC 0.741 / AUPRC 0.762** (ETNA) against lower MUNK values. [V]
  - **Caveats:** the SL sets are balanced samples from BioGRID, and the hyperparameter gold standard is GO-based. Both would be forbidden for us. Treat as method precedent (optional ortholog-anchored PPI embedding feature), not a benchmark number. The 0.74 figure is on a balanced task and is not comparable to our AUPRC.
- **ELISL (Tepeli et al. 2024).** Tepeli YI, Seale C, Gonçalves JP. *ELISL: early-late integrated synthetic lethality prediction in cancer.* Bioinformatics 40(1):btad764 (2024). DOI 10.1093/bioinformatics/btad764; PMID 38113447. [A]
  - Forest ensembles on protein-sequence embeddings (early integration) and cell-line/tissue omics (late integration); more robust to selection bias than topology-based methods. Supports using sequence embeddings as a cold-start signal. Human/cancer setting.
- **Messner et al. 2023.** Messner CB, Demichev V, Muenzner J, et al. *The proteomic landscape of genome-wide genetic perturbations.* Cell 186(9):2018–2034.e21 (2023). DOI 10.1016/j.cell.2023.03.026; PMID 37080200. [A]
  - DIA proteomics of the S. cerevisiae knockout library; proteome-profile similarity annotates gene function. It is the budding-yeast counterpart of Öztürk 2022 (§6) and the source for cerevisiae-side KO-proteome features (M4 transfer).
- **Ahlmann-Eltze, Huber & Anders 2025.** *Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines.* Nat Methods 22(8):1657–1661 (2025). DOI 10.1038/s41592-025-02772-6; PMID 40759747. [A]
  - Five foundation models and two other deep models failed to beat deliberately simple baselines for single and double perturbation transcriptome prediction (a different data type from GIs). Supports keeping additive/linear baselines in the model ladder.

## 5. Evaluation pitfalls for pair-input prediction
- **Park Y, Marcotte EM.** *Flaws in evaluation schemes for pair-input computational predictions.* Nat Methods 9:1134–1136 (2012). DOI 10.1038/nmeth.2259; PMID 23223166. [V]
  - Test pairs fall into **C1** (both components in the training set), **C2** (one) and **C3** (neither). Seven PPI predictors performed significantly differently across these classes.
  - In random cross-validation, **C1 made up >99% of test pairs**. In the human population of pairs, C1 is only 19.2%, C2 49.2% and C3 31.6%.
  - They recommend **reporting performance separately per class**.
  - **For us:** under the target population (all non-essential S. pombe pairs), most pairs involve at least one gene with no Ryan profile. Roughly half of about 3,500 non-essential genes have no profile, so C2/C3 dominate.
- **Hub/degree bias.**
  - Yu J, Guo M, Needham CJ, Huang Y, Cai L, Westhead DR. *Simple sequence-based kernels do not predict protein-protein interactions.* Bioinformatics 26:2610–2614 (2010). DOI 10.1093/bioinformatics/btq483; PMID 20801913 [A]. Accuracies were inflated by hub proteins dominating the positive set. They proposed degree-preserving "balanced" negatives.
  - Hamp T, Rost B. Bioinformatics 31:1521–1525 (2015). DOI 10.1093/bioinformatics/btu857; PMID 25586513 [A, title only]. Further pair-input pitfalls.
  - Eid FE, et al. *Systematic auditing is essential to debiasing machine learning in biology.* Commun Biol 4:183 (2021). DOI 10.1038/s42003-021-01674-5; PMID 33568741 [A]. "ML models tend to learn primarily from data biases when there is insufficient signal". They propose auditing with **node-degree-only and shuffled-feature baselines**.
  - Seale 2022 (§4.5) makes the same point for SL.
  - **Required controls for us:**
    - (a) a **gene-propensity-only baseline**: score(A,B) = f(rate_A, rate_B), with rates estimated on training data only, and for unseen genes predicted from gene features;
    - (b) a feature-permutation control;
    - (c) a label-permutation control within query.
- **Imbalance and metrics.**
  - Saito T, Rehmsmeier M. *The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets.* PLoS One 10:e0118432 (2015). DOI 10.1371/journal.pone.0118432; PMID 25738806 [A].
  - AUPRC's random baseline equals prevalence (here **0.0188** for S2 < −2.3, ≈2.0% in viable×viable [P]; not 0.04, which is the raw-score prevalence), so report AUPRC together with prevalence or as a fold-enrichment over prevalence.
  - **Prevalence differs between C1/C2/C3 and between queries.** Compare models within the same test set, never across splits with different prevalence.
  - Balanced 1:1 evaluations (SINaTRA, SLant, most DL papers) are not comparable to ours. Feng 2024 shows that F1 at 1:1 can be at chance (0.667) without anyone noticing.
- **Gene-disjoint splits and E-MAP-specific details:**
  - Split by **gene**, not allele: one gene can be a query with several alleles and also an array strain.
  - Assign the **reciprocal measurements** (A×B and B×A) of a pair to the same fold, or average them first.
  - **Hold out query genes and array genes separately.** Within a held-out query, all its arrays are test pairs. C1/C2/C3 should be defined on the union of query and array roles.
  - **Profile-derived features** (profile similarity, KNN imputation, gene degree) must be recomputed **inside each training fold**. Computing them on the full matrix leaks test labels.
  - **Prevent hyperparameter leakage** with nested CV or a fixed validation fold.
  - Optionally, **stratify unseen genes** by how similar they are to training genes (sequence identity, same complex). Near-duplicates such as paralogs can turn "unseen" into effectively "seen".
- **Label noise and the ceiling.** Replicate r = 0.4 (Ryan 2012 Fig. S1A). Within-*S. cerevisiae* reproducibility of negatives across methods is 37% (all pairs) to 62% (within complex). Expect a hard ceiling on AUPRC well below 1. A replicate-concordance estimate of that ceiling, e.g. from reciprocal A×B vs B×A pairs, should be part of the design.
- **Missing-not-at-random pairs.** Linked gene pairs were removed and noisy strains pruned. Predictions for genomically adjacent pairs cannot be evaluated against Ryan, so flag them.

## 6. Feature sources and their circularity risk
Risk legend: **Low** = experimentally independent of GI data and of GO/text mining; **Medium** = curated or derived in a way that could embed GI or GO information, so it needs filtering; **High / forbidden** = excluded.

| Source | What it gives | Coverage / citation | Circularity risk and handling |
|---|---|---|---|
| **Costanzo 2016** (S. cerevisiae) | ortholog-pair ε and P-value, profile PCC (NxN/ExN/ExE), SMF, degree | §3.1 [V] | **Allowed by PROMPT.md** ("make full use of … Costanzo et al 2016"). Ortholog mapping needed. Use deletion and TS/DAmP strains carefully (allele-level rows). Low risk for S. pombe labels. |
| **FissionNet systematic Y2H** | binary PPI | Vo TV, et al. Cell 164:310–323 (2016). DOI 10.1016/j.cell.2015.11.037; PMID 26771498 [V]. 4,989 ORFs tested pairwise (>99% of genes); 2,278 high-quality interactions | **Low.** Systematic and unbiased, but sparse, so most genes have degree 0. Missing ≠ non-interacting. |
| **BioGRID physical** (S. pombe and S. cerevisiae) | affinity-capture MS, Y2H, co-crystal, reconstituted complex, etc. | Oughtred R, et al. Protein Sci 30:187–200 (2021). DOI 10.1002/pro.3978; PMID 33070389 [A] | **Medium.** BioGRID is manually curated, not text-mined, but it is literature-biased (study and degree bias). **Filter to `Experimental System Type = physical`. Never load genetic rows.** Low-throughput physical entries come from hypothesis-driven papers that may have been motivated by GI findings. Consider a "high-throughput physical only" variant as a sensitivity check. |
| **ORFeome localization** | YFP localization of 4,431 proteins | Matsuyama A, et al. Nat Biotechnol 24:841–847 (2006). DOI 10.1038/nbt1222; PMID 16823372 [A] | **Low**, if the raw localization calls are used. Do **not** use the GO cellular-component annotations derived from it. |
| **Deletion collection / essentiality** | viability, heterozygous growth | Kim DU, et al. Nat Biotechnol 28:617–623 (2010). DOI 10.1038/nbt.1628; PMID 20473289 [A]. 4,836 heterozygous deletions; 83% conserved dispensability | **Low.** Needed to define the "non-essential" target universe and ortho-essentiality. |
| **S. pombe single-mutant fitness** | colony-size fitness from SGA control screen | Koch 2012 Additional file 2 [V] | **Low** for the fitness column. **Drop** the GI-degree and GO-multifunctionality columns from the same file. |
| **Deletion phenomics** | fitness of 3,509 deletion mutants in 129 usable conditions (the paper's condition key lists 131), giving phenotype-correlation profiles | Rodríguez-López M, et al. eLife 12:RP88229 (2023). DOI 10.7554/eLife.88229; PMID 37787768 [A] | **Low** for the raw growth phenotypes. A **strong pair feature** (phenotype-profile correlation) that **covers unseen genes**. **Exclude NET-FF GO predictions** (GO-derived; its "protein-network" inputs may include STRING [unverified]). |
| **Deletion-mutant proteomes** | proteome changes in 3,308 knockouts | Öztürk M, et al. Nat Commun 13:6153 (2022). DOI 10.1038/s41467-022-33814-8; PMID 36257942 [A] | **Low.** It gives a second genome-wide **functional-profile similarity** for non-essential genes, analogous to GI-profile similarity but independent of GI data. |
| **Absolute transcript and protein abundance** | copies per cell, proliferating and quiescent | Marguerat S, et al. Cell 151:671–683 (2012). DOI 10.1016/j.cell.2012.09.019; PMID 23101633 [A]. Carpy A, et al. Mol Cell Proteomics 13:1925–1936 (2014). DOI 10.1074/mcp.M113.035824; PMID 24763107 [A] | **Low.** Gene-level features: abundance correlates with hub status in S. cerevisiae (Costanzo 2016). |
| **Expression compendia / co-expression** | co-expression and expression variability | Pancaldi V, Schubert F, Bähler J. Mol Biosyst 6:543–552 (2010). DOI 10.1039/b913876p; PMID 20174682 [A]; GEO/ArrayExpress | **Low to medium.** Avoid compendia that include double-mutant expression or that were assembled from GI-selected strains. Expression variability is a known degree predictor (Koch). Feng 2024 warns that SL pairs skew to positive co-expression, so **never use co-expression to choose negatives**. |
| **Morphology / cell-cycle screens** | deletion shape phenotypes | Hayles J, et al. Open Biol 3:130053 (2013). DOI 10.1098/rsob.130053; PMID 23697806 [A] | **Low** for the raw screen calls. **Medium** if taken from PomBase FYPO annotations, because FYPO also holds multi-allele (genetic-interaction) phenotypes. Use **single-allele** annotations only, or the raw screen. |
| **Sequence-derived features** | Pfam/InterPro domains, disorder, length, codon usage, dN/dS, conservation, phylogenetic profiles, paralogs | Pfam/InterPro, InParanoid-style orthology, PomBase | **Low**, with one caveat: **do not use InterPro2GO mappings**. Phylogenetic profiles and orphan status are strong gene-level priors (Ryan 2012 Fig. 5D). |
| **Orthology S. pombe ↔ S. cerevisiae** | ortholog pairs, 1:1 vs 1:many | PomBase manually curated ortholog list (Ryan 2012 used Wood 2006 [V]); Harris MA, et al. Genetics 220:iyab222 (2022). DOI 10.1093/genetics/iyab222; PMID 35100366 [A] | **Low to medium.** The PomBase curation uses sequence plus literature, and could rarely use functional or genetic evidence to resolve paralogs. Sensitivity check: compare with a purely algorithmic orthology method (InParanoid/OrthoFinder). |
| **Protein language-model embeddings** | per-protein vectors; pair cosine similarity | ESM-2: Lin Z, et al. Science 379:1123–1130 (2023). DOI 10.1126/science.ade2574; PMID 36927031 [A]. ProtT5: Elnaggar A, et al. IEEE TPAMI 44:7112–7127 (2022). DOI 10.1109/TPAMI.2021.3095381; PMID 34232869 [A] | **Low.** Sequence only, and covers every gene, which is good for C3. Embedding similarity largely encodes homology and family membership. Watch for paralog leakage across gene-disjoint splits (§5). Check for local-compute feasibility before committing (ESM-2 650M on about 5,100 proteins is likely feasible on CPU/MPS [R]). |
| **AlphaFold-derived complexes** | predicted pairwise interfaces (pDockQ/ipTM) | Humphreys IR, et al. Science 374:eabm4805 (2021). DOI 10.1126/science.abm4805; PMID 34762488 [A] (S. cerevisiae, about 8.3M pairs screened [R from search summary]); SPOC/Predictomes (Mol Cell 2025; PMID 40015271 [A]) | **Low** for circularity. **Coverage problem:** I found no published proteome-wide S. pombe AF-Multimer screen, and the search did not surface one. Options are to map *S. cerevisiae* predictions through orthologs, or to compute a targeted subset locally (likely too expensive genome-wide). AFDB monomer structures exist for the S. pombe proteome [R]. Also note that AF-Multimer screens often use co-evolution/MSA pairing and confidence classifiers trained on PDB and curated PPIs; this is not GI data, so the risk is low. |
| **Complex Portal** | curated complexes (S. pombe and S. cerevisiae) | Meldal BHM, et al. NAR 50:D578–D586 (2022). DOI 10.1093/nar/gkab991; PMID 34718729 [A] | **Medium.** Manually curated from the literature (not text-mined). It is closely tied to GO cellular-component curation, and evidence could include genetic data. A PI decision is needed. A safer alternative is cerevisiae CYC2008 / Costanzo Data File S12 mapped through orthologs. |
| **STRING, GO, text-mined co-mention** | — | — | **Forbidden by PROMPT.md** (no text mining or derived resources such as STRING; no GO). |
| **PomBase GO/complex annotations, GeneMANIA, HumanNet** | — | — | **Excluded by project interpretation** of the PROMPT.md GO / text-mining rules (GO-derived or integrate text-mined/GO evidence). |
| **PomBase genetic-interaction file, BioGRID genetic rows, FYPO multi-allele phenotypes, Roguev/Dixon/Frost/Koch S. pombe GI screens** | — | — | **Forbidden by PROMPT.md** (no fission-yeast GI data from any source other than Ryan 2012); applying it to these specific curated files is a project interpretation. |
| **Ryan 2012 Dataset S3 (published profile similarity)** | precomputed similarity | — | It is derived from *all* Ryan data, so it **leaks test labels**. Recompute similarity inside each training fold instead. |

## 7. Implications for our design

### 7.1 Strongest ideas
1. **Decompose the score into gene propensity × pair relatedness.** Most predictive signal in the literature is gene-level: SMF, complex membership, ortho-essentiality, orphan status, conservation, PPI degree and expression variability. Learn a gene-propensity model from Ryan training genes using gene features, so it extends to unseen genes as in Koch 2012 (cross-species r = 0.51 for degree). Then add pair-level relatedness on top. This also gives the mandatory degree-only baseline for free.
2. **Use Costanzo 2016 in three ways, not one:**
   - (a) ortholog-pair ε (helps within complexes and processes);
   - (b) **ortholog profile similarity** (PCC from Data File S3), which captures conserved module membership; Ryan 2012 shows module-level conservation is much stronger than edge-level;
   - (c) cerevisiae gene degree and SMF as gene priors, with ExN/TS strains covering ortho-essential genes.
   Handle 1:many orthologs by max or mean aggregation, and add "has ortholog" indicator features.
3. **Rank-normalize gene-level network features within each species** (SINaTRA: AUC 0.67→0.86). Also use within-species rank or percentile for SMF and abundance.
4. **Add functional-profile similarities that exist for unseen genes:** phenomics correlation (Rodríguez-López 2023, 129 usable conditions), knockout-proteome similarity (Öztürk 2022), co-expression, co-localization (Matsuyama 2006), and PLM-embedding similarity. Together these are the only non-GO substitutes for "same process" information, which is the strongest pair-level signal in yeast.
5. **For C1/C2 pairs, use Ryan-internal information leakage-safely:** fold-internal profile similarity, neighbour-based imputation (Ulitsky 2009 / Ryan 2011 style), and query/array propensity estimates. Train with a model that can fall back to features when these are missing (C3), and report C1/C2/C3 separately.
6. **Use simple, robust learners first.** Gradient-boosted trees or logistic models on engineered features match or beat graph DL on unseen genes (Feng 2024; Seale 2022). GNN/matrix-factorization methods should be tested only as add-ons, with gene-disjoint evaluation.

### 7.2 Biggest risks
- **Leakage through curated resources** that ingest S. pombe GI data or GO: BioGRID genetic rows, FYPO multi-allele phenotypes, NET-FF/GO predictions, Complex Portal, Koch 2012 degree columns, and the published Ryan Dataset S3. Mitigate with an explicit allow-list in DATA.md and a test that greps every feature table for forbidden sources.
- **Profile features computed on the full matrix**, which leaks test labels. All Ryan-derived features must be recomputed per fold.
- **Allele and reciprocal-pair leakage:** the same gene appears as several query alleles and as an array; A×B and B×A are both measured. Split by gene.
- **Hypomorph (DAmP) queries.** The target is non-essential × non-essential pairs. Decide explicitly whether to (i) train on all alleles with an allele-class feature, or (ii) restrict training labels to deletion alleles. Either way, **evaluate on deletion × deletion pairs of non-essential genes**.
- **Degree and propensity shortcuts.** A model can reach a good overall AUPRC by ranking whole queries. Always report against the propensity-only baseline, and also report per-query (macro) AUPRC.
- **Query-selection bias.** Queries are skewed to conserved nuclear/chromatin genes, so a model trained on them may not represent the array-only and unscreened genome. Unseen-gene performance may be overestimated if held-out genes are drawn from the same biased pool. Consider stratifying held-out genes by process or conservation.
- **Noise ceiling.** With replicate r = 0.4, strong apparent gains on a single split may be noise. Use multiple seeds and gene-disjoint folds, with bootstrap CIs over genes.

### 7.3 Realistic expected AUPRC
No published study evaluates our exact setting (S. pombe, S < −2.3, gene-disjoint, natural prevalence). **All ranges below are my extrapolations**, anchored on the cited evidence and expressed as lift over prevalence. They are not literature numbers. They were rescaled on 2026-10-28 from the earlier 0.04 anchor to the correct prevalence by keeping the lift multiples fixed; AUPRC does not scale exactly linearly with prevalence, so treat them as order-of-magnitude orientation only.
- **Prevalence/random baseline: 0.0188** (S2 < −2.3, all pairs with S2) and **≈0.020 in viable×viable pairs** [P]. (The earlier 0.04 is the raw-score prevalence.)
- **C1 (both genes seen):** gene propensity alone should give a multiple of prevalence, because degree is heavy-tailed (Costanzo hubs have about 3.5× average degree). Profile similarity and imputation add further signal. Plausible AUPRC is **0.06–0.15** (about 3–7.5× lift; previously 0.12–0.30 at the wrong anchor). The ceiling is set by label noise (replicate r = 0.4; within-species reproducibility of negatives 37–62%), so AUPRC above about **0.25** (≈12× lift) would be suspicious and should trigger a leakage audit. (Threshold also an extrapolation.)
- **C2 (one gene unseen):** the seen partner's propensity carries much of the signal, and the unseen gene's propensity must come from features (Koch: r ≈ 0.5 for degree). Plausible AUPRC is **0.04–0.09** (about 2–4.5× lift; previously 0.08–0.18).
- **C3 (neither seen):** Feng 2024 found near-chance performance for SL-graph methods. Feature-based models (SINaTRA, SLant, Koch) retain signal, but it was measured on balanced or ROC metrics. Plausible AUPRC is **0.025–0.05**, i.e. 1.2–2.5× prevalence (previously 0.05–0.10). Ortholog ε gives **high precision on the small subset of intra-complex ortholog pairs** (68% conservation) and little elsewhere.
- **Pairs covered by Costanzo orthologs:** expect AUPRC gains concentrated in conserved, high-relatedness pairs. Report a "has 1:1 ortholog pair in Costanzo" stratum.

## 8. What I could not verify
- Exact number of technical replicates per pair in Ryan 2012, the composition of Datasets S1/S2 (raw vs averaged; whether Roguev 2008 data are merged in), and the exact count of query genes vs array genes in the published matrix. The data agent's counts (876 alleles / 873 query genes / about 1,950 arrays) differ from the paper's "953 alleles of 876 genes" and need reconciling.
- The exact allele classes in Ryan 2012 beyond "deletion" and "hypomorphic alleles of essential genes" (e.g. DAmP). The [P] labels (point mutant, ts, truncation) are inferred from strain names by the data agent.
- Wong 2004 feature ranking (I could not access the full text). The "2-hop features most predictive" claim is [R], partially supported by Pandey 2010's description.
- A paper or tool named "Slorth": not in PROMPT.md and not found; only SLant (Benstead-Hume 2019) was verified.
- Fan 2020 (EMF) details beyond the abstract (use of Ryan 2012 with Costanzo 2010, R² 0.049 → 0.067 at 10% training, pair-random splits only): the abstract and the lrgr/EMF repository (Costanzo/Ryan configs, 10-fold Monte-Carlo CV) were checked, but the full text was not accessible. The numbers come from the prior-work review and are **unverified**.
- The ε = f_ij − f_i f_j formula (standard, but [R]). Whether Costanzo intermediate thresholds use strict or non-strict inequalities (sources differ).
- Any proteome-wide AlphaFold-Multimer screen for S. pombe (none found).
- Whether the PomBase-hosted Rodríguez-López NET-FF predictions draw on STRING.

## 9. Reference list (verified identifiers)
- Ahlmann-Eltze C, Huber W, Anders S. Nat Methods 22:1657 (2025). DOI 10.1038/s41592-025-02772-6. PMID 40759747.
- Baryshnikova A, et al. Nat Methods 7:1017 (2010). DOI 10.1038/nmeth.1534. PMID 21076421.
- Benstead-Hume G, et al. PLoS Comput Biol 15:e1006888 (2019). DOI 10.1371/journal.pcbi.1006888. PMID 30995217.
- Cai R, et al. Bioinformatics 36:4458 (2020). DOI 10.1093/bioinformatics/btaa211. PMID 32221609.
- Carpy A, et al. Mol Cell Proteomics 13:1925 (2014). DOI 10.1074/mcp.M113.035824. PMID 24763107.
- Chen X, et al. Brief Bioinform 26:bbaf142 (2025). DOI 10.1093/bib/bbaf142. PMID 40194555.
- Chipman KC, Singh AK. BMC Bioinformatics 10:17 (2009). DOI 10.1186/1471-2105-10-17. PMID 19138426.
- Collins SR, et al. Genome Biol 7:R63 (2006). DOI 10.1186/gb-2006-7-7-r63. PMID 16859555.
- Conde-Pueyo N, et al. BMC Syst Biol 3:116 (2009). DOI 10.1186/1752-0509-3-116. PMID 20015360.
- Costanzo M, et al. Science 353:aaf1420 (2016). DOI 10.1126/science.aaf1420. PMID 27708008.
- De Kegel B, et al. Cell Syst 12:1144 (2021). DOI 10.1016/j.cels.2021.08.006. PMID 34529928.
- Dixon SJ, et al. PNAS 105:16653 (2008). DOI 10.1073/pnas.0806261105. PMID 18931302.
- Eid FE, et al. Commun Biol 4:183 (2021). DOI 10.1038/s42003-021-01674-5. PMID 33568741.
- Elnaggar A, et al. IEEE TPAMI 44:7112 (2022). DOI 10.1109/TPAMI.2021.3095381. PMID 34232869.
- Fan J, Li XC, Crovella M, Leiserson MDM. Bioinformatics 36(Suppl_2):i866 (2020). DOI 10.1093/bioinformatics/btaa818. PMID 33381837. (EMF/XSMF)
- Fan J, et al. Nucleic Acids Res 47:e51 (2019). DOI 10.1093/nar/gkz132. PMID 30847485. (MUNK)
- Feng Y, et al. Nat Commun 15:9058 (2024). DOI 10.1038/s41467-024-52900-7. PMID 39428397. Correction: PMID 39948334.
- Frost A, et al. Cell 149:1339 (2012). DOI 10.1016/j.cell.2012.04.028. PMID 22682253.
- Hamp T, Rost B. Bioinformatics 31:1521 (2015). DOI 10.1093/bioinformatics/btu857. PMID 25586513.
- Hao Z, et al. IEEE JBHI 25:4041 (2021). DOI 10.1109/JBHI.2021.3079302. PMID 33974548.
- Harris MA, et al. Genetics 220:iyab222 (2022). DOI 10.1093/genetics/iyab222. PMID 35100366.
- Hayles J, et al. Open Biol 3:130053 (2013). DOI 10.1098/rsob.130053. PMID 23697806.
- Humphreys IR, et al. Science 374:eabm4805 (2021). DOI 10.1126/science.abm4805. PMID 34762488.
- Jacunski A, Dixon SJ, Tatonetti NP. PLoS Comput Biol 11:e1004506 (2015). DOI 10.1371/journal.pcbi.1004506. PMID 26451775.
- Kim DU, et al. Nat Biotechnol 28:617 (2010). DOI 10.1038/nbt.1628. PMID 20473289.
- Koch EN, et al. Genome Biol 13:R57 (2012). DOI 10.1186/gb-2012-13-7-r57. PMID 22747640.
- Kuzmin E, et al. Science 360:eaao1729 (2018). DOI 10.1126/science.aao1729. PMID 29674565.
- Kuzmin E, et al. Science 368:eaaz5667 (2020). DOI 10.1126/science.aaz5667. PMID 32586993.
- Lin Z, et al. Science 379:1123 (2023). DOI 10.1126/science.ade2574. PMID 36927031.
- Liu Y, et al. IEEE/ACM TCBB 17:748 (2020). DOI 10.1109/TCBB.2019.2909908. PMID 30969932.
- Li L, et al. Bioinformatics 39:btad529 (2023). DOI 10.1093/bioinformatics/btad529. PMID 37632792. (ETNA)
- Long Y, et al. Bioinformatics 37:2432 (2021). DOI 10.1093/bioinformatics/btab110. PMID 33609108.
- Long Y, et al. Bioinformatics 38:2254 (2022). DOI 10.1093/bioinformatics/btac100. PMID 35171981.
- Ma J, et al. Nat Methods 15:290 (2018). DOI 10.1038/nmeth.4627. PMID 29505029.
- Marguerat S, et al. Cell 151:671 (2012). DOI 10.1016/j.cell.2012.09.019. PMID 23101633.
- Matsuyama A, et al. Nat Biotechnol 24:841 (2006). DOI 10.1038/nbt1222. PMID 16823372.
- Messner CB, et al. Cell 186:2018 (2023). DOI 10.1016/j.cell.2023.03.026. PMID 37080200.
- Meldal BHM, et al. Nucleic Acids Res 50:D578 (2022). DOI 10.1093/nar/gkab991. PMID 34718729.
- Michaelis AC, et al. Nature 624:192 (2023). DOI 10.1038/s41586-023-06739-5. PMID 37968396. (S. cerevisiae AP-MS interactome; optional cerevisiae PPI source.)
- Oughtred R, et al. Protein Sci 30:187 (2021). DOI 10.1002/pro.3978. PMID 33070389.
- Öztürk M, et al. Nat Commun 13:6153 (2022). DOI 10.1038/s41467-022-33814-8. PMID 36257942.
- Paladugu SR, et al. BMC Bioinformatics 9:426 (2008). DOI 10.1186/1471-2105-9-426. PMID 18844977.
- Pancaldi V, Schubert F, Bähler J. Mol Biosyst 6:543 (2010). DOI 10.1039/b913876p. PMID 20174682.
- Pandey G, et al. PLoS Comput Biol 6:e1000928 (2010). DOI 10.1371/journal.pcbi.1000928. PMID 20838583.
- Park Y, Marcotte EM. Nat Methods 9:1134 (2012). DOI 10.1038/nmeth.2259. PMID 23223166.
- Qi Y, et al. Genome Res 18:1991 (2008). DOI 10.1101/gr.077693.108. PMID 18832443.
- Rodríguez-López M, et al. eLife 12:RP88229 (2023). DOI 10.7554/eLife.88229. PMID 37787768.
- Roguev A, et al. Science 322:405 (2008). DOI 10.1126/science.1162609. PMID 18818364.
- Ryan C, et al. Methods Mol Biol 781:353 (2011). DOI 10.1007/978-1-61779-276-2_17. PMID 21877290.
- Ryan CJ, et al. Mol Cell 46:691 (2012). DOI 10.1016/j.molcel.2012.05.028. PMID 22681890.
- Saito T, Rehmsmeier M. PLoS One 10:e0118432 (2015). DOI 10.1371/journal.pone.0118432. PMID 25738806.
- Seale C, Tepeli Y, Gonçalves JP. Bioinformatics 38:4360 (2022). DOI 10.1093/bioinformatics/btac523. PMID 35876858.
- Srivas R, et al. Mol Cell 63:514 (2016). DOI 10.1016/j.molcel.2016.06.022. PMID 27453043.
- Ulitsky I, Krogan NJ, Shamir R. Genome Biol 10:R140 (2009). DOI 10.1186/gb-2009-10-12-r140. PMID 20003301.
- Tepeli YI, Seale C, Gonçalves JP. Bioinformatics 40:btad764 (2024). DOI 10.1093/bioinformatics/btad764. PMID 38113447. (ELISL)
- Vo TV, et al. Cell 164:310 (2016). DOI 10.1016/j.cell.2015.11.037. PMID 26771498.
- Wang Q, et al. Brief Bioinform 27:bbag470 (2026). DOI 10.1093/bib/bbag470. PMID 42704265.
- Wang S, et al. (KG4SL) Bioinformatics 37:i418 (2021). DOI 10.1093/bioinformatics/btab271. PMID 34252965.
- Wang S, et al. (NSF4SL) Bioinformatics 38:ii13 (2022). DOI 10.1093/bioinformatics/btac462. PMID 36124790.
- Wong SL, et al. PNAS 101:15682 (2004). DOI 10.1073/pnas.0406614101. PMID 15496468.
- Yu J, et al. Bioinformatics 26:2610 (2010). DOI 10.1093/bioinformatics/btq483. PMID 20801913.

## 10. Changelog
- **2026-10-28 (corrections from reviews/round1_prior_work.md item 11 and reviews/round2_algorithms_prior.md item 10):**
  - Prevalence: 0.04 replaced by **0.0188** (S2 < −2.3; ≈2.0% in viable×viable) in §1.2, §5 and §7.3; 0.04 is stated to be the raw-score prevalence. Expected-AUPRC ranges in §7.3 rescaled (C1 0.06–0.15, C2 0.04–0.09, C3 0.025–0.05) and flagged as extrapolations; leakage-audit threshold lowered from 0.5 to about 0.25.
  - Phenomics: 129 usable conditions (the paper's key lists 131), §6 and §7.1.
  - Attributions: removed "the brief" claims not in PROMPT.md (Koch 2012 as Genes Dev; "Slorth"; Feng et al. "2023"); "forbidden by the brief" now cites PROMPT.md only for Costanzo (allowed), STRING/text mining, GO, and non-Ryan fission-yeast GI; other exclusions labelled as project interpretation. Koch 2012 is Genome Biology 13:R57 (kept).
  - Added §4.7: EMF/XSMF (Fan 2020), MUNK (Fan 2019), ETNA (Li 2023), ELISL (Tepeli 2024), Messner 2023, Ahlmann-Eltze 2025, each checked against Europe PMC. Fan 2020 numbers quoted by the reviewer remain unverified (full text not accessible). Added the six references to §9 and the unverified items to §8.
  - Checked and left unchanged: Öztürk 2022 is Nat Commun 13:6153 (the page number 6232 in `docs/data_aux_inventory.md` is wrong).
