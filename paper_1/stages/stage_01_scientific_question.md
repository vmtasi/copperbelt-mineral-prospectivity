# Stage 01: Introduction, Background and Scientific Context

## 1.1 Geological context and prospectivity mapping

Mineral prospectivity mapping in complex metallogenic belts requires integrating multiple geological and geophysical indicators to estimate the spatial probability of undiscovered mineral deposits. The Central African Copperbelt is one of the world's premier sediment-hosted stratiform copper-cobalt provinces, extending over 400 kilometres across diverse structural and tectonic domains in Zambia and the Democratic Republic of the Congo. Mineralization in this belt is not associated with a single structural or lithological feature; rather, ore formation is widely recognized as the product of an interconnected earth system involving basin architecture, fluid pathways, host-rock reactivity, and regional crustal configuration.

To reflect this multivariate system, this study evaluates three principal continuous geological and geophysical predictors alongside host-rock stratigraphy:
1. **Distance to major faults ($D_{fault}$):** Structural discontinuities that served as primary conduits for hydrothermal basin fluids.
2. **Distance to lithological contacts ($D_{lith}$):** Stratigraphic boundaries where chemical reactivity, redox contrasts, and fluid-mixing processes favored copper-cobalt precipitation.
3. **Bouguer gravity anomaly ($X_{grav}$):** Regional gravity structure providing complementary information on basement architecture, sub-basin depocentres, and crustal density variations.
4. **Host lithological classes:** Categorical rock-type indicators representing the immediate stratigraphic host environment.

Because geological histories, deformation intensity, and stratigraphic host settings vary markedly along the Copperbelt, the relationships between these predictors and mineralization are unlikely to remain uniform from one sector of the belt to another.

## 1.2 Uncertainty and spatially separated predictive transfer

A fundamental challenge in mineral prospectivity modeling is spatial autocorrelation. Mineral deposits and geological features cluster in space, meaning that conventional random cross-validation—where randomly selected grid cells are held out—inevitably allows training and testing observations from the same local cluster to mingle. This produces severe data leakage and artificially inflated performance estimates that fail to indicate how well a model can predict prospectivity in genuinely unexplored frontier regions.

Evaluating prospectivity models therefore demands **spatially separated predictive validation**, in which entire contiguous geographical sectors are held out out-of-fold (OOF). A model that achieves high predictive accuracy when interpolating between known deposits may collapse entirely when required to extrapolate across distinct geological sectors. Spatially separated validation provides an honest, rigorous test of whether learned multivariate relationships transfer along the orogenic strike.

Furthermore, geological data are inherently observational, incomplete, and subject to spatial sampling bias. An exploration model must therefore quantify predictive and parametric uncertainty in a principled manner, avoiding overconfident assertions regarding unobservable frontier ground.

## 1.3 Spatial heterogeneity and hierarchical Bayesian inference

The tension between regional geological differences and belt-wide commonalities presents a classic statistical trade-off. Fitting a single global model across the entire Copperbelt assumes that structural and lithological relationships are spatially stationary, potentially obscuring vital regional nuances. Conversely, fitting completely separate models within individual sub-basins discards shared geological knowledge and fails entirely in sectors where known mineral deposits are sparse or absent.

This study resolves that tension by formulating a **hierarchical Bayesian prospectivity model with partial pooling across Daly's tectonic domains**. Daly's six recognized geological domains provide a natural geological stratification of the belt. Under this hierarchical architecture:
- Base-rate log-odds intercepts vary by domain, accommodating regional baseline differences while sharing a belt-wide population distribution;
- Proximity relationships for faults and lithological contacts are allowed to vary across domains, reflecting local structural style while shrinking toward population-level means;
- Regional Bouguer gravity and valid lithological units provide global stabilizing constraints.

The hierarchy serves as an uncertainty-aware statistical framework for evaluating spatially varying relationships; it does not, by itself, assume or prove geological causation.

## 1.4 Distance response: representation and functional form

Proximity to faults and contacts is a cornerstone of mineral exploration vectoring. However, how distance relationships should be mathematically represented remains an open question. Proximity is generally expected to correlate with increased prospectivity, but the exact response shape may not be strictly monotonic or uniform:
- At very large distances, structural or lithological vectoring influence diminishes toward background levels.
- At proximal distances, the response may be linear on the logit scale, follow a power-law or logarithmic decay, or exhibit localized curvature resulting from fault-damage zones, alteration haloes, or optimal fluid-mixing windows.

To investigate whether distance responses exhibit curvature, earlier iterations of this project introduced quadratic distance terms on the standardized scale:

$$
\eta(z) = \alpha + \beta_{1} z + \beta_{2} z^2.
$$

Where positive quadratic curvature ($\beta_{2} > 0$) is supported, an algebraic stationary point can be derived:

$$
z^* = -\frac{\beta_{1}}{2\beta_{2}}, \qquad D^* = \mu_{train} + \sigma_{train} z^*.
$$

Historically, $D^*$ was treated as a candidate "optimal" or "characteristic" distance. However, rigorous scientific inference requires testing whether this quadratic curvature and its derived stationary point are representation-dependent and sensitive to mathematical representation choices (e.g., raw versus logarithmic distances, or polynomial versus linear functional forms). In this study, $D^*$ is treated strictly as a **model-derived diagnostic** whose interpretation is not robust to the distance representation and cannot be interpreted as an invariant geological quantity. Bouguer gravity enters as a linear predictor and consequently has no corresponding turning-point quantity; its lack of a $D^*$ reflects model structure, not an assertion of lesser geological importance.

## 1.5 Central scientific objective and questions

The central objective of this study is:

> **To establish an uncertainty-aware Bayesian mineral prospectivity mapping framework under spatially separated along-belt prediction, and to investigate whether geological and geophysical predictor relationships exhibit spatial heterogeneity across the distinct tectonic regimes of the Central African Copperbelt.**

Specifically, the study addresses five primary scientific questions:
1. **Multivariate Predictor Association:** How do fault distance, lithology-contact distance, Bouguer gravity, and host stratigraphy jointly relate to copper-cobalt mineralization across the Copperbelt when modeled within a Bayesian framework?
2. **Spatial Transferability:** How well do the learned multivariate prospectivity relationships transfer to held-out geographical sectors under honest four-fold along-belt spatial cross-validation?
3. **Spatial Heterogeneity:** Do the relative contributions of structural, lithological, and geophysical predictors vary systematically across the along-belt folds and tectonic domains?
4. **Distance Representation Robustness:** Are proximity associations and apparent quadratic curvatures robust to alternative mathematical representations (raw versus logarithmic scaling, linear versus quadratic functional forms)?
5. **Diagnostic Status of $D^*$:** Does the derived stationary point $D^*$ constitute an identifiable, representation-robust geological quantity, or is it an unstable mathematical property of specific polynomial parameterizations?

## 1.6 Methodological boundaries

This study does not presume that Daly's tectonic hypothesis is fully proven. It evaluates whether empirical evidence from a multivariate hierarchical model is consistent with regional geological heterogeneity along the Copperbelt. The study does not claim that a single universal distance controls mineralization across the entire belt, nor does it present $D^*$ as a validated physical drilling target. All empirical findings are grounded in the defined modeling grid (1,872 cells, 138 deposit-positive observations), the implemented Bayesian hierarchical specification, and the four-fold along-belt validation framework.
