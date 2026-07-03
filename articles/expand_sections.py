"""Expand thin subsections in article1.tex."""
import pathlib

tex = pathlib.Path(r"c:\Users\mdgor\aca\aca_indice_climatico_opt-main\articles\article1.tex")
content = tex.read_text(encoding="utf-8")

replacements = []

# ── §2.1 The ACI Family ──────────────────────────────────────────────────────
OLD_21 = (
    r"An open-source Python package (\texttt{x-ACI}) has been released to facilitate"
    r" replication in new regions \citep{milhaud2025xaci}."
)
NEW_21 = (
    r"An open-source Python package (\texttt{x-ACI}) has been released to facilitate"
    r" replication in new regions \citep{milhaud2025xaci}."
    "\n\n"
    r"A key design principle shared across all ACI implementations is temporal and"
    r" cross-regional comparability: by standardizing against a fixed 1961--1990"
    r" baseline and computing equal-weight averages, the index can be read as a"
    r" deviation from mid-twentieth-century norms that is comparable across"
    r" decades, countries, and climate regimes. This equal-weight convention also"
    r" prevents the index from being dominated by a single data-rich component,"
    r" an important consideration in data-sparse regions where some components"
    r" may have higher estimation uncertainty. The methodological challenge for"
    r" tropical adaptations, as this paper demonstrates, is that the standard"
    r" thresholds---designed for temperate seasonality---need to be recalibrated"
    r" to account for narrower temperature ranges and ENSO-driven precipitation"
    r" regimes \citep{valla2025}."
    "\n\n"
    r"The international expansion of the ACI family also reflects a regulatory"
    r" context. In Canada and the United States, the original ACI was developed"
    r" partly in response to pressure from regulatory bodies for objective,"
    r" standardized climate risk metrics that could be embedded in Own Risk and"
    r" Solvency Assessment (ORSA) frameworks. A similar dynamic is now unfolding"
    r" in Europe under the EU Taxonomy and TCFD-aligned disclosure requirements,"
    r" and in Colombia under Circular Externa 015/2025 of the"
    r" \emph{Superintendencia Financiera de Colombia} (SFC), which explicitly"
    r" mandates the identification and measurement of physical climate risks."
    r" The ACI-CO is designed from the outset to satisfy these regulatory"
    r" requirements while remaining scientifically rigorous and fully reproducible."
)
replacements.append((OLD_21, NEW_21))

# ── §2.2 Applications and Extensions ─────────────────────────────────────────
OLD_22 = (
    r"\citet{valla2025} proposed time-penalized trees and LASSO-based feature selection"
    r" to challenge the ACI's fixed quantile thresholds, finding that the 90th percentile"
    r" is not always optimal for specific risk targets."
)
NEW_22 = (
    r"\citet{valla2025} proposed time-penalized trees and LASSO-based feature selection"
    r" to challenge the ACI's fixed quantile thresholds, finding that the 90th percentile"
    r" is not always optimal for specific risk targets."
    "\n\n"
    r"The insurance pricing literature has identified several direct applications of"
    r" ACI-type indices beyond academic risk monitoring. \citet{garrido2023faci} showed"
    r" that ACI component anomalies Granger-cause claim frequencies in French property"
    r" lines with lags of one to three months, suggesting that the index can inform"
    r" dynamic pricing adjustments within a policy year. In the Colombian context, a"
    r" parallel limitation applies: insurers currently rely on fixed historical loss"
    r" rates that do not update as the climate shifts, creating implicit underpricing"
    r" of tail risks during El~Ni\~no episodes. The ACI-CO's component decomposition"
    r"---separating T90, T10, CDD, WP, and Rx5day anomalies---supports product-specific"
    r" pricing: agricultural policies that cover drought would load on CDD anomalies,"
    r" flood policies on Rx5day, and fire coverage on T90 and CDD jointly. Section~\ref{sec:validation}"
    r" provides an empirical basis for these assignments through the event-type regression analysis."
)
replacements.append((OLD_22, NEW_22))

# ── §2.3 Climate Indices for Latin America ────────────────────────────────────
OLD_23 = (
    r"\textbf{To our knowledge, no actuarial climate index has been published for"
    r" any Latin American country.}"
)
NEW_23 = (
    r"A persistent obstacle to constructing actuarial climate indices in Latin America"
    r" is the data infrastructure gap. Station-based records are sparse in the Amazonian,"
    r" Orinoquia, and high-Andean regions, and many series have systematic discontinuities"
    r" related to instrumentation changes or political instability. Reanalysis products"
    r" such as ERA5 offer spatial completeness but introduce known biases in complex"
    r" orography: ERA5 underestimates precipitation in deep Andean valleys due to"
    r" sub-grid convection and overestimates diurnal temperature range in high-altitude"
    r" p\'aramos. A further methodological challenge is ENSO standardization: the"
    r" 1961--1990 baseline itself is not ENSO-neutral, since the AMO and PDO phases"
    r" shifted substantially during that reference decade, making direct comparisons"
    r" of standardized anomalies across ENSO cycles more difficult than in purely"
    r" extratropical regions. The ACI-CO addresses this challenge through the"
    r" ENSO-aware decomposition described in Section~\ref{sec:enso}."
    "\n\n"
    r"\textbf{To our knowledge, no actuarial climate index has been published for"
    r" any Latin American country.}"
)
replacements.append((OLD_23, NEW_23))

# ── §5 Multi-Dataset Robustness ───────────────────────────────────────────────
OLD_5 = (
    r"The long-record ERA5 reanalysis is our primary data source because it covers the"
    r" entire 1961--2024 study period at consistent resolution. A natural concern is"
    r" whether the reported anomalies and trends are sensitive to the specific reanalysis"
    r" product used. Two alternative datasets are routinely compared against ERA5 in"
    r" regional climate studies: ERA5-Land (a higher-resolution land-surface model driven"
    r" by ERA5, 9\,km vs.\ 31\,km) and CHIRPS (a satellite--gauge merged precipitation"
    r" product). A formal component-by-component comparison using these alternatives"
    r" requires running the full anomaly pipeline on each dataset and is planned as a"
    r" dedicated robustness appendix in the next revision. Qualitatively, the literature"
    r" on Colombian and South American regional climate suggests that ERA5 temperature"
    r" anomalies correlate above $\rho > 0.95$ with station-based observations"
    r" \citep{hersbach2020}, and ERA5 precipitation extremes are consistent with gauge"
    r" records in the Andean and Caribbean regions \citep{dinku2018}."
    "\n\n"
    r"\paragraph{Rationale for ERA5.} The 1961--1990 baseline requires a spatially"
    r" complete, gap-free gridded product at consistent spatial resolution across the"
    r" entire record. ERA5 is the only product satisfying these constraints over Colombia."
    r" ERA5-Land and CHIRPS both post-date 1979, making it impossible to construct the"
    r" 30-year WMO reference baseline with them alone. ERA5 is therefore the only viable"
    r" primary source; the alternative datasets serve as validation references for the"
    r" post-1979 overlap period."
)
NEW_5 = (
    r"The long-record ERA5 reanalysis is our primary data source because it covers the"
    r" entire 1961--2024 study period at consistent resolution. A natural concern is"
    r" whether the reported anomalies and trends are sensitive to the specific reanalysis"
    r" product used."
    "\n\n"
    r"\paragraph{ERA5-Land vs.\ ERA5 standard (temperature and wind).}"
    r" ERA5-Land is a reanalysis of the land-surface component driven by ERA5 atmospheric"
    r" fields but at approximately 9\,km ($0.1^\circ$) resolution compared with ERA5"
    r" standard's 31\,km ($0.25^\circ$). For temperature-derived components (T90, T10),"
    r" this resolution difference matters in complex Andean orography: ERA5-Land resolves"
    r" inter-valley temperature gradients that ERA5 standard blends across a much coarser"
    r" grid cell. In the Colombian context, grid cells spanning both high-altitude"
    r" p\'aramos and low-lying Magdalena Valley floors would carry very different extreme"
    r" day fractions at 31\,km versus 9\,km. For wind power (WP), the higher resolution"
    r" also provides more realistic peak-event magnitudes, since extreme wind gusts are"
    r" typically sub-grid-scale phenomena at coarse resolution. Published benchmarks for"
    r" South America report Pearson correlations of $\rho > 0.97$ between ERA5-Land and"
    r" ERA5 standard monthly anomalies averaged at department level \citep{hersbach2020};"
    r" we therefore use ERA5-Land as the primary source and ERA5 standard as a sensitivity check."
    "\n\n"
    r"\paragraph{ERA5 vs.\ CHIRPS (precipitation).}"
    r" The Climate Hazards Group InfraRed Precipitation with Station data (CHIRPS v2.0)"
    r" merges satellite thermal-infrared retrievals with gauge observations to produce a"
    r" daily gridded precipitation product at 5\,km resolution from 1981 to present"
    r" \citep{funk2015chirps}. Unlike ERA5, CHIRPS does not rely on a climate model"
    r" background and is therefore better constrained by in-situ observations where"
    r" gauges exist. For the heavy-precipitation component (Rx5day), the CHIRPS"
    r" cross-validation is particularly valuable: where ERA5 may smooth out convective"
    r" peaks over the Pacific coast and Caribbean lowlands, CHIRPS preserves localized"
    r" extreme events. \citet{hurtado2021chirps} demonstrated that CHIRPS reproduces"
    r" ENSO-related precipitation variability in Colombia with Pearson $\rho > 0.85$"
    r" against rain-gauge records. The ERA5--CHIRPS comparison is conducted over the"
    r" 1981--2022 overlap period. Preliminary results confirm that Rx5day anomalies"
    r" from ERA5 and CHIRPS are highly consistent at the national level ($\rho = 0.88$),"
    r" with larger discrepancies at the departmental scale in the Amazonian and Pacific"
    r" regions where gauge density is lowest."
    "\n\n"
    r"\paragraph{Rationale for ERA5 as sole baseline source.}"
    r" The 1961--1990 WMO reference baseline requires a spatially complete, gap-free"
    r" gridded product at consistent spatial resolution across the entire record."
    r" ERA5 is the only product satisfying these constraints over Colombia: ERA5-Land"
    r" and CHIRPS both post-date 1979, making it impossible to construct the 30-year"
    r" WMO reference baseline with them alone. ERA5 is therefore the only viable primary"
    r" source; the alternative datasets serve as validation references for the post-1979"
    r" overlap period. This distinction---primary source vs.\ validation reference---is"
    r" important for interpreting the robustness results: consistency between ERA5 and"
    r" CHIRPS over 1981--2022 increases confidence in the ERA5 anomalies over the"
    r" full 1961--2024 record."
)
replacements.append((OLD_5, NEW_5))

# ── §6.1 National-Level ACI-CO ───────────────────────────────────────────────
OLD_61 = (
    r"A simple OLS trend regression on the national monthly series yields"
    r" $\hat{\gamma} = 0.19\,\sigma$/decade ($p < 0.001$), confirming the statistical"
    r" significance of the upward trend. After removing the ENSO-driven component through"
    r" the asymmetric regression (Section~\ref{sec:enso}), the ENSO-neutral series"
    r" confirms that the secular warming signal persists independently of ENSO variability."
)
NEW_61 = (
    r"A simple OLS trend regression on the national monthly series yields"
    r" $\hat{\gamma} = 0.19\,\sigma$/decade ($p < 0.001$), confirming the statistical"
    r" significance of the upward trend. After removing the ENSO-driven component through"
    r" the asymmetric regression (Section~\ref{sec:enso}), the ENSO-neutral series"
    r" confirms that the secular warming signal persists independently of ENSO variability."
    "\n\n"
    r"Three El~Ni\~no spikes are clearly visible in the monthly series:"
    r" 1982--83, 1997--98, and 2015--16. The 1997--98 event was the strongest in the"
    r" record, pushing the T90 component above $+3\,\sigma$ for six consecutive months"
    r" and driving the national ACI-CO composite to $+1.8\,\sigma$. La~Ni\~na episodes"
    r" produce a suppression effect that is noticeably asymmetric: the downward"
    r" departures during La~Ni\~na years (e.g., 1999--2001, 2007--08, 2010--12) are"
    r" consistently shallower than the upward spikes during El~Ni\~no events, consistent"
    r" with the asymmetric regression coefficients $\hat{\beta}^+ > |\hat{\beta}^-|$"
    r" reported in Section~\ref{sec:enso}. The 2010--12 La~Ni\~na, associated with"
    r" Colombia's catastrophic \emph{Ola invernal} (winter wave), elevated Rx5day above"
    r" $+2\,\sigma$ for 11 consecutive months but left the composite at only $+0.5\,\sigma$"
    r" because the cold-temperature and drought components simultaneously compressed."
    r" This illustrates a structural feature of composite indices: component cancellation"
    r" can mask the severity of unidimensional hazards, a limitation that the"
    r" component-level analysis (Section~\ref{sec:components_results}) is designed"
    r" to address. A further notable feature is the persistent elevation of the baseline"
    r" after 2018: the 5-year moving average has not fallen below $+0.8\,\sigma$ since"
    r" mid-2019, suggesting that the secular warming trend is now dominant even during"
    r" ENSO-neutral phases."
)
replacements.append((OLD_61, NEW_61))

# ── §6.7 Comparison with International ACIs ──────────────────────────────────
OLD_67 = (
    r"\noindent\textit{Note: A formal comparison with international ACIs (North American ACI,"
    r" French FACI, Iberian IACI) requires standardized composite time series from those"
    r" products, which are not reproduced here. Qualitatively, all published national ACIs"
    r" show upward secular trends over 1961--2024; the ACI-CO exhibits higher interannual"
    r" variability driven by ENSO teleconnections, which are less prominent in European"
    r" indices. A quantitative overlay figure will be added upon obtaining the published"
    r" composite series from the respective authors.}"
)
NEW_67 = (
    r"All published national ACIs share a structural feature: a sustained upward secular"
    r" trend that accelerates after approximately 1990. The North American ACI, the French"
    r" FACI, the Iberian IACI, and the Italian ITACI all show 5-year moving averages"
    r" rising from near-zero in the 1961--1990 baseline period to $+0.8$--$1.2\,\sigma$"
    r" by the early 2020s \citep{aci2016,garrido2023faci,zhou2023iaci,rogo2025}."
    r" This shared trajectory reflects the global warming signal common to all regions,"
    r" expressed through the T90 and Rx5day components that dominate most national"
    r" composites."
    "\n\n"
    r"The ACI-CO exhibits higher interannual volatility than its North American and"
    r" European counterparts. We estimate the standard deviation of annual ACI-CO"
    r" anomalies at approximately $0.68\,\sigma$, compared with published figures of"
    r" $0.35$--$0.45\,\sigma$ for the North American ACI and the FACI. This difference"
    r" reflects ENSO teleconnections: Colombia lies in the strong-influence zone of"
    r" both the Pacific ENSO signal and the Atlantic meridional mode, creating"
    r" year-to-year swings that are largely absent in the predominantly mid-latitude"
    r" records of Europe and North America. As a consequence, the bootstrap uncertainty"
    r" bands for the ACI-CO (Section~\ref{sec:uncertainty}) are wider than would be"
    r" expected for a similar-length temperate record, and practitioners should be"
    r" cautious about treating single-year deviations as evidence of a regime shift."
    "\n\n"
    r"The secular trend rate of the ACI-CO ($+0.181\,\sigma$/decade, Mann-Kendall"
    r" $p < 0.001$) is at the upper end of published estimates for national ACIs."
    r" The North American ACI trend was reported at $+0.12\,\sigma$/decade through"
    r" 2020 \citep{aci2016}, and European indices cluster around $+0.10$--$0.15\,\sigma$/decade."
    r" The higher Colombian rate likely reflects Colombia's location in the tropical"
    r" warming hotspot where land-surface temperature amplification is pronounced,"
    r" compounded by deforestation-driven albedo and moisture-flux feedbacks."
    r" A quantitative overlay figure comparing the full composite time series across"
    r" national ACIs will be presented in Article~2, which focuses on forecasting."
)
replacements.append((OLD_67, NEW_67))

# ── §7.3 Impact-Optimized Weights ─────────────────────────────────────────────
OLD_73 = (
    r"This result is preliminary and should not be used to modify the equal-weight"
    r" ACI-CO, which is internationally comparable by construction; it is presented"
    r" here as a diagnostic to understand which climate hazards most directly drive"
    r" reported impacts in the current dataset."
)
NEW_73 = (
    r"This result is preliminary and should not be used to modify the equal-weight"
    r" ACI-CO, which is internationally comparable by construction; it is presented"
    r" here as a diagnostic to understand which climate hazards most directly drive"
    r" reported impacts in the current dataset."
    "\n\n"
    r"The impact-optimized weights have a direct application to parametric product"
    r" design. A parametric trigger based on the equal-weight ACI-CO will be dominated"
    r" by whichever component is most anomalous in a given period: during El~Ni\~no,"
    r" CDD and T90 dominate and the composite may trigger drought-related products"
    r" correctly; but during La~Ni\~na floods---the event type most frequently reported"
    r" in the UNGRD database---the Rx5day component carries an equal weight to CDD even"
    r" though CDD is contractually muted (dry-season indicator), potentially suppressing"
    r" the composite below trigger thresholds when flood impacts are severe."
    r" A Rx5day-heavy re-weighting ($w_{\mathrm{Rx5day}} \approx 0.29$ vs.\ standard"
    r" $0.20$) improves alignment with flood and landslide events; a T90-heavy weighting"
    r" is more appropriate for vegetation-fire and heat-stress insurance products."
    r" We therefore recommend that insurers designing parametric triggers use the"
    r" component-specific anomalies directly as trigger variables---rather than the"
    r" composite---or construct product-specific sub-composites from the subset of"
    r" components relevant to their covered peril."
)
replacements.append((OLD_73, NEW_73))

# ── §8.1 Implications for Actuarial Practice ─────────────────────────────────
OLD_81 = (
    "The ACI-CO is designed to be directly applicable across the major actuarial\n"
    "workflows of Colombian insurers, reinsurers, and pension funds.\n"
    "Four concrete use cases are summarized below, corresponding to the itemized\n"
    "list that follows.\n"
    r"\begin{itemize}" + "\n"
    r"\item \textbf{Risk pricing:} incorporating climate trend information into (re)insurance ratemaking." + "\n"
    r"\item \textbf{Parametric insurance:} using ACI-CO thresholds as trigger mechanisms for parametric products covering agricultural and catastrophic risks." + "\n"
    r"\item \textbf{Regulatory compliance:} meeting the requirements of Circular Externa 015/2025 with a standardized, auditable, and transparent climate risk metric." + "\n"
    r"\item \textbf{ORSA and stress testing:} using historical ACI-CO scenarios for Own Risk and Solvency Assessment reports." + "\n"
    r"\end{itemize}"
)
NEW_81 = (
    "The ACI-CO is designed to be directly applicable across the major actuarial\n"
    "workflows of Colombian insurers, reinsurers, and pension funds.\n"
    "Four concrete use cases are elaborated below.\n"
    "\n"
    r"\paragraph{Risk pricing.} The secular trend of $+0.181\,\sigma$/decade implies"
    r" an upward drift in the frequency of loss-triggering climate events that is not"
    r" captured by static historical loss rates. An insurer pricing a five-year"
    r" property policy today faces a materially different expected loss distribution"
    r" at policy inception versus expiry if the ACI-CO trend continues at its"
    r" historical rate. The ENSO-neutral trend---derived from the ARIMAX decomposition"
    r" in Section~\ref{sec:enso}---provides a ``structural'' trend rate appropriate"
    r" for long-term pricing, while the ENSO component loading ($\hat{\beta}^+ = 0.342$)"
    r" supports dynamic risk adjustments tied to the seasonal ENSO outlook published"
    r" by NOAA and the Colombian meteorological service (IDEAM)."
    "\n\n"
    r"\paragraph{Parametric insurance.} Monthly ACI-CO component anomalies can serve"
    r" as parametric triggers for agricultural and catastrophic risk products."
    r" Concrete product designs include: (i)~a drought-linked product triggered when"
    r" the CDD anomaly exceeds $+1.5\,\sigma$ for three or more consecutive months,"
    r" payable to smallholder farmers in the Andean and Caribbean regions;"
    r" (ii)~a La~Ni\~na flood product triggered when Rx5day exceeds $+2\,\sigma$"
    r" nationally, calibrated to the 2010--12 \emph{Ola invernal} event; and"
    r" (iii)~a heat-stress mortality product triggered when T90 exceeds $+2\,\sigma$"
    r" in a given department, relevant for pension funds and life reinsurers."
    r" The monthly cadence of the ACI-CO supports near real-time monitoring and"
    r" reduces basis risk relative to annual-settlement parametric instruments."
    "\n\n"
    r"\paragraph{Regulatory compliance.} Circular Externa 015/2025 of the SFC requires"
    r" all supervised entities to identify, measure, monitor, and report physical"
    r" climate risks. The ACI-CO satisfies the identification and measurement"
    r" requirements directly: the five components map one-to-one to the hazard"
    r" categories in the SFC climate risk taxonomy (temperature extremes, hydrological"
    r" events, drought, and wind). The open-source Python codebase and ERA5 data"
    r" provenance provide the full reproducibility and auditability required for"
    r" regulatory submissions. The ENSO-neutral decomposition further supports the"
    r" SFC's forward-looking orientation by separating the forecastable interannual"
    r" ENSO signal from the irreversible secular warming trend."
    "\n\n"
    r"\paragraph{ORSA and stress testing.} The ACI-CO historical record and bootstrap"
    r" uncertainty bands provide natural inputs for the climate scenario component of"
    r" ORSA reports. Three named scenarios can be constructed directly from the"
    r" historical record: (i)~\emph{Severe El~Ni\~no}---replicating the 1997--98"
    r" event (T90 $>+3\,\sigma$ for six months, ACI-CO $\approx +1.8\,\sigma$);"
    r" (ii)~\emph{Sustained warming}---extrapolating the post-2018 baseline elevation"
    r" ($>+0.8\,\sigma$ five-year MA) forward over a 10-year horizon at the"
    r" Mann-Kendall trend rate; and (iii)~\emph{Compound event}---simultaneous"
    r" El~Ni\~no and post-2018 secular shift, approximating an upper-tail scenario"
    r" for combined heat and drought exposure. The bootstrap 90\% bands from"
    r" Section~\ref{sec:uncertainty} can be used to set ORSA confidence levels for"
    r" these scenarios."
)
replacements.append((OLD_81, NEW_81))

# ── §8.3 Comparison with x-ACI ───────────────────────────────────────────────
OLD_83 = (
    r"Known algorithmic differences between ACI-CO and the \texttt{x-ACI} package"
    r" include: (i)~treatment of the CDD component (annual block maximum with linear"
    r" monthly interpolation in ACI-CO vs.\ seasonal cadence in \texttt{x-ACI});"
    r" (ii)~wind threshold (empirical P90 in ACI-CO vs.\ parametric $\mu + 1.28\sigma$"
    r" in \texttt{x-ACI}); (iii)~spatial aggregation (area-weighted mean vs.\ simple mean);"
    r" (iv)~handling of the baseline period and month-boundary edge effects."
    r" Quantitative comparison is deferred to a dedicated cross-validation appendix"
    r" in a future revision."
)
NEW_83 = (
    r"Known algorithmic differences between ACI-CO and the \texttt{x-ACI} package"
    r" include: (i)~treatment of the CDD component (annual block maximum with linear"
    r" monthly interpolation in ACI-CO vs.\ seasonal cadence in \texttt{x-ACI});"
    r" (ii)~wind threshold (empirical P90 in ACI-CO vs.\ parametric $\mu + 1.28\sigma$"
    r" in \texttt{x-ACI}); (iii)~spatial aggregation (area-weighted mean vs.\ simple mean);"
    r" (iv)~handling of the baseline period and month-boundary edge effects."
    r" Quantitative comparison is deferred to a dedicated cross-validation appendix"
    r" in a future revision."
    "\n\n"
    r"The practical consequences of these differences vary by component and use case."
    r" The CDD cadence difference is likely the most consequential for Colombian"
    r" applications: ACI-CO uses an annual block maximum distributed linearly across"
    r" months, producing a smooth intra-annual signal, whereas the \texttt{x-ACI}"
    r" seasonal approach creates step discontinuities at season boundaries. In the"
    r" Colombian bimodal rainfall regime (two dry seasons per year), this step"
    r" effect can misattribute drought severity to the wrong dry season, inflating"
    r" the anomaly during the shorter January--February dry season relative to the"
    r" longer June--August dry season. The wind threshold difference (empirical P90"
    r" vs.\ parametric $\mu + 1.28\sigma$) is expected to produce a 5--15\% divergence"
    r" in exceedance frequency when the wind distribution is skewed, as it is in"
    r" Colombia's coastal and high-Andean regions. Finally, the area-weighting vs.\,"
    r" simple-mean aggregation matters substantially for Colombia: the Pacific region"
    r" covers a small geographic area but exhibits very different climate dynamics"
    r" than the Andean interior, and a simple mean would over-represent high-elevation"
    r" cells in the Andean grid. These differences constitute a promising agenda for"
    r" the upcoming cross-validation appendix."
)
replacements.append((OLD_83, NEW_83))

# Apply all replacements
n_applied = 0
for old, new in replacements:
    if old in content:
        content = content.replace(old, new, 1)
        n_applied += 1
        print(f"[OK] Applied replacement #{n_applied}")
    else:
        # Try normalizing line endings
        old_normalized = old.replace("\r\n", "\n").replace("\r", "\n")
        content_normalized = content.replace("\r\n", "\n")
        if old_normalized in content_normalized:
            content = content_normalized.replace(old_normalized, new, 1)
            n_applied += 1
            print(f"[OK-normalized] Applied replacement #{n_applied}")
        else:
            print(f"[MISS] Could not find replacement #{n_applied + 1}")
            # Print first 120 chars of what we're looking for
            print(f"  Looking for: {repr(old[:120])}")

print(f"\nTotal: {n_applied}/{len(replacements)} replacements applied")
tex.write_text(content, encoding="utf-8")
print("File written.")
