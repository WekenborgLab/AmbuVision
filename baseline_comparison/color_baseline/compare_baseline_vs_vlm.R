#!/usr/bin/env Rscript
# =============================================================================
# Reviewer 2, Comment 6 -- fair head-to-head: cheap color baseline vs. VLM
# =============================================================================
#
# This reproduces the EXACT statistical pipeline of the manuscript so the
# cheap color baseline is evaluated identically to the VLM:
#
#   * Table 1  : multilevel model predicting the greenness indicator from
#                SUBJECTIVE (self-reported) greenness, entered at
#                Level 2 (between-person, person means) and
#                Level 1 (within-person, person-mean-centered).
#                  lmer(greenness ~ subj_L2 + subj_L1 + (1 | participant))
#
#   * Table 2  : multilevel model predicting the greenness indicator from
#                positive & negative affect, each at Level 2 and Level 1.
#                  lmer(greenness ~ PA_L2 + PA_L1 + NA_L2 + NA_L1
#                                   + (1 | participant))
#
#   * Stress   : Pearson correlation between the participant-mean greenness
#                indicator and PSS chronic stress (complementary analysis).
#
# For each model it reports, in the SAME layout as the paper's tables:
#   Estimates, 95% CI, p-value, sigma^2, tau00, ICC,
#   marginal/conditional R^2, N participants, observations.
#
# Run it once on the VLM greenness.csv and once on the color-baseline
# greenness.csv -> the two output blocks are directly comparable, and a
# side-by-side comparison CSV is written. Because both inputs share the
# identical schema (PictureId, Folder, Greenness, ...) the baseline is a
# true drop-in for the VLM rater.
#
# Matches the manuscript environment: R 4.4.x, lme4 1.1-35.x,
# lmerTest (Satterthwaite p-values), MuMIn (R^2).
#
# -----------------------------------------------------------------------------
# USAGE
#   Rscript compare_baseline_vs_vlm.R \
#       --greenness   greenness.csv          \  # VLM or baseline file
#       --modelling   modelling_dataset.csv  \  # OSF self-report dataset
#       --label       baseline               \  # tag for the output
#       --outdir      .
#
#   To get the comparison, run twice (label=vlm and label=baseline) into the
#   same --outdir; the second run appends and writes comparison_summary.csv.
#
# -----------------------------------------------------------------------------
# >>> THE ONE THING YOU MUST PROVIDE: the modelling dataset <<<
# The self-reports (subjective greenness, positive/negative affect, PSS) are
# NOT in this repo -- they are the OSF file
# "Outputs from the VLM analyses used for statistical modelling, together
#  with anonymized participant-level data". Point --modelling at it and set
# the column names + join key in the CONFIG block below to match it.
# =============================================================================

suppressWarnings(suppressMessages({
  need <- c("lme4", "lmerTest", "MuMIn", "optparse")
  miss <- need[!sapply(need, requireNamespace, quietly = TRUE)]
  if (length(miss))
    stop("Missing R packages: ", paste(miss, collapse = ", "),
         "\nInstall with: install.packages(c(",
         paste(sprintf('\"%s\"', miss), collapse = ", "), "))")
  library(lme4); library(lmerTest); library(MuMIn); library(optparse)
}))

# ----------------------------- CONFIG ---------------------------------------
# Edit these to match the column names in YOUR modelling dataset.
CFG <- list(
  participant_col  = "Folder",        # participant id in BOTH files
  # join key between greenness.csv and the modelling dataset:
  #   "PictureId"  -> exact image filename match (preferred, exact)
  #   "TimeCreated"-> EXIF timestamp match (fallback; same logic as VLM)
  join_cols        = c("Folder", "PictureId"),
  greenness_col    = "Greenness",     # the indicator column in greenness.csv
  subj_green_col   = "subj_greenness",# self-reported momentary greenness
  pos_affect_col   = "positive_affect",
  neg_affect_col   = "negative_affect",
  pss_col          = "PSS"            # chronic stress (participant-level)
)
# ----------------------------------------------------------------------------

opt <- parse_args(OptionParser(option_list = list(
  make_option("--greenness", type = "character"),
  make_option("--modelling", type = "character"),
  make_option("--label",     type = "character", default = "model"),
  make_option("--outdir",    type = "character", default = ".")
)))
stopifnot(!is.null(opt$greenness), !is.null(opt$modelling))
dir.create(opt$outdir, showWarnings = FALSE, recursive = TRUE)

g <- read.csv(opt$greenness, stringsAsFactors = FALSE, check.names = FALSE)
m <- read.csv(opt$modelling, stringsAsFactors = FALSE, check.names = FALSE)

# --- merge greenness indicator onto the self-report rows --------------------
jc <- CFG$join_cols
miss_g <- setdiff(c(jc, CFG$greenness_col), names(g))
miss_m <- setdiff(jc, names(m))
if (length(miss_g)) stop("greenness file missing columns: ",
                         paste(miss_g, collapse = ", "))
if (length(miss_m)) stop("modelling file missing join columns: ",
                         paste(miss_m, collapse = ", "),
                         " -- adjust CFG$join_cols.")
g2 <- g[, c(jc, CFG$greenness_col)]
names(g2)[names(g2) == CFG$greenness_col] <- "GREEN"
d  <- merge(m, g2, by = jc, all.x = FALSE)
message(sprintf("Merged %d photo-level rows for %d participants (label=%s).",
                nrow(d), length(unique(d[[CFG$participant_col]])), opt$label))

pid <- d[[CFG$participant_col]]

# Level-2 = person mean ; Level-1 = person-mean-centered (Methods p.16-17)
center2 <- function(x) {
  mu <- tapply(x, pid, mean, na.rm = TRUE)
  L2 <- as.numeric(mu[as.character(pid)])
  list(L2 = L2, L1 = x - L2, mu = mu)
}

# z-standardise so VLM vs baseline COEFFICIENTS are directly comparable
# (paper reports raw-scale coefficients; we report both: raw table layout
#  plus standardized betas for the head-to-head).
sg <- center2(d[[CFG$subj_green_col]])
pa <- center2(d[[CFG$pos_affect_col]])
na <- center2(d[[CFG$neg_affect_col]])
d$GREEN_L2 <- NA  # outcome stays raw for the paper-format table

fit_block <- function(form, data, predictors) {
  fm  <- lmerTest::lmer(form, data = data, REML = TRUE,
                        control = lmerControl(optimizer = "bobyqa"))
  cf  <- summary(fm)$coefficients
  ci  <- tryCatch(confint(fm, method = "Wald")[rownames(cf), , drop = FALSE],
                  error = function(e) matrix(NA, nrow(cf), 2))
  vc  <- as.data.frame(lme4::VarCorr(fm))
  s2  <- vc$vcov[vc$grp == "Residual"]
  t00 <- vc$vcov[vc$grp == CFG$participant_col]
  r2  <- tryCatch(MuMIn::r.squaredGLMM(fm)[1, ], error = function(e) c(NA, NA))
  list(
    table = data.frame(
      Predictor   = rownames(cf),
      Estimate    = round(cf[, "Estimate"], 3),
      CI_low      = round(ci[, 1], 3),
      CI_high     = round(ci[, 2], 3),
      p_value     = signif(cf[, "Pr(>|t|)"], 3),
      row.names   = NULL),
    sigma2 = round(s2, 3), tau00 = round(t00, 3),
    ICC    = round(t00 / (t00 + s2), 3),
    R2m    = round(r2[1], 3), R2c = round(r2[2], 3),
    N      = length(unique(pid)), Obs = nrow(data))
}

emit <- function(title, blk) {
  cat("\n", strrep("=", 78), "\n", title, "  [label: ", opt$label, "]\n",
      strrep("=", 78), "\n", sep = "")
  print(blk$table, row.names = FALSE)
  cat(sprintf("\n  sigma^2=%.3f  tau00=%.3f  ICC=%.3f  R2m=%.3f  R2c=%.3f",
              blk$sigma2, blk$tau00, blk$ICC, blk$R2m, blk$R2c))
  cat(sprintf("\n  N=%d participants   Observations=%d\n", blk$N, blk$Obs))
}

# ---- Table 1: greenness ~ subjective greenness (L2 + L1) -------------------
d1 <- data.frame(GREEN = d$GREEN, pid = pid,
                  subj_L2 = sg$L2, subj_L1 = sg$L1)
t1 <- fit_block(GREEN ~ subj_L2 + subj_L1 + (1 | pid), d1)
emit("TABLE 1  -- indicator predicted by SUBJECTIVE greenness", t1)

# ---- Table 2: greenness ~ positive & negative affect (L2 + L1) -------------
d2 <- data.frame(GREEN = d$GREEN, pid = pid,
                  PA_L2 = pa$L2, PA_L1 = pa$L1,
                  NA_L2 = na$L2, NA_L1 = na$L1)
t2 <- fit_block(GREEN ~ PA_L2 + PA_L1 + NA_L2 + NA_L1 + (1 | pid), d2)
emit("TABLE 2  -- indicator predicted by AFFECT", t2)

# ---- Stress: Pearson r of participant-mean indicator vs PSS ----------------
gmu  <- tapply(d$GREEN, pid, mean, na.rm = TRUE)
pss  <- tapply(d[[CFG$pss_col]], pid, function(x) x[1])  # 1 value/person
ct   <- cor.test(as.numeric(gmu), as.numeric(pss[names(gmu)]),
                 method = "pearson")
cat("\n", strrep("=", 78),
    "\nSTRESS  -- Pearson r( participant-mean indicator , PSS )  [label: ",
    opt$label, "]\n", strrep("=", 78), "\n", sep = "")
cat(sprintf("  r = %.3f   95%% CI [%.3f, %.3f]   p = %.4f   n = %d\n",
            ct$estimate, ct$conf.int[1], ct$conf.int[2],
            ct$p.value, sum(!is.na(gmu))))

# ---- machine-readable summary (for the side-by-side comparison) ------------
summ <- data.frame(
  label              = opt$label,
  T1_subjL1_beta     = t1$table$Estimate[t1$table$Predictor == "subj_L1"],
  T1_subjL1_p        = t1$table$p_value [t1$table$Predictor == "subj_L1"],
  T1_subjL2_beta     = t1$table$Estimate[t1$table$Predictor == "subj_L2"],
  T1_subjL2_p        = t1$table$p_value [t1$table$Predictor == "subj_L2"],
  T1_R2m = t1$R2m, T1_R2c = t1$R2c, T1_ICC = t1$ICC,
  T2_PAL2_beta = t2$table$Estimate[t2$table$Predictor == "PA_L2"],
  T2_PAL2_p    = t2$table$p_value [t2$table$Predictor == "PA_L2"],
  T2_NAL2_beta = t2$table$Estimate[t2$table$Predictor == "NA_L2"],
  T2_NAL2_p    = t2$table$p_value [t2$table$Predictor == "NA_L2"],
  T2_R2m = t2$R2m, T2_R2c = t2$R2c,
  stress_r = round(unname(ct$estimate), 3),
  stress_p = signif(ct$p.value, 3),
  N = t1$N, Obs_T1 = t1$Obs)

cmp <- file.path(opt$outdir, "comparison_summary.csv")
if (file.exists(cmp)) {
  prev <- read.csv(cmp, stringsAsFactors = FALSE)
  summ <- rbind(prev[prev$label != opt$label, , drop = FALSE], summ)
}
write.csv(summ, cmp, row.names = FALSE)
cat("\nAppended results to", cmp,
    "\nRun with --label vlm AND --label baseline to get the head-to-head.\n")
