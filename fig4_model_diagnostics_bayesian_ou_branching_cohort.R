## ============================================================
## Figure 4: Model diagnostics for Bayesian OU–Branching cohort
## Input: kmt2a_cohort_diagnostics.csv
## Output: Figure4_model_diagnostics.(png|pdf)
## @Author: seung-hwan.kim
## ============================================================

# Load packages ------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(tidyr)
library(patchwork)

# Read diagnostics data ----------------------------------------------------
diag_df <- read.csv("kmt2a_cohort_diagnostics.csv", stringsAsFactors = FALSE)

# Inspect (optional)
# str(diag_df)
# head(diag_df)

# Ensure expected columns exist:
# patient_id, group, rmse, ppc_95, r2_bayes, mean_loglik

# Order clinical groups (adjust if your spelling differs) ------------------
diag_df <- diag_df %>%
  mutate(
    group = factor(
      group,
      levels = c("Very early",
                 "Very early/refractory",
                 "Early",
                 "Early/refractory",
                 "Late",
                 "Remission")
    )
  )

# Global plotting theme ----------------------------------------------------
theme_fig4 <- theme_bw(base_size = 11) +
  theme(
    plot.title   = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(margin = margin(t = 6)),
    axis.title.y = element_text(margin = margin(r = 6)),
    axis.text.x  = element_text(angle = 30, hjust = 1)
  )

# -------------------------------------------------------------------------
# Panel A: RMSE by clinical group
# -------------------------------------------------------------------------
pA <- ggplot(diag_df_clean, aes(x = group, y = rmse)) +
  geom_violin(trim = FALSE, fill = "grey90", color = "grey40") +
  geom_boxplot(width = 0.15, outlier.shape = NA, fill = "white") +
  geom_jitter(width = 0.08, height = 0, alpha = 0.6, size = 1) +
  labs(
    x = "Clinical group",
    y = "RMSE",
    title = "A. RMSE by clinical group"
  ) +
  theme_fig4

# -------------------------------------------------------------------------
# Panel B: Posterior predictive coverage (95% band) by group
# -------------------------------------------------------------------------
pB <- ggplot(diag_df_clean, aes(x = group, y = ppc_95)) +
  geom_violin(trim = FALSE, fill = "grey90", color = "grey40") +
  geom_boxplot(width = 0.15, outlier.shape = NA, fill = "white") +
  geom_jitter(width = 0.08, height = 0, alpha = 0.6, size = 1) +
  geom_hline(yintercept = 0.95, linetype = "dashed") +
  coord_cartesian(ylim = c(0, 1.1)) +
  labs(
    x = "Clinical group",
    y = "Coverage of 95% PPC band",
    title = "B. Posterior predictive coverage"
  ) +
  theme_fig4

# -------------------------------------------------------------------------
# Panel C: Bayesian R^2 distribution across patients
# NOTE: If extreme negative values compress the scale,
# adjust 'limits' or breaks in scale_x_continuous().
# -------------------------------------------------------------------------
pC <- ggplot(diag_df_clean, aes(x = r2_bayes)) +
  geom_histogram(bins = 20, fill = "grey85", color = "grey40") +
  xlim(-30, 1) +
  geom_vline(xintercept = 0, linetype = "dotted") +
  labs(
    x = expression(paste("Bayesian ", R^2)),
    y = "Number of patients",
    title = "C. Bayesian model fit"
  ) +
  theme_fig4 +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))

# -------------------------------------------------------------------------
# Panel D: Spearman correlation among diagnostics
# -------------------------------------------------------------------------
cor_mat <- diag_df_clean %>%
  select(rmse, ppc_95, r2_bayes, mean_loglik) %>%
  cor(method = "spearman", use = "pairwise.complete.obs")

# Winsorize extreme R^2 values for plotting & correlations
diag_df_clean <- diag_df %>%
  mutate(r2_bayes = pmax(r2_bayes, -30))   # choose threshold

cor_df <- as.data.frame(as.table(cor_mat))
colnames(cor_df) <- c("metric1", "metric2", "rho")

pD <- ggplot(cor_df, aes(x = metric1, y = metric2, fill = rho)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(
    name   = expression(rho),
    limits = c(-1, 1),
    breaks = c(-1, -0.5, 0, 0.5, 1)
  ) +
  labs(
    x = NULL,
    y = NULL,
    title = "D. Spearman correlation among diagnostics"
  ) +
  theme_fig4 +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1),
    axis.text.y = element_text(angle = 0, hjust = 1)
  )

# -------------------------------------------------------------------------
# Combine into 2x2 layout and export
# -------------------------------------------------------------------------
fig4 <- (pA | pB) / (pC | pD)

# Save PNG (for manuscript submission)
ggsave(
  filename = "Figure4_model_diagnostics.png",
  plot     = fig4,
  width    = 8.5,   # inches
  height   = 6.5,
  dpi      = 300
)

# Save PDF (for vector graphics)
ggsave(
  filename = "Figure4_model_diagnostics.pdf",
  plot     = fig4,
  width    = 8.5,
  height   = 6.5
)