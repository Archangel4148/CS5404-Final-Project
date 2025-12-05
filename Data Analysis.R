library(readr)
library(dplyr)
library(lme4)
library(lmerTest)
library(emmeans)
library(ggplot2)

data <- read_csv("Fall 2025/Computer Vision/Final Project/parsed_results-re-evaluated.csv")

data$object <- factor(data$object)
data$blurLevel <- factor(data$blurLevel, levels = c("none","low","high"))
data$noiseLevel <- factor(data$noiseLevel, levels = c("none","low","high"))
data$exposureLevel <- factor(data$exposureLevel, levels = c("none","low","high"))

data$chamferDistance <- as.numeric(data$chamferDistance)
data$F0.1 <- as.numeric(data$F0.1)
data$F0.2 <- as.numeric(data$F0.2)
data$F0.5 <- as.numeric(data$F0.5)

distortions <- c("blurLevel","noiseLevel","exposureLevel")
responses <- c("chamferDistance","F0.1","F0.2","F0.5")

results_table <- data.frame(
  Distortion = character(),
  Response = character(),
  Level = character(),
  MeanDiff_vs_Original = numeric(),
  p_value = numeric(),
  Significant = character(),
  stringsAsFactors = FALSE
)

for (dist in distortions) {
  for (resp in responses) {
    #Mixed-effects model with object as random effect
    formula <- as.formula(paste(resp, "~", dist, "+ (1|object)"))
    model <- lmer(formula, data = data)
    
    #Pairwise comparison vs original
    emm <- emmeans(model, specs = dist)
    contrast_table <- contrast(emm, method = "trt.vs.ctrl", ref = "none")
    contrast_df <- as.data.frame(contrast_table)
    
    #sig col
    contrast_df$Significant <- ifelse(contrast_df$p.value < 0.05, "Yes", "No")
    
    contrast_df <- contrast_df %>%
      mutate(
        Distortion = dist,
        Response = resp,
        Level = contrast
      ) %>%
      select(Distortion, Response, Level, estimate, p.value, Significant)
    
    names(contrast_df)[4] <- "MeanDiff_vs_Original"
    
    # Append to results table
    results_table <- rbind(results_table, contrast_df)
  }
}
#Final Table:
View(results_table)

for (name in names(results)) {
  cat("\n=== ANOVA for", name, "===\n")
  print(results[[name]]$anova)
}

for (name in names(results)) {
  cat("\n=== Contrasts vs original for", name, "===\n")
  print(results[[name]]$contrast)
}

results <- list()

for (dist in distortions) {
  for (resp in responses) {
  
    formula <- as.formula(paste(resp, "~", dist, "+ (1|object)"))
    model <- lmer(formula, data = data)
    
    anova_table <- anova(model)
    
    emm <- emmeans(model, specs = dist)
    contrast_table <- contrast(emm, method = "trt.vs.ctrl", ref = "none")
    
    results[[paste(dist, resp, sep = "_")]] <- list(
      anova = anova_table,
      contrast = contrast_table
    )
  }
}

#-----------------Plots-----------------------------

# Q-Q plot
qqnorm(resid_vals)
qqline(resid_vals)

# Histogram
ggplot(data.frame(resid = resid_vals), aes(x = resid)) +
  geom_histogram(bins = 15, fill = "steelblue", color = "black") +
  theme_minimal() +
  ggtitle("Histogram of residuals (blur -> chamferDistance)")

# Residuals vs fitted
fitted_vals <- fitted(model)
ggplot(data.frame(fitted = fitted_vals, resid = resid_vals, group = data$blurLevel), aes(x = fitted, y = resid, color = group)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  theme_minimal() +
  ggtitle("Residuals vs Fitted (blur -> chamferDistance)")


