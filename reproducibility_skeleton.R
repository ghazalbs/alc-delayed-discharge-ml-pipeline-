# reproducibility_skeleton.R
# High-level analysis skeleton 
# Purpose: document the end-to-end pipeline, key parameters, and outputs.

# ---- Packages (illustrative) ----
library(dplyr)
library(tidyr)
library(pROC)
library(xgboost)
library(ggplot2)
library(DALEX)
library(modelStudio)
library(caret)
library(fairness)

# ---- Parameters ----
params <- list(
  seed = 42,
  # Windows/cohort
  n=2,
  lookback_days  = n*365,
  lookahead_days = c(90,180,365),
  outcome_name   = "ALC",
  # Splits
  split = list(train = 0.90, test = 0.10),
  temporal = list(train_end = "2014", temporal_start = "2015"),
  # Models
  models = c("LR", "XGB"),
  # Tuning
  cv_folds = 5,
  # Calibration
  calibration = list(method = "platt", group_stratified = TRUE,
                     groups = c("sex", "residency", "instability")),
  # Fairness (threshold-free)
  fairness_groups = c("sex", "residency", "instability"),
 # Evaluation metrics
    eval_metrics = c("AUROC", "Brier", "ECE", "Cal_Intercept", "Cal_Slope"),
 # Fairness metrics
    fairness_metrics = c("AUC_parity", "xAUC_parity", "ECE_parity"),
 # Explainability
    explainability = list(local = TRUE, global = TRUE),
 # Outputs
  out_dir = "outputs/"
)

# ---- 1) Data access (ICES-restricted placeholders) ----
load_ices_analytic_dataset <- function() {
        
  # Placeholder: load pre-derived analytic dataset inside ICES environment
  # Must return an episode-level dataframe with outcome + predictors
  # Each patient may have multiple records (visits/hospital admissions)
  df <- read.csv("...")  # not shareable
  return(df)
  stop("ICES-restricted: analytic dataset loading placeholder.")
}

# ---- 2) Cohort + feature preparation ----
prepare_analysis_data <- function(df, params) {
    # Apply inclusion/exclusion criteria to filter the dataset
    # Define the index date and outcome window (look-ahead period)
    # Construct predictors including demographic, socioeconomic, geographic,
    # clinical, and administrative features
    # Ensure the resulting dataframe is episode-level with the necessary predictors
   return(df_model)    
}

# ---- 2b) Missingness handling ----
handle_missingness <- function(df_model) {
    # Strategy: Complete-case analysis (exclude rows with any missing values)
    # Rationale: Avoid imputation assumptions; assume missingness is negligible
    
    n_before <- nrow(df_model)
    df_clean <- df_model[complete.cases(df_model), ]
    n_after <- nrow(df_clean)
    n_removed <- n_before - n_after
    pct_removed <- round(100 * n_removed / n_before, 2)
    
    cat(sprintf(
        "Missingness summary:\n  Before: %d rows\n  After: %d rows\n  Removed: %d (%.2f%%)\n",
        n_before, n_after, n_removed, pct_removed
    ))
    
    return(df_clean)
}

# ---- 3) Splits (random + temporal hold-out) ----
split_random <- function(df, params,
                                id_col = "ID",
                                outcome_col = "ALC",
                                train_p = 0.90) {
  # Split at the patient (ID) level to avoid leakage when patients have multiple rows.
  # 1) Create ID-level outcome for stratified sampling (e.g., max across a patient's episodes)
  patient_summary <- aggregate(df[[outcome_col]],
                               by = list(df[[id_col]]),
                               FUN = max)

  colnames(patient_summary) <- c(id_col, "outcome_id")

  set.seed(params$seed)

  # 2) Stratified sampling of IDs into train/test (implementation may use caret::createDataPartition)
  # train_index <- caret::createDataPartition(patient_summary$outcome_id, p = train_p, list = FALSE)
  train_index <- seq_len(nrow(patient_summary))  # placeholder

  train_ids <- patient_summary[[id_col]][train_index]
  test_ids  <- setdiff(patient_summary[[id_col]], train_ids)

  # 3) Map ID splits back to observation-level data
  train_df <- df[df[[id_col]] %in% train_ids, , drop = FALSE]
  test_df  <- df[df[[id_col]] %in% test_ids,  , drop = FALSE]

  return(list(train = train_df, test = test_df))
}

split_temporal <- function(df, params,
                                    id_col = "ID",
                                    time_col = "admission_year",   # or admission_date (see note below)
                                    cutoff = 2014) {               # example; keep consistent with manuscript
  # Purpose:
  #   Temporal validation split at the patient (ID) level to avoid leakage.
  #   Patients are assigned based on their FIRST observed admission time.
  #   All rows (episodes/observations) for that patient follow the same assignment.
  #
  # Logic (as used in our analysis):
  #   1) Compute first admission time per patient
  #   2) Assign patient to train vs temporal test based on the cutoff
  #   3) Join the split label back to the full observation-level table

  # 1) First observed admission time per patient
  patient_first <- df |>
    dplyr::group_by(.data[[id_col]]) |>
    dplyr::summarise(first_time = min(.data[[time_col]], na.rm = TRUE),
                     .groups = "drop")

  # 2) Assign split using the first admission time
  patient_split <- patient_first |>
    dplyr::mutate(split = ifelse(first_time < cutoff, "train_temporal", "test_temporal"))

  # 3) Map assignment back to all observations/episodes
  df_labeled <- df |>
    dplyr::left_join(patient_split |>
                       dplyr::select(.data[[id_col]], split),
                     by = id_col)

  train_temporal <- df_labeled |> dplyr::filter(split == "train_temporal") |> dplyr::select(-split)
  test_temporal  <- df_labeled |> dplyr::filter(split == "test_temporal")  |> dplyr::select(-split)

  return(list(train_temporal = train_temporal, test_temporal = test_temporal))
}

# ---- 4) Hyperparameter tuning (CV inside training only) ----
tune_xgb_cv <- function(train_df, params) {
  # Perform k-fold CV to select hyperparameters for XGBoost
  # Grid search or random search over parameter space
 return(best_params)
}

# ---- 5) Model fitting ----
fit_lr <- function(train_df, 
                   outcome_col = "ALC",
                   exclude_cols = c("ID")) {
  # Fit a logistic regression model using all predictors except ID or other exclusions.
  # Input:
  #   train_df: observation-level dataframe with outcome + predictors
  #   outcome_col: name of binary outcome column
  #   exclude_cols: columns to exclude from model formula (IDs, dates, etc.)
  #
  # Output:
  #   glm object (family = binomial, link = logit)
  
  # Identify predictor columns (all except outcome and exclusions)
  pred_cols <- setdiff(names(train_df), c(outcome_col, exclude_cols))
  
  # Build formula dynamically
  formula_str <- paste(outcome_col, "~", paste(pred_cols, collapse = " + "))
  formula_obj <- as.formula(formula_str)
  
  # Fit logistic regression
  fit <- glm(formula_obj, 
             family = binomial(link = "logit"),
             data = train_df)
  
  # Summary of model fit (optional; for diagnostics)
  cat("\n--- Logistic Regression Fit Summary ---\n")
  cat("Formula:", deparse(formula(fit)), "\n")
  cat("N observations:", nrow(train_df), "\n")
  cat("N predictors:", length(pred_cols), "\n\n")
  
  return(fit)
}

fit_xgb <- function(train_df, best_params,
                    outcome_col = "ALC",
                    exclude_cols = c("ID")) {
  # Fit XGBoost using caret::train() with pre-tuned hyperparameters
  # Input:
  #   train_df: observation-level dataframe with outcome + predictors
  #   best_params: tuning result from caret (or params list to pass to trainControl)
  #   outcome_col: name of binary outcome column
  #   exclude_cols: columns to exclude from model formula
  #
  # Output:
  #   caret train object
  
  # Identify predictor columns
  pred_cols <- setdiff(names(train_df), c(outcome_col, exclude_cols))
  
  # Prepare X and y
  X <- train_df[, pred_cols, drop = FALSE]
  y <- as.factor(train_df[[outcome_col]])
  
  # Fit using caret::train with best_params
  fit <- caret::train(
    x = X,
    y = y,
    method = "xgbTree",
    tuneGrid = best_params,
    trControl = caret::trainControl(method = "none")
  )
  
  cat("\n--- XGBoost Fit Summary ---\n")
  cat("N observations:", nrow(train_df), "\n")
  cat("N predictors:", length(pred_cols), "\n\n")
  
  return(fit)
}

predict_prob <- function(fit, new_df, model_type = c("LR", "XGB")) {
    # Return a numeric vector of predicted probabilities based on the model type
    if (model_type == "LR") {
        # For logistic regression (glm), use the predict function with type = "response"
        return(predict(fit, newdata = new_df, type = "response"))
    } else if (model_type == "XGB") {
        # For XGBoost, use the predict function directly on the new data
        return(predict(fit, newdata = new_df))
    } else {
        stop("Invalid model type specified. Use 'LR' or 'XGB'.")
    }
}

# ---- 6) Calibration (Platt; global or group-stratified) ----
platt_fit <- function(p, y, eps = 1e-6) {
    # Fit a Platt scaling model to recalibrate predicted probabilities.
    #
    # Inputs:
    #   p   : numeric vector of predicted probabilities from the base model
    #   y   : binary outcome (0/1)
    #   eps : small value to avoid infinite logits when p is 0 or 1
    #
    # Method:
    #   - clip p to (eps, 1-eps) to avoid numerical issues
    #   - fit logistic regression: y ~ logit(p)
    #
    # Output:
    #   A fitted glm object mapping raw p -> calibrated p

    # Validate inputs
    if (length(p) != length(y)) {
        stop("Length of p and y must match.")
    }
    if (!all(y %in% c(0, 1))) {
        stop("y must contain only 0 and 1.")
    }
    if (any(is.na(p)) || any(is.na(y))) {
        stop("p and y must not contain missing values.")
    }

    # Clip probabilities to avoid infinite logits
    p_clipped <- pmax(eps, pmin(1 - eps, p))

    # Fit logistic regression on logit-transformed p
    platt_model <- glm(y ~ qlogis(p_clipped),
                                         family = binomial(link = "logit"))

    return(platt_model)
}

platt_apply <- function(platt_model, p, eps = 1e-6) {
    # Apply a fitted Platt model to produce calibrated probabilities.
    #
    # Inputs:
    #   platt_model : glm object from platt_fit() mapping logit(p) -> calibrated p
    #   p           : numeric vector of raw predicted probabilities [0, 1]
    #   eps         : small value to avoid infinite logits (same as in platt_fit)
    #
    # Output:
    #   numeric vector of calibrated probabilities in [0, 1]
    #
    # Method:
    #   - clip p to (eps, 1-eps)
    #   - apply platt_model to logit-transformed p
    #   - return predicted probabilities from the calibrated model

    # Validate inputs
    if (!is.numeric(p) || any(is.na(p))) {
        stop("p must be a numeric vector without missing values.")
    }
    if (!inherits(platt_model, "glm")) {
        stop("platt_model must be a glm object from platt_fit().")
    }

    # Clip probabilities to avoid infinite logits
    p_clipped <- pmax(eps, pmin(1 - eps, p))

    # Apply Platt model to get calibrated probabilities
    p_calibrated <- predict(platt_model, 
                                          newdata = data.frame(platt_coef = qlogis(p_clipped)),
                                          type = "response")

    return(p_calibrated)
}

platt_fit_group <- function(p, y, group) {
    # Group-stratified Platt scaling:
    # Fit separate Platt models within each subgroup (e.g., sex, residency, instability).
    # Each group receives its own calibration mapping: logit(p) → calibrated p.
    #
    # Inputs:
    #   p     : numeric vector of predicted probabilities from base model
    #   y     : binary outcome (0/1)
    #   group : factor/character vector of group labels (same length as p and y)
    #
    # Output:
    #   List of fitted glm objects, one per unique group value
    #   Names correspond to group levels for easy lookup during application
        
    # Validate inputs
    if (length(p) != length(y) || length(p) != length(group)) {
        stop("Lengths of p, y, and group must match.")
    }
    if (!all(y %in% c(0, 1))) {
        stop("y must contain only 0 and 1.")
    }
    
    # Create data frame for easier grouping
    df_platt <- data.frame(p = p, y = y, group = group, stringsAsFactors = FALSE)
    
    # Get unique groups and summary statistics
    groups <- unique(df_platt$group)
    n_groups <- length(groups)
    
    cat("\n--- Group-Stratified Platt Scaling Summary ---\n")
    cat("Number of groups:", n_groups, "\n")
    cat("Groups:", paste(groups, collapse = ", "), "\n\n")
    
    # Fit Platt model within each group
    group_models <- list()
    
    for (grp in groups) {
        df_grp <- df_platt[df_platt$group == grp, ]
        n_grp <- nrow(df_grp)
        n_events <- sum(df_grp$y)
        prev_grp <- round(100 * n_events / n_grp, 2)
        
        cat(sprintf("  Group '%s': n=%d, events=%d (%.2f%%)\n", 
                                grp, n_grp, n_events, prev_grp))
        
        # Fit Platt model for this group
        group_models[[grp]] <- platt_fit(df_grp$p, df_grp$y)
    }
    
    cat("\n")
    
    return(group_models)
}

# ---- 7) Evaluation (discrimination + calibration +clinical utility) ----
eval_metrics <- function(p, y) {
    # Compute discrimination and calibration metrics.
    #
    # Inputs:
    #   p : numeric vector of predicted probabilities [0, 1]
    #   y : binary outcome (0/1)
    #
    # Outputs:
    #   Data frame with:
    #     - AUROC: Area Under the Receiver Operating Characteristic Curve
    #     - Brier: Brier Score = mean((p - y)^2)
    #     - ECE: Expected Calibration Error (binned approach)
    #     - Cal_Intercept: Calibration intercept (should be ~0 if well-calibrated)
    #     - Cal_Slope: Calibration slope (should be ~1 if well-calibrated)
    
    # Validate inputs
    if (length(p) != length(y)) {
        stop("Length of p and y must match.")
    }
    if (!all(y %in% c(0, 1))) {
        stop("y must contain only 0 and 1.")
    }
    
    # 1) AUROC: Use pROC package
    roc_obj <- pROC::roc(y, p, quiet = TRUE)
    auroc <- as.numeric(roc_obj$auc)
    
    # 2) Brier Score: mean((p - y)^2)
    brier <- mean((p - y)^2)
    
    # 3) ECE (Expected Calibration Error)
    ece <- ece_score(y, p, n_bins = n_bins)

    # 4) Calibration Intercept and Slope
    #    Fit: y ~ logit(p) using logistic regression
    #    Intercept should be ~0, Slope should be ~1
    p_clipped <- pmax(1e-6, pmin(1 - 1e-6, p))
    cal_model <- glm(y ~ qlogis(p_clipped), 
                                     family = binomial(link = "logit"))
    cal_intercept <- coef(cal_model)[1]
    cal_slope <- coef(cal_model)[2]
    
    return(data.frame(
        AUROC = auroc,
        Brier = brier,
        ECE = ece,
        Cal_Intercept = cal_intercept,
        Cal_Slope = cal_slope
    ))
}

# ---- Clinical utility: Decision Curve Analysis (DCA) ----
eval_dca <- function(df, outcome_col, pred_cols,
                    thresholds = seq(0, 0.5, by = 0.01),
                    bootstraps = 25,
                    study_design = "cohort") {
  # Decision Curve Analysis comparing multiple prediction models.
  #
  # Inputs:
  #   df          : evaluation dataset (test or temporal cohort)
  #   outcome_col : binary outcome variable
  #   pred_cols   : named character vector of prediction columns
  #                 e.g., c(XGB = "test_pred", LR = "test_pred_lr")
  #
  # Output:
  #   Named list of decision_curve objects (one per model)

  dca_models <- list()

  for (m in names(pred_cols)) {

    pred_col <- pred_cols[m]

    dca_models[[m]] <- rmda::decision_curve(
      formula = stats::as.formula(paste(outcome_col, "~", pred_col)),
      data = df,
      thresholds = thresholds,
      confidence.intervals = TRUE,
      study.design = study_design,
      bootstraps = bootstraps
    )
  }

  return(dca_models)
}


# ---- 8) Fairness (threshold-free parity) ----
eval_fairness_parity <- function(y, p, group, privileged_value, n_bins = n_bins) {
  # Compute threshold-free fairness metrics for a single protected attribute.
  #
  # Inputs:
  #   y                : binary outcome (0/1)
  #   p                : predicted probabilities
  #   group            : protected attribute (binary)
  #   privileged_value : reference group value
  #   n_bins           : number of bins for ECE
  #
  # Metrics returned as protected / privileged ratios:
  #   - AUC parity  (higher is worse for protected if >1)
  #   - ECE parity  (lower is better; >1 means protected worse)
  #   - xAUC parity (directional ranking fairness)

  # Split data by group
  idx_priv <- group == privileged_value
  idx_prot <- !idx_priv

  # --- Discrimination (AUC parity) ---
  auc_priv <- compute_auc(y[idx_priv], p[idx_priv])
  auc_prot <- compute_auc(y[idx_prot], p[idx_prot])
  auc_parity <- auc_prot / auc_priv

  # --- Calibration (ECE parity) ---
  ece_priv <- compute_ece(y[idx_priv], p[idx_priv], n_bins)
  ece_prot <- compute_ece(y[idx_prot], p[idx_prot], n_bins)
  ece_parity <- ece_prot / ece_priv

  # --- Ranking fairness (directional xAUC) ---
  xauc_prot_priv <- compute_xauc(y[idx_prot], p[idx_prot],
                                 y[idx_priv], p[idx_priv])
  xauc_priv_prot <- compute_xauc(y[idx_priv], p[idx_priv],
                                 y[idx_prot], p[idx_prot])
  xauc_parity <- xauc_prot_priv / xauc_priv_prot

  data.frame(
    AUC_Parity  = auc_parity,
    ECE_Parity  = ece_parity,
    xAUC_Parity = xauc_parity,
    Protected   = unique(group[idx_prot]),
    Privileged  = privileged_value
  )
}

# ---- 9) Explainability (local → aggregated summaries) ----
explain_local <- function(fit, df_subset, B = 100) {
  # Generate local explanations and aggregate them for PCA clustering.
  #
  # Methods:
  #   - SHAP values 
  #   - Breakdown 
  #
  # Inputs:
  #   fit        : trained model wrapped in an explainer object
  #   df_subset  : subset of observations to explain (e.g., high-risk cases)
  #   B          : number of permutations for SHAP approximation
  #
  # Output:
  #   Feature-level summary table with contribution, direction, and frequency
  #   (used downstream for PCA-based clustering)

  # --- SHAP (local explanations) ---
  shap_values <- DALEX::predict_parts(
    fit,
    new_observation = df_subset,
    type = "shap",
    B = B
  )

  # --- Aggregate SHAP across observations ---
  shap_summary <- shap_values |>
    dplyr::group_by(variable) |>
    dplyr::summarise(
      mean_positive = mean(contribution[contribution > 0], na.rm = TRUE),
      mean_negative = mean(contribution[contribution < 0], na.rm = TRUE),
      frequency = dplyr::n(),
      .groups = "drop"
    )

  # --- Breakdown explanations (used for local case inspection) ---
  breakdown_summary <- DALEX::predict_parts(
    fit,
    new_observation = df_subset,
    type = "break_down"
  )

  return(list(
    shap_summary = shap_summary,
    breakdown = breakdown_summary
  ))
}

# --- Global explainability (feature importance + PDP) ----

explain_global <- function(explainer, new_data, top_k = 20, pdp_vars = NULL) {
  # Global explainability using DALEX:
  #   (1) Permutation feature importance (variable_importance)
  #   (2) Partial dependence profiles (model_profile)
  #
  # Inputs:
  #   explainer : DALEX explainer object (e.g., explain_xgb)
  #   new_data  : evaluation data used to compute global explanations (e.g., test set)
  #   top_k     : number of top features to retain
  #   pdp_vars  : optional character vector of variables for PDP;
  #              if NULL, compute PDP for top_k important features
  #
  # Output:
  #   list(feature_importance = ..., pdp_profiles = ...)

  # 1) Permutation feature importance
  fi <- DALEX::model_parts(
    explainer,
    new_data = new_data,
    type = "variable_importance"
  )

  # Keep top-k features (consistent with your code)
  top_features <- fi |>
    dplyr::arrange(dplyr::desc(dropout_loss)) |>
    dplyr::slice_head(n = top_k) |>
    dplyr::pull(variable)

  # 2) Partial dependence profiles (PDP)
  # NOTE: PDPs summarize average marginal effects; can be sensitive to correlated predictors.
  vars_for_pdp <- if (is.null(pdp_vars)) top_features else pdp_vars

  pdp <- DALEX::model_profile(
    explainer,
    variables = vars_for_pdp,
    new_data = new_data,
    N = NULL  # use all rows (or set N for subsampling in restricted environments)
  )

  return(list(
    feature_importance = fi,
    top_features = top_features,
    pdp_profiles = pdp
  ))
}

# ---- 10) PCA clustering over explanation summaries ----
pca_cluster_features <- function(feature_summary_df,
                                 id_col = "ShortName",
                                 feature_col = "Feature",
                                 category_col = "Category",
                                 cols_for_pca = c("Contribution", "Rank", "Percentage"),
                                 scale_inputs = TRUE,
                                 verbose = FALSE) {
  # Input:
  #   feature_summary_df: per-feature summary table containing at minimum:
  #     - ShortName (feature short label)
  #     - Contribution, Rank, Percentage (or other summary dimensions)
  #     - Category (predefined group) and optionally Feature (full name)
  #
  # Output:
  #   List containing:
  #     - pca_table: data frame with PC1/PC2 coordinates, quadrant assignment
  #     - pca_object: fitted prcomp object (for variance explained, loadings)
  #     - audit_log: audit trail with data dimensions, scaling, and validation

  # --- Validation ---
  if (!all(c(id_col, cols_for_pca) %in% names(feature_summary_df))) {
    stop("Required columns not found in feature_summary_df.")
  }

  n_features <- nrow(feature_summary_df)
  if (verbose) cat(sprintf("PCA Clustering: %d features, %d dimensions\n",
                           n_features, length(cols_for_pca)))

  # 1) Subset to PCA dimensions and set rownames by feature short name
  X <- feature_summary_df |>
    dplyr::select(dplyr::all_of(c(id_col, cols_for_pca))) |>
    tibble::column_to_rownames(id_col)

  # 2) Scale inputs and log scaling decision
  if (scale_inputs) {
    X <- scale(X)
    if (verbose) cat("Input scaling: applied (mean=0, sd=1)\n")
  } else if (verbose) {
    cat("Input scaling: not applied (raw values used)\n")
  }

  # 3) PCA and capture variance explained
  pca <- prcomp(X)
  var_exp <- (pca$sdev^2) / sum(pca$sdev^2)
  pc1_var <- round(100 * var_exp[1], 2)
  pc2_var <- round(100 * var_exp[2], 2)
  if (verbose) cat(sprintf("Variance explained: PC1=%.2f%%, PC2=%.2f%%\n", pc1_var, pc2_var))

  # 4) Create PC table (PC1/PC2) + quadrant assignment
  pca_table <- as.data.frame(pca$x[, 1:2, drop = FALSE]) |>
    tibble::rownames_to_column(id_col) |>
    dplyr::rename(PC1 = PC1, PC2 = PC2)

  pca_table <- pca_table |>
    dplyr::mutate(
      Quadrant = dplyr::case_when(
        PC1 >= 0 & PC2 >= 0 ~ "Q1",
        PC1 <  0 & PC2 >= 0 ~ "Q2",
        PC1 <  0 & PC2 <  0 ~ "Q3",
        PC1 >= 0 & PC2 <  0 ~ "Q4"
      )
    )

  # 5) Join feature metadata (full name + predefined category) for auditability
  pca_table <- pca_table |>
    dplyr::left_join(
      feature_summary_df |>
        dplyr::select(dplyr::all_of(c(feature_col, id_col, category_col))),
      by = id_col
    ) |>
    dplyr::arrange(Quadrant, dplyr::desc(abs(PC1) + abs(PC2)))

  # 6) Audit log
  audit_log <- list(
    n_features = n_features,
    n_dimensions = length(cols_for_pca),
    dimensions_used = cols_for_pca,
    scaled = scale_inputs,
    var_pc1_pct = pc1_var,
    var_pc2_pct = pc2_var,
    timestamp = Sys.time(),
    loadings = pca$rotation[, 1:2]
  )

  return(list(
    pca_table = pca_table,
    pca_object = pca,
    audit_log = audit_log
  ))
}

# ---- 11) Main pipeline ----
run_pipeline <- function(params) {
    # Load and prepare data
    df <- load_ices_analytic_dataset()
    df_model <- prepare_analysis_data(df, params)
    df_clean <- handle_missingness(df_model)
    
    # Random split workflow
    spl_random <- split_random(df_clean, params)
    
    # Temporal split workflow
    spl_temporal <- split_temporal(df_clean, params, 
                                                                    id_col = "ID",
                                                                    time_col = "admission_year",
                                                                    cutoff = as.numeric(params$temporal$train_end))
    
    # Tuning + fit on random train set
    best_xgb <- tune_xgb_cv(spl_random$train, params)
    fit_lr_obj  <- fit_lr(spl_random$train, outcome_col = params$outcome_name)
    fit_xgb_obj <- fit_xgb(spl_random$train, best_xgb, outcome_col = params$outcome_name)
    
    # Predict + evaluate on random test set
    p_lr_test  <- predict_prob(fit_lr_obj,  spl_random$test, "LR")
    p_xgb_test <- predict_prob(fit_xgb_obj, spl_random$test, "XGB")
    
    metrics_lr_test  <- eval_metrics(p_lr_test,  spl_random$test[[params$outcome_name]])
    metrics_xgb_test <- eval_metrics(p_xgb_test, spl_random$test[[params$outcome_name]])
    
    # Predict on temporal test set
    p_lr_temporal  <- predict_prob(fit_lr_obj,  spl_temporal$test_temporal, "LR")
    p_xgb_temporal <- predict_prob(fit_xgb_obj, spl_temporal$test_temporal, "XGB")
    
    metrics_lr_temporal  <- eval_metrics(p_lr_temporal,  spl_temporal$test_temporal[[params$outcome_name]])
    metrics_xgb_temporal <- eval_metrics(p_xgb_temporal, spl_temporal$test_temporal[[params$outcome_name]])
    
    # Global calibration (Platt scaling on random test set)
    platt_lr  <- platt_fit(p_lr_test,  spl_random$test[[params$outcome_name]])
    platt_xgb <- platt_fit(p_xgb_test, spl_random$test[[params$outcome_name]])
    
    p_lr_cal  <- platt_apply(platt_lr,  p_lr_test)
    p_xgb_cal <- platt_apply(platt_xgb, p_xgb_test)
    
    metrics_lr_cal  <- eval_metrics(p_lr_cal,  spl_random$test[[params$outcome_name]])
    metrics_xgb_cal <- eval_metrics(p_xgb_cal, spl_random$test[[params$outcome_name]])
    
    # Group-stratified calibration
    for (grp_attr in params$calibration$groups) {
        platt_lr_grp  <- platt_fit_group(p_lr_test,  spl_random$test[[params$outcome_name]], 
                                                                            spl_random$test[[grp_attr]])
        platt_xgb_grp <- platt_fit_group(p_xgb_test, spl_random$test[[params$outcome_name]], 
                                                                            spl_random$test[[grp_attr]])
    }
    
    # Fairness evaluation (threshold-free parity)
    fairness_results <- list()
    for (grp_attr in params$fairness_groups) {
        fairness_results[[grp_attr]] <- eval_fairness_parity(
            y = spl_random$test[[params$outcome_name]],
            p = p_xgb_cal,
            group = spl_random$test[[grp_attr]],
            privileged_value = unique(spl_random$test[[grp_attr]])[1]
        )
    }
    
    # Decision Curve Analysis (DCA)
    dca_random_test <- eval_dca(
        df = spl_random$test,
        outcome_col = params$outcome_name,
        pred_cols = c(LR = "p_lr", XGB = "p_xgb"),
        thresholds = seq(0, 0.5, by = 0.01),
        bootstraps = 25,
        study_design = "cohort"
    )
    
    dca_temporal_test <- eval_dca(
        df = spl_temporal$test_temporal,
        outcome_col = params$outcome_name,
        pred_cols = c(LR = "p_lr_temporal", XGB = "p_xgb_temporal"),
        thresholds = seq(0, 0.5, by = 0.01),
        bootstraps = 25,
        study_design = "cohort"
    )
    
    # Explainability (local and global)
    explain_results <- list()
    
    if (params$explainability$local) {
        explain_results$local <- explain_local(fit_xgb_obj, spl_random$test)
    }
    
    if (params$explainability$global) {
        explain_results$global <- explain_global(fit_xgb_obj, spl_random$test, top_k = 20)
    }
    
    # PCA clustering over feature importance summaries (placeholder)
    # feature_summary_df <- aggregate feature explanations (local/global)
    # pca_results <- pca_cluster_features(feature_summary_df)
    
    cat("\n=== Pipeline Complete ===\n")
    cat("Random test metrics (LR):", "\n"); print(metrics_lr_test)
    cat("Random test metrics (XGB):", "\n"); print(metrics_xgb_test)
    cat("Temporal test metrics (LR):", "\n"); print(metrics_lr_temporal)
    cat("Temporal test metrics (XGB):", "\n"); print(metrics_xgb_temporal)
    
    invisible(list(
        models = list(fit_lr = fit_lr_obj, fit_xgb = fit_xgb_obj),
        calibration = list(platt_lr = platt_lr, platt_xgb = platt_xgb),
        metrics = list(
            random_test = list(lr = metrics_lr_test, xgb = metrics_xgb_test),
            temporal_test = list(lr = metrics_lr_temporal, xgb = metrics_xgb_temporal),
            calibrated = list(lr = metrics_lr_cal, xgb = metrics_xgb_cal)
        ),
        fairness = fairness_results,
        dca = list(random_test = dca_random_test, temporal_test = dca_temporal_test),
        explainability = explain_results
    ))
}

run_pipeline(params)

