#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(mirt)
  library(dplyr)
})

# ==========================
# CONFIG: edit these values
# ==========================
num_items <- 21      # number of items
R_reps    <- 1000      # Monte Carlo replications
N_min     <- 100      # minimum sample size
N_max     <- 5000     # maximum sample size
N_step    <- 100   # step size for N

# Directory to write output CSV into (will be created if it doesn't exist)
output_dir <- "/Users/tt25013/Documents/GitHub/Psychometrics---Autumn-2025/data/experimental"

# Optional: customise output filename
output_file <- file.path(
  output_dir,
  sprintf("mse_21_Gyrate_items.csv")
)

# Make sure output directory exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# ---------- helper: define hypothetical items ----------
define_hypothetic_items <- function(num_items,
                                    a_value = 1,
                                    b_min   = -2,
                                    b_max   =  2) {
  b_vals <- seq(b_min, b_max, length.out = num_items)
  a_vals <- rep(a_value, num_items)
  d_vals <- -a_vals * b_vals  # mirt uses (a, d); b = -d/a

  data.frame(
    item_id = as.character(seq_len(num_items)),
    a       = a_vals,
    b       = b_vals,
    d       = d_vals
  )
}

# ---------- helper: simulate responses ----------
simulate_responses <- function(item_df, N, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  a <- item_df$a
  d <- item_df$d

  resp <- mirt::simdata(
    a        = a,
    d        = d,
    N        = N,
    itemtype = "2PL"
  )

  as.data.frame(resp)
}

# ---------- fit IRT and return recovered b ----------
fit_irt_get_bhat <- function(resp_df) {
  mod <- mirt::mirt(
    data     = resp_df,
    model    = 1,
    itemtype = "2PL",
    method   = "EM",
    SE       = FALSE,
    verbose  = FALSE
  )

  coefs <- mirt::coef(mod, IRTpars = TRUE, simplify = TRUE)
  items <- as.data.frame(coefs$items)
  items$item_id <- rownames(coefs$items)

  if (!"b" %in% names(items) && all(c("a1", "d") %in% names(items))) {
    items$b <- -items$d / items$a1
  }

  items[, c("item_id", "b"), drop = FALSE]
}

# ---------- MSE(b) ----------
compute_mse_b <- function(true_items, rec_items) {
  # rely on the fact that item order is consistent
  n_true <- nrow(true_items)
  n_rec  <- nrow(rec_items)

  if (n_true == 0L || n_rec == 0L) return(NA_real_)
  if (n_true != n_rec) {
    warning("Number of true and recovered items differ; cannot align rows safely.")
    return(NA_real_)
  }

  b_true <- true_items$b
  b_hat  <- rec_items$b

  mean((b_hat - b_true)^2)
}

# ==========================
# MAIN MONTE CARLO LOOP
# ==========================

true_items <- define_hypothetic_items(num_items)
N_values   <- seq(N_min, N_max, by = N_step)

summary_rows <- list()

for (N in N_values) {
  message("\n============================")
  message("Running Monte-Carlo for N = ", N)

  mse_reps <- numeric(R_reps)

  for (r in seq_len(R_reps)) {
    message("  Replication ", r, "/", R_reps)
    seed <- sample.int(1e7, 1)

    resp_df <- simulate_responses(true_items, N = N, seed = seed)

    tryCatch({rec_items <- fit_irt_get_bhat(resp_df)}, error = function(msg){
      return(NA)
    })

    mse_b <- compute_mse_b(true_items, rec_items)
    mse_reps[r] <- mse_b
  }

  # aggregated error for this N
  mean_mse <- mean(mse_reps, na.rm = TRUE)
  sd_mse   <- sd(mse_reps, na.rm = TRUE)

  message("✅ Converged result for N=", N,
          ": mean MSE(b) = ", round(mean_mse, 4),
          " (sd = ", round(sd_mse, 4), ")")

  summary_rows[[length(summary_rows) + 1L]] <- data.frame(
    N          = N,
    mean_mse_b = mean_mse,
    sd_mse_b   = sd_mse,
    R_effective = sum(!is.na(mse_reps))
  )
}

res_df <- dplyr::bind_rows(summary_rows)

message("\n============================")
message("Final aggregated MSE(b) per N written to:")
message(output_file)

write.table(
  res_df,
  file      = output_file,
  sep       = ",",
  row.names = FALSE,
  col.names = TRUE,
  quote     = FALSE
)

