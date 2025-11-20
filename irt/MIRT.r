#' Fit a unidimensional 3PL IRT model with mirt and save outputs
#'
#' @param input_csv        Path to a wide CSV (rows = persons, cols = items; 0/1; NA allowed).
#'                         May include an ID column (e.g., "participant_id" or "AccountId").
#' @param out_abilities_csv Path to write the abilities (EAP thetas). Default "abilities.csv".
#' @param out_items_csv     Path to write item parameters (a, b, c). Default "item_params.csv".
#' @param id_cols          Candidate ID column names (first present will be used).
#' @param n_factors        Number of latent factors (default 1).
#' @param itemtype         mirt item type (default "3PL").
#' @param method           Estimation method for mirt() (default "EM").
#' @param verbose          Verbosity for mirt() (default FALSE).
#'
#' @return (invisibly) a list with elements: abilities_df, items_df, fit
#' @examples
#' fit_irt_3pl(
#'   input_csv = "data/WordMatch/mirt_in.csv",
#'   out_abilities_csv = "out/abilities.csv",
#'   out_items_csv = "out/item_params.csv"
#' )
fit_irt <- function(input_csv,
                        out_abilities_csv = "abilities.csv",
                        out_items_csv    = "item_params.csv",
                        id_cols          = c("participant_id", "AccountId"),
                        n_factors        = 1,
                        itemtype         = "2PL",
                        method           = "EM",
                        verbose          = TRUE) {
  if (!requireNamespace("mirt", quietly = TRUE)) {
    stop("Package 'mirt' is required but not installed. Install with install.packages('mirt').")
  }

  # --- Load data ---
  dat <- utils::read.csv(input_csv, check.names = FALSE, stringsAsFactors = FALSE)

  # Detect and separate person IDs (first matching column name wins)
  id_col <- NULL
  for (nm in id_cols) {
    if (nm %in% names(dat)) { id_col <- nm; break }
  }

  if (!is.null(id_col)) {
    person_ids <- as.character(dat[[id_col]])
    dat[[id_col]] <- NULL
  } else {
    person_ids <- paste0("P", seq_len(nrow(dat)))
  }

  if (ncol(dat) == 0L) stop("No item columns found after removing ID column(s).")

  # Ensure numeric 0/1/NA matrix for mirt
  dat[] <- lapply(dat, function(x) suppressWarnings(as.numeric(x)))
  dat_mat <- as.data.frame(dat)
  # Ensure numeric 0/1/NA matrix for mirt
  dat[]   <- lapply(dat, function(x) suppressWarnings(as.numeric(x)))
  dat_mat <- as.data.frame(dat)

  # --- INSERT THIS BLOCK (auto-skip items with no variability) ---
  # Items with < 2 distinct non-missing values are all-0 or all-1 (or all-NA),
  # which mirt cannot estimate.
  distinct_nonmissing <- vapply(dat_mat, function(x) length(unique(stats::na.omit(x))), integer(1))
  invariant_items <- names(distinct_nonmissing[distinct_nonmissing < 2])

  if (length(invariant_items)) {
    message("⚠️ Dropping invariant items (all 0s or all 1s): ",
            paste(invariant_items, collapse = ", "))
    dat_mat[invariant_items] <- NULL
  }

  if (ncol(dat_mat) == 0L) {
    stop("All items were invariant; nothing to fit.")
  }
  # --- Fit IRT model ---
  fit <- mirt::mirt(dat_mat,
                    model   = n_factors,
                    itemtype = itemtype,
                    method   = method,
                    verbose  = verbose)

  # --- Extract item parameters (IRT parameterization) ---
  co <- mirt::coef(fit, IRTpars = TRUE, simplify = TRUE)
  if (is.null(co$items)) stop("Could not extract item parameters from mirt::coef().")

  it <- co$items
  cn <- colnames(it)

  # Discrimination a (or a1)
  if ("a" %in% cn) {
    a <- it[, "a"]
  } else if ("a1" %in% cn) {
    a <- it[, "a1"]
  } else stop("No 'a' or 'a1' column found in item coefficients.")

  # Difficulty b; fall back from d (where b = -d/a in IRT metric)
  if ("b" %in% cn) {
    b <- it[, "b"]
  } else if ("d" %in% cn) {
    b <- -it[, "d"] / a
  } else stop("No 'b' or 'd' column found in item coefficients.")

  # Guessing c (mirt uses 'g')
  if ("g" %in% cn) {
    c_param <- it[, "g"]
  } else {
    c_param <- rep(NA_real_, nrow(it))
  }

  items_df <- data.frame(
    item_id = rownames(it),
    a = as.numeric(a),
    b = as.numeric(b),
    c = as.numeric(c_param),
    row.names = NULL,
    check.names = FALSE
  )

  # --- Person abilities (EAP) ---
  theta <- as.numeric(mirt::fscores(fit, method = "EAP")[, 1])
  abilities_df <- data.frame(
    participant_id = person_ids,
    theta = theta,
    row.names = NULL,
    check.names = FALSE
  )

  # --- Write outputs ---
  utils::write.csv(abilities_df, out_abilities_csv, row.names = FALSE)
  utils::write.csv(items_df, out_items_csv, row.names = FALSE)

  message(sprintf("✅ Wrote %s and %s (%s, %d factor%s)",
                  out_abilities_csv, out_items_csv, itemtype,
                  n_factors, ifelse(n_factors == 1, "", "s")))

  invisible(list(abilities_df = abilities_df, items_df = items_df, fit = fit))
}
# (your fit_irt() function definition above)

# ====== run the model ======
in_csv  <- "/Users/du25016/Documents/GitHub/Psychometrics---Autumn-2025/data/quickcalc/mirt_quickcalc.csv"
out_dir <- "/Users/du25016/Documents/GitHub/Psychometrics---Autumn-2025/out"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

abilities_csv <- file.path(out_dir, "abilities.csv")
items_csv     <- file.path(out_dir, "item_params.csv")

ret <- fit_irt(
  input_csv         = in_csv,
  out_abilities_csv = abilities_csv,
  out_items_csv     = items_csv,
  n_factors         = 1,
  itemtype          = "2PL",
  method            = "EM",
  verbose           = FALSE
)

# ====== plots ======
library(mirt)
library(ggplot2)
library(ggmirt)
# create the plot objects first (lattice objects)
p_icc  <- plot(ret$fit, type = "trace")      # ICCs
p_iifs <- plot(ret$fit, type = "infotrace")  # Item information (overlay)
p_test <- plot(ret$fit, type = "info")       # Test information
# Get item params in IRT metric
co <- mirt::coef(ret$fit, IRTpars = TRUE, simplify = TRUE)
stopifnot(!is.null(co$items))
it <- co$items
cn <- colnames(it)

a <- if ("a" %in% cn) it[, "a"] else it[, "a1"]
b <- if ("b" %in% cn) it[, "b"] else -it[, "d"] / a
c <- if ("g" %in% cn) it[, "g"] else rep(0, nrow(it))

params <- data.frame(item = rownames(it), a = as.numeric(a), b = as.numeric(b), c = as.numeric(c))

# Theta grid (match your desired range)
theta <- seq(-6, 6, length.out = 601)

# Build ICC dataframe: P(theta) = c + (1-c)*logistic(a*(theta-b))
icc_list <- lapply(seq_len(nrow(params)), function(i) {
  ai <- params$a[i]; bi <- params$b[i]; ci <- params$c[i]
  L  <- 1 / (1 + exp(-ai * (theta - bi)))
  P  <- ci + (1 - ci) * L
  data.frame(item = params$item[i], theta = theta, P = P)
})
icc_df <- do.call(rbind, icc_list)

# ====== plots ======
library(mirt)
library(ggplot2)
library(ggmirt)   # <-- needed for tracePlot(), itemInfoPlot(), testInfoPlot()

# LATTICE PLOTS FROM mirt (multi-panel trace, IIF overlay, TIF)
p_icc_lattice  <- plot(ret$fit, type = "trace")      # ICCs (two traces per item)
p_iifs_lattice <- plot(ret$fit, type = "infotrace")  # Item information (overlay)
p_tif_lattice  <- plot(ret$fit, type = "info")       # Test information

png(file.path(out_dir, "icc_grid.png"), width = 1600, height = 1200, res = 150); print(p_icc_lattice);  dev.off()
png(file.path(out_dir, "item_info.png"), width = 1400, height = 700,  res = 150); print(p_iifs_lattice); dev.off()
png(file.path(out_dir, "test_info.png"), width = 1400, height = 700,  res = 150); print(p_tif_lattice);  dev.off()

# YOUR MANUAL GGPlot OVERLAY (one curve per item, uses item_params you built earlier)
p_manual <- ggplot(icc_df, aes(theta, P, color = item)) +
  geom_line(linewidth = 1) +
  geom_hline(yintercept = 0.5) +
  coord_cartesian(xlim = c(-6, 6), ylim = c(0, 1)) +
  labs(title = "Probability Tracelines", x = expression(theta), y = expression(P(theta))) +
  theme_minimal(base_size = 14)
ggsave(file.path(out_dir, "icc_overlay.png"), p_manual, width = 14, height = 8, dpi = 150)

# GGMIRT VERSIONS (ggplot-native, simple and pretty)
p_icc_ggmirt <- tracePlot(
  model = ret$fit, items = NULL, facet = FALSE, theta_range = c(-6, 6),
  title = "Item Characteristic Curves (overlay)"
)
ggsave(file.path(out_dir, "icc_overlay_ggmirt.png"), p_icc_ggmirt, width = 14, height = 8, dpi = 150)

p_iif_ggmirt <- itemInfoPlot(
  model = ret$fit, items = NULL, facet = FALSE, theta_range = c(-6, 6),
  title = "Item Information Functions (overlay)"
)
ggsave(file.path(out_dir, "item_info_ggmirt.png"), p_iif_ggmirt, width = 12, height = 7, dpi = 150)

p_tif_ggmirt <- testInfoPlot(
  model = ret$fit, theta_range = c(-6, 6),
  title = "Test Information Curve"
)
ggsave(file.path(out_dir, "test_info_ggmirt.png"), p_tif_ggmirt, width = 12, height = 7, dpi = 150)
# ====== end plots ======