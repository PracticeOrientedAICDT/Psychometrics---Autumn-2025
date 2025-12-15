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
                        itemtype         = "3PL",
                        method           = "EM",
                        verbose          = FALSE) {
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