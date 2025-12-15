fit_irt_mirt <- function(input_csv,
                    out_abilities_csv = "abilities.csv",
                    out_items_csv    = "item_params.csv",
                    id_cols          = c("participant_id", "AccountId"),
                    n_factors        = 1,
                    itemtype         = "3PL",
                    method           = "EM",
                    verbose          = FALSE,
                    mirt_args        = list(),
                    technical        = list()) {

  if (!requireNamespace("mirt", quietly = TRUE)) {
    stop("Package 'mirt' is required but not installed. Install with install.packages('mirt').")
  }

  # --- Load data ---
  dat <- utils::read.csv(input_csv, check.names = FALSE, stringsAsFactors = FALSE)

  # --- Detect and separate person IDs (AccountId or participant_id) ---
  id_cols <- c("AccountId", "participant_id")  # priority order
  id_col  <- NULL

  for (nm in id_cols) {
    if (nm %in% names(dat)) {
      id_col <- nm
      break
    }
  }

  if (!is.null(id_col)) {
    person_ids <- as.character(dat[[id_col]])
    dat[[id_col]] <- NULL  # remove ID column before fitting
  } else {
    person_ids <- paste0("P", seq_len(nrow(dat)))
    warning("⚠️ No AccountId or participant_id column found; using generated fallback IDs")
  }


  if (ncol(dat) == 0L) stop("No item columns found after removing ID column(s).")

  # Ensure numeric 0/1/NA matrix for mirt
  dat[] <- lapply(dat, function(x) suppressWarnings(as.numeric(x)))
  dat_mat <- as.data.frame(dat)

  # --- Fit IRT model ---
  mirt_call_args <- c(
    list(
      data      = dat_mat,
      model     = n_factors,
      itemtype  = itemtype,
      method    = method,
      verbose   = verbose,
      technical = technical
    ),
    mirt_args
  )

  fit <- do.call(mirt::mirt, mirt_call_args)

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

fit_irt_safe <- function(...) {
  tryCatch(
    {
      res <- fit_irt_mirt(...)   # call the original
      res$success <- TRUE
      res$reason  <- "ok"
      res
    },
    error = function(e) {
      warning("fit_irt_mirt failed: ", conditionMessage(e))
      list(
        success      = FALSE,
        reason       = "mirt_error",
        error        = e,
        abilities_df = NULL,
        items_df     = NULL,
        fit          = NULL
      )
    }
  )
}

# =======================
# CLI entry point
# =======================
if (sys.nframe() == 0) {  # only run when called as a script
  args <- commandArgs(trailingOnly = TRUE)

  get_arg <- function(flag, default = NULL) {
    if (flag %in% args) {
      i <- which(args == flag)
      if (i < length(args)) args[[i + 1]] else default
    } else {
      default
    }
  }

  input_csv     <- get_arg("--input")
  out_abilities <- get_arg("--out_abilities", "abilities.csv")
  out_items     <- get_arg("--out_items", "item_params.csv")
  n_factors     <- as.integer(get_arg("--factors", "1"))
  itemtype_arg  <- get_arg("--itemtype", "3PL")
  method_arg    <- get_arg("--method", "EM")
  verbose_arg   <- "--verbose" %in% args

  # Optional EM cycle control from CLI
  max_em_arg    <- get_arg("--max_cycles", NA)
  if (!is.na(max_em_arg)) {
    max_em <- as.integer(max_em_arg)
  } else {
    max_em <- NA_integer_
  }

  if (is.null(input_csv) || is.na(input_csv) || input_csv == "") {
    stop("Missing required --input argument")
  }

  message("Running fit_irt_mirt with:")
  message("  input_csv     = ", input_csv)
  message("  out_abilities = ", out_abilities)
  message("  out_items     = ", out_items)
  message("  n_factors     = ", n_factors)
  message("  itemtype      = ", itemtype_arg)
  message("  method        = ", method_arg)
  message("  verbose       = ", verbose_arg)
  message("  max_cycles    = ", ifelse(is.na(max_em), "default", max_em))

  technical <- list()
  if (!is.na(max_em)) {
    # ✅ this is the correct mirt technical name
    technical$NCYCLES <- max_em
  }

  res <- fit_irt_mirt(
    input_csv         = input_csv,
    out_abilities_csv = out_abilities,
    out_items_csv     = out_items,
    n_factors         = n_factors,
    itemtype          = itemtype_arg,
    method            = method_arg,
    verbose           = verbose_arg,
    technical         = technical
  )

  message("Done. Files written.")
}

