#' Fit a unidimensional (or multidimensional) IRT model with mirt and save outputs
#'
#' If itemtype == "3PL" and the initial fit fails, this function pre-fits a 2PL (EM)
#' to build stable starts (incl. g bounds) and retries 3PL.
#'
#' @param input_csv         Path to wide 0/1/NA CSV (rows=persons, cols=items). May include an ID column.
#' @param out_abilities_csv Output path for person scores (EAP/MAP/etc). 
#' @param out_items_csv     Output path for item parameters (a,b,c).
#' @param id_cols           Candidate ID column names.
#' @param n_factors         Number of latent factors (default 1).
#' @param itemtype          "2PL", "3PL", "Rasch", etc.
#' @param method            "EM", "QMCEM", or "MHRM".
#' @param verbose           Verbosity for mirt().
#' @param technical         List passed to mirt(..., technical=). DO NOT put quadpts here.
#' @param mirt_args         Extra args forwarded to mirt(); put quadpts here for EM/QMCEM.
#' @param fscores_method    "EAP" (default), "MAP", "ML", or "WLE".
#' @param fscores_args      Extra args to fscores().
#' @return (invisibly) list(abilities_df, items_df, fit)
fit_irt <- function(input_csv,
                    out_abilities_csv = "abilities.csv",
                    out_items_csv    = "item_params.csv",
                    id_cols          = c("participant_id", "AccountId"),
                    n_factors        = 1,
                    itemtype         = "3PL",
                    method           = "EM",
                    verbose          = FALSE,
                    technical        = NULL,
                    mirt_args        = list(),
                    fscores_method   = "EAP",
                    fscores_args     = list()) {

  if (!requireNamespace("mirt", quietly = TRUE)) {
    stop("Package 'mirt' is required but not installed. install.packages('mirt')")
  }

  # ---- Load data & separate optional ID ----
  dat <- utils::read.csv(input_csv, check.names = FALSE, stringsAsFactors = FALSE)

  id_col <- NULL
  for (nm in id_cols) if (nm %in% names(dat)) { id_col <- nm; break }
  if (!is.null(id_col)) {
    person_ids <- as.character(dat[[id_col]])
    dat[[id_col]] <- NULL
  } else {
    person_ids <- paste0("P", seq_len(nrow(dat)))
  }
  if (ncol(dat) == 0L) stop("No item columns found after removing ID column(s).")

  # Coerce to numeric 0/1/NA
  dat[] <- lapply(dat, function(x) suppressWarnings(as.numeric(x)))
  dat_mat <- as.data.frame(dat)

  # ---- Helpers --------------------------------------------------------------

  # sanitize 'technical' and 'mirt_args' according to method
  sanitize_controls <- function(method, technical, mirt_args) {
    methodU <- toupper(method)

    # move quadpts out of technical (if user put it there by mistake)
    if (!is.null(technical) && "quadpts" %in% names(technical)) {
      mirt_args$quadpts <- technical$quadpts
      technical$quadpts <- NULL
    }

    if (methodU %in% c("EM", "QMCEM")) {
      # allowed in technical: NCYCLES (and other EM controls); quadpts must be in mirt_args
      if (!is.null(mirt_args$quadpts) && !is.numeric(mirt_args$quadpts))
        stop("quadpts must be numeric when provided.")
      # drop MHRM-only controls if present
      drop <- intersect(names(technical %||% list()), c("MHDRAWS","BURNIN","SEMCYCLES","MHRM_SE_draws"))
      if (length(drop)) technical[drop] <- NULL
    } else if (methodU == "MHRM") {
      # MHRM ignores/complains about quadpts
      if (!is.null(mirt_args$quadpts)) {
        warning("Ignoring quadpts for method='MHRM'. Removing it from mirt_args.")
        mirt_args$quadpts <- NULL
      }
      # keep typical MHRM controls; NCYCLES, MHDRAWS, BURNIN, SEMCYCLES, MHRM_SE_draws are fine
    }
    list(technical = technical, mirt_args = mirt_args)
  }

  `%||%` <- function(a, b) if (!is.null(a)) a else b

  # 3PL starts from 2PL (EM): add g rows, set g starts/bounds, tame a-bounds
  build_3pl_starts_from_2pl <- function(dat_mat) {
    fit_2pl <- mirt::mirt(dat_mat, n_factors, itemtype = "2PL", method = "EM", verbose = FALSE)
    vals <- mirt::mod2values(fit_2pl)

    g_rows <- vals$name == "g"
    if (!any(g_rows)) {
      tmp <- mirt::mirt(dat_mat, n_factors, itemtype = "3PL", method = "EM",
                        technical = list(NCYCLES = 10), verbose = FALSE)
      vals <- mirt::mod2values(tmp)
      g_rows <- vals$name == "g"
    }

    vals$value[g_rows]  <- 0.05
    vals$est[g_rows]    <- TRUE
    vals$lbound[g_rows] <- ifelse(is.finite(vals$lbound[g_rows]), pmax(vals$lbound[g_rows], 0.001), 0.001)
    vals$ubound[g_rows] <- ifelse(is.finite(vals$ubound[g_rows]), pmin(vals$ubound[g_rows], 0.35), 0.35)

    a_rows <- vals$name %in% c("a","a1")
    vals$lbound[a_rows] <- ifelse(is.finite(vals$lbound[a_rows]), pmax(vals$lbound[a_rows], 0.20), 0.20)
    vals$ubound[a_rows] <- ifelse(is.finite(vals$ubound[a_rows]), pmin(vals$ubound[a_rows], 3.00), 3.00)
    vals$est[a_rows]    <- TRUE

    vals
  }

  # ---- Build primary call & fit --------------------------------------------

  # sanitize controls first
  sc <- sanitize_controls(method, technical, mirt_args)
  technical <- sc$technical
  mirt_args <- sc$mirt_args

  mirt_call <- c(
    list(
      data     = dat_mat,
      model    = n_factors,
      itemtype = itemtype,
      method   = method,
      verbose  = verbose
    ),
    if (!is.null(technical)) list(technical = technical) else NULL,
    mirt_args
  )

  fit <- NULL
  err_msg <- NULL
  fit <- tryCatch(
    do.call(mirt::mirt, mirt_call),
    error = function(e) { err_msg <<- conditionMessage(e); NULL }
  )

  # --- Auto-fallback for 3PL failures: 2PL->3PL starts
  if (is.null(fit) && toupper(itemtype) == "3PL") {
    message("⚠️ Initial 3PL fit failed (", err_msg, "). Trying 2PL→3PL starts...")

    vals <- build_3pl_starts_from_2pl(dat_mat)

    # if user didn't supply technical, apply safe MHRM defaults
    technical2 <- technical %||% list(
      NCYCLES   = 2000L,
      MHDRAWS   = 20L,
      BURNIN    = 200L,
      SEMCYCLES = 200L
    )

    # choose MHRM unless user explicitly asked EM/QMCEM
    method2 <- method
    if (!toupper(method) %in% c("EM","QMCEM","MHRM")) method2 <- "MHRM"

    # sanitize again for the retry method
    sc2 <- sanitize_controls(method2, technical2, mirt_args)
    technical2 <- sc2$technical
    mirt_args2 <- sc2$mirt_args

    mirt_call2 <- c(
      list(
        data     = dat_mat,
        model    = n_factors,
        itemtype = "3PL",
        method   = method2,
        verbose  = verbose,
        pars     = vals
      ),
      if (!is.null(technical2)) list(technical = technical2) else NULL,
      mirt_args2
    )

    fit <- tryCatch(
      do.call(mirt::mirt, mirt_call2),
      error = function(e) { stop("3PL fallback with 2PL starts also failed: ", conditionMessage(e)) }
    )
    message("✅ 3PL fit succeeded after seeding with 2PL starts.")
  } else if (is.null(fit)) {
    stop(err_msg)  # non-3PL failure: surface error
  }

  # ---- Extract item parameters (IRT metric) --------------------------------
  co <- mirt::coef(fit, IRTpars = TRUE, simplify = TRUE)
  if (is.null(co$items)) stop("Could not extract item parameters from mirt::coef().")
  it <- co$items
  cn <- colnames(it)

  # a / a1
  if ("a" %in% cn) {
    a <- it[, "a"]
  } else if ("a1" %in% cn) {
    a <- it[, "a1"]
  } else stop("No 'a' or 'a1' found in item coefficients.")

  # b or d (b = -d/a)
  if ("b" %in% cn) {
    b <- it[, "b"]
  } else if ("d" %in% cn) {
    b <- -it[, "d"] / a
  } else stop("No 'b' or 'd' found in item coefficients.")

  # g (guessing)
  c_param <- if ("g" %in% cn) it[, "g"] else rep(NA_real_, nrow(it))

  # use original item names if available; else fallback to colnames(dat_mat)
  item_names <- rownames(it)
  if (is.null(item_names) || any(is.na(item_names)) || any(item_names == "")) {
    item_names <- colnames(dat_mat)
  }

  items_df <- data.frame(
    item_id = item_names,
    a = as.numeric(a),
    b = as.numeric(b),
    c = as.numeric(c_param),
    row.names = NULL,
    check.names = FALSE
  )

  # ---- Person scores --------------------------------------------------------
  fs_args <- c(list(object = fit, method = fscores_method), fscores_args)
  theta_mat <- as.data.frame(do.call(mirt::fscores, fs_args))
  if (ncol(theta_mat) == 1) {
    names(theta_mat) <- "theta"
  } else {
    names(theta_mat) <- paste0("theta", seq_len(ncol(theta_mat)))
  }

  abilities_df <- data.frame(
    participant_id = person_ids,
    theta_mat,
    row.names = NULL,
    check.names = FALSE
  )

  # ---- Write outputs --------------------------------------------------------
  # directory containing mirt_in.csv
  base_dir <- dirname(input_csv)

  # modelling subfolder inside that directory
  modelling_dir <- file.path(base_dir, "modelling")

  # create modelling folder if missing
  if (!dir.exists(modelling_dir)) {
    dir.create(modelling_dir, recursive = TRUE)
  }

  # final output CSV paths
  
  base_dir <- dirname(input_csv)
  modelling_dir <- file.path(base_dir, "modelling")
  out_abilities_csv <- file.path(modelling_dir, "abilities.csv")
  out_items_csv     <- file.path(modelling_dir, "item_params.csv")
  
  utils::write.csv(abilities_df, out_abilities_csv, row.names = FALSE)
  utils::write.csv(items_df, out_items_csv, row.names = FALSE)

  message(sprintf(
  "✅ Wrote:\n  abilities → %s\n  items     → %s\n(itemtype=%s, method=%s, factors=%d)",
  normalizePath(out_abilities_csv, mustWork = FALSE),
  normalizePath(out_items_csv, mustWork = FALSE),
  itemtype,
  method,
  n_factors
  ))

  invisible(list(abilities_df = abilities_df, items_df = items_df, fit = fit))
}
