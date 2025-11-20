# fit_3pl.R
library(mirt)

# --- Load data (wide: rows=persons, cols=items; 0/1, NA for missing) ---
dat <- read.csv("/Users/tt25013/Documents/GitHub/Psychometrics---Autumn-2025/data/WordMatch/mirt_in.csv",
                check.names = FALSE, stringsAsFactors = FALSE)

# If your CSV includes an ID column, keep it; otherwise auto-generate
id_col <- NULL
if ("participant_id" %in% names(dat)) id_col <- "participant_id"
if (is.null(id_col) && "AccountId" %in% names(dat)) id_col <- "AccountId"

if (!is.null(id_col)) {
  person_ids <- as.character(dat[[id_col]])
  dat[[id_col]] <- NULL           # remove ID from the numeric matrix
} else {
  person_ids <- paste0("P", seq_len(nrow(dat)))
}

# Ensure numeric 0/1/NA
dat[] <- lapply(dat, function(x) suppressWarnings(as.numeric(x)))

# --- Fit 3PL (1 factor) ---
fit <- mirt(dat, 1, itemtype = "3PL", method = "EM", verbose = FALSE)

# --- Extract item params: a, b, c (mirt names guessing 'g') ---
it <- coef(fit, IRTpars = TRUE, simplify = TRUE)$items
cn <- colnames(it)

# Discrimination (a)
a <- if ("a" %in% cn) it[, "a"] else if ("a1" %in% cn) it[, "a1"] else stop("No 'a' or 'a1' in coef().")

# Difficulty (b)  (fallback from 'd' if needed)
b <- if ("b" %in% cn) it[, "b"] else if ("d" %in% cn) -it[, "d"] / a else stop("No 'b' or 'd' in coef().")

# Guessing (c)
c_param <- if ("g" %in% cn) it[, "g"] else rep(NA_real_, nrow(it))

items_df <- data.frame(
  item_id = rownames(it),
  a = as.numeric(a),
  b = as.numeric(b),
  c = as.numeric(c_param),
  row.names = NULL
)

# --- Abilities (theta via EAP) ---
theta <- fscores(fit, method = "EAP")[, 1]
abilities_df <- data.frame(
  participant_id = person_ids,              # <-- use your actual IDs
  ability = as.numeric(theta),
  row.names = NULL
)

# --- Save CSVs (written to current working directory) ---
write.csv(abilities_df, "abilities.csv", row.names = FALSE)
write.csv(items_df, "item_params.csv", row.names = FALSE)

cat("✅ Wrote abilities.csv and item_params.csv (3PL)\n")
