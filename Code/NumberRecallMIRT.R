# Setup paths
data_dir <- "/Users/op24226/Desktop/PsychGames/Data"
code_dir <- "/Users/op24226/Desktop/PsychGames/Psychometrics---Autumn-2025/Code"
input_csv <- "/Users/op24226/Desktop/PsychGames/Data/NumberRecall_MIRT_Format.csv"

dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(code_dir, recursive = TRUE, showWarnings = FALSE)

# Packages
library(mirt)

# Read data
dat <- read.csv(input_csv , check.names = FALSE, stringsAsFactors = FALSE)

# Participant IDS
 id_col <- intersect(c("participant_id", "AccountId"), names(dat))[1]
if (!is.na(id_col)) {
       person_ids <- as.character(dat[[id_col]])
       dat[[id_col]] <- NULL   # remove the ID from the item matrix
   } else {
         person_ids <- paste0("P", seq_len(nrow(dat)))
   }
 
 # Numeric 0/1/NA
dat[] <- lapply(dat, function(x) suppressWarnings(as.numeric(x)))

# Pick item columns which have multiple entries only
item_cols <- grep("^item_id\\b", names(dat), value = TRUE, ignore.case = TRUE)
if (length(item_cols) == 0L) stop("No columns starting with 'item_id' were found.")
 
dat_items <- dat[item_cols]
varying <- vapply(dat_items, function(x) length(unique(na.omit(x))) > 1, logical(1))
dat_items <- dat_items[varying]
if (ncol(dat_items) < 2L) stop("Not enough varying 'item_id' columns to fit a model.")
 
# Fit 3PL model
fit <- mirt(dat_items, 1, itemtype = "3PL", method = "EM", verbose = FALSE)

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

# Abilities/ Person Scores
theta <- fscores(fit, method = "EAP")[, 1]
abilities_df <- data.frame(
       participant_id = person_ids,              # <-- use your actual IDs
       ability = as.numeric(theta),
       row.names = NULL
   )

# Save outputs
write.csv(abilities_df, file.path(data_dir, "abilities.csv"), row.names = FALSE)
write.csv(items_df,     file.path(data_dir, "item_params.csv"), row.names = FALSE)

 