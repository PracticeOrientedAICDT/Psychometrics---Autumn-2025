library(purrr)
library(dplyr)
library(stringr)

N_vals <- c(10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000)


sim_dir <- "/Users/tt25013/Documents/GitHub/Psychometrics---Autumn-2025/data/experimental/60_items"

# get all csvs in folder
csv_paths <- list.files(sim_dir, pattern = "IRTMatrix\\.csv$", full.names = TRUE)

# keep only those where the FIRST number in filename matches one of your N values
csv_paths <- csv_paths[ sapply(basename(csv_paths), function(x) as.numeric(str_extract(x, "^[0-9]+")) %in% N_vals) ]

results <- map(csv_paths, function(path) {
  # parse N from filename if you encoded it there, e.g. "1000_IRTMatrix.csv"
  # this is optional and depends on your naming scheme
  fname <- basename(path)
  N     <- as.numeric(str_extract(fname, "^[0-9]+"))

  out_base <- file.path(sim_dir, tools::file_path_sans_ext(fname))

  res <- fit_irt_safe(
    input_csv         = path,
    out_abilities_csv = paste0(out_base, "_abilities.csv"),
    out_items_csv     = paste0(out_base, "_item_params.csv"),
    itemtype          = "2PL",
    method            = "EM",
    mirt_args         = list(SE = TRUE, quadpts = 41),
    technical         = list(NCYCLES = 2000000)
  )

  res$N      <- N
  res$path   <- path
  res$fname  <- fname

  res
})
