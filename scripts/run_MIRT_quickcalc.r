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