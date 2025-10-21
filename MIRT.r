library(tidyverse)
library(mirt)
library(ggmirt)

# Load data
data <- read.csv("irt_data.csv")
mod <- mirt(data, 1, itemtype = "2PL", verbose = FALSE)

# Plot item-person map
itempersonMap(mod)
tracePlot(mod)
tracePlot(mod, facet = F, legend = T)

coef(mod)
