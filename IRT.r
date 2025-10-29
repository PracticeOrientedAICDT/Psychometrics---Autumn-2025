library(tidyverse)
library(mirt)
library(ggmirt)

# Load data
data <- read.csv("EyeBall.csv")

# Assign column names to items in the model
item_names <- colnames(data)
mod <- mirt(data, 1, itemtype = "2PL", verbose = FALSE)

# Plot item-person map
itempersonMap(mod)
tracePlot(mod)
tracePlot(mod, facet = F)

coef(mod)
