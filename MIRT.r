library(tidyverse)
library(mirt)
library(ggmirt)

# Simulate some data
data <- sim_irt(500, 8, seed = 123)

# Save the simulated data to a CSV file
write.csv(data, "simulated_data.csv", row.names = FALSE)

# Read the data back from the CSV file
data <- read.csv("simulated_data.csv")

# Run IRT model with mirt
mod <- mirt(data, 1, itemtype = "2PL", verbose = FALSE)

# Plot item-person map
#itempersonMap(mod)

# Plot item information curves
#itemInfoPlot(mod, facet = T)


summary(mod)
