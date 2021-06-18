#+ setup, message = FALSE, warning = FALSE, error = FALSE


# Partial dependence plots ------------------------------------------------


# http://uc-r.github.io/iml-pkg
# Book: https://christophm.github.io/interpretable-ml-book/


# The Partial class implements partial dependence plots (PDPs) and individual conditional expectation (ICE) curves. The procedure follows the traditional methodology documented in Friedman (2001) and Goldstein et al. (2015) where the ICE curve for a certain feature illustrates the predicted value for each observation when we force each observation to take on the unique values of that feature. The PDP curve represents the average prediction across all observations. 



# Packages ----------------------------------------------------------------


pacman::p_load(tidyverse, iml, ggpubr)


theme_set(theme_bw(base_size = 10, base_family = "Times") + theme(panel.grid = element_blank()))


# Data --------------------------------------------------------------------

# load("bpd_ml.RData")
source("repo/Severity_BPD/ml-bpd.R")


mod <- Predictor$new(
  model = rf,
  data = tr_treated,
  y = "BEST_TOTAL_POS"
)


# eff <- FeatureEffect$new(
#   mod,
#   feature = imp_rf[1],
#   method = "pdp+ice",
#   center.at = min(tr_treated$BEST_TOTAL_POS),
#   grid.size = 50
# )
# 
# eff$plot()


eff_Said <- function(predictor) {
  
  eff <- FeatureEffect$new(
    mod,
    feature = predictor,
    method = "pdp+ice",
    center.at = tr_treated %>% 
      dplyr::select(all_of(predictor)) %>% 
      min()
  )
  
  eff$plot()
}


plots_rf <- map(imp_rf, eff_Said)

plots_rf[[1]] <- plots_rf[[1]] + 
  labs(x = "General self scale (EOSS)") 


plots_rf[[2]] <- plots_rf[[2]] + 
  labs(x = "I never get compromised (AAQ)")


plots_rf[[3]] <- plots_rf[[3]] + 
  labs(x = "Paranoia scale (SCIID)")


plots_rf[[4]] <- plots_rf[[4]] + 
  labs(x = "Non-judgment scale (FFMQ)")


plots_rf[[5]] <- plots_rf[[5]] + 
  labs(x = "I am off somewhere watching myself (EOSS)")


plots_rf[[6]] <- plots_rf[[6]] + 
  labs(x = "Narcissistic relations scale (IPO)")


plots_rf[[7]] <- plots_rf[[7]] +
  labs(x = "How well are you able to get around? (WHOQOL)")


plots_rf[[8]] <- plots_rf[[8]] +
  labs(x = "I can not do anything to feel better (DERS)")


plots_rf[[9]] <- plots_rf[[9]] +
  labs(x = "Support you get from your friends (WHOQOL)") 

partial_plots <- ggpubr::ggarrange(plotlist = plots_rf[-10])

#artial_plots <- annotate_figure(partial_plots,
#                               top = text_grob("Clinical predictors of severity in BPD",
#                               family = "Times", 
#                               face = "bold",
#                               size = 15))

#ggsave(
#  "partial_plots.png", 
#  plot = partial_plots,
#  dpi = 300, 
#  device = "png", 
#  units = "cm",
#  width = 26, 
#  height = 18
#)
#

