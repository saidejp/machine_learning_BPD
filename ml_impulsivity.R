#+ setup, message = FALSE, warning = FALSE, error = FALSE


# Machine Learning and BPD ------------------------------------------------
# Script to reproduce paper results and figures related to impulsiveness

# Data cleaning -----------------------------------------------------------



# Packages ----------------------------------------------------------------

pacman::p_load(tidyverse, readxl, caret, ModelMetrics, vtreat, 
               selectiveInference, glmnet, ranger, vcd, pdp, iml)

theme_set(theme_bw(base_size = 10, base_family = "Times") + 
            theme(panel.grid = element_blank()))



# Reading data ------------------------------------------------------------

train_set <- read_csv("training_impulsivity.csv") %>% 
  mutate(Edad = as.integer(Edad),
         Hijos = as.integer(Hijos) %>% replace_na(0),
         Sexo = factor(Sexo, levels = c(1, 2),
                       labels = c("Mujer", "Hombre")),
         Educación = as.factor(Educación),
         Ocupación = as.factor(Ocupación),
         Edo_civil = as.factor(Edo_civil)) %>% 
  mutate_at(.vars = vars(DI:last_col()), as.numeric) 





test_set <- read_csv("testing_impulsivity.csv") %>% 
  mutate(Edad = as.integer(Edad),
         Hijos = as.integer(Hijos) %>% replace_na(0),
         Sexo = factor(Sexo, levels = c(1, 2),
                       labels = c("Mujer", "Hombre")),
         Educación = as.factor(Educación),
         Ocupación = as.factor(Ocupación),
         Edo_civil = as.factor(Edo_civil)) %>% 
  mutate_at(.vars = vars(DI:last_col()), as.numeric) 



# https://cran.r-project.org/web/packages/vtreat/vignettes/vtreatVariableTypes.html

treat_plan <- designTreatmentsZ(
  dframe = train_set,
  varlist = colnames(train_set) %>%  .[. != c("EXP.", "GPO.") ],
  codeRestriction = c("clean", "lev"),
  missingness_imputation = median
)


tr_treated <- prepare(treat_plan, dframe = train_set)
te_treated <- prepare(treat_plan, dframe = test_set)


# To train both models with the exact same folds
set.seed(42)
my_folds <- createFolds(tr_treated$BIS_TOTAL_POS, k = 10)



# Control parameters ------------------------------------------------------

control <- trainControl(method = "cv", 
                        number = 10,
                        verboseIter = T,
                        index = my_folds)



# Random Forest -----------------------------------------------------------

# ncol(tr_treated) - 1

mtry_param <- seq(2, to =  ncol(tr_treated) - 1, length.out = 6) %>% round()

rf <- train(
  BIS_TOTAL_POS ~ .,
  data = tr_treated,
  method = "ranger",
  trControl = control,
  importance = "impurity",
  tuneGrid = expand.grid(
    .mtry = mtry_param,
    .min.node.size = 5,
    .splitrule = c("extratrees", "variance")
  )
)

plot(rf)
plot(varImp(rf), top = 10)


# Lasso -------------------------------------------------------------------
set.seed(12)

lasso <- train(
  BIS_TOTAL_POS ~ .,
  data = tr_treated,
  method = "glmnet",
  trControl = control,
  tuneGrid = expand.grid(
    .alpha = 1,
    .lambda = seq(0.01, 3, length = 10))
)

plot(lasso)
plot(lasso$finalModel, xvar = "lambda", label = T)
plot(varImp(lasso), top = 10)




# Compare models ----------------------------------------------------------

models <- list(lasso = lasso, rf = rf)

res <- resamples(models)
summary(res)

diff <- diff(res)
summary(diff)

splom(res, metric = "RMSE")
dotplot(res, metric = "RMSE")
densityplot(res, metric = "RMSE")
bwplot(res,  metric = "RMSE")



# Testing models ----------------------------------------------------------

lasso %>% 
  predict(te_treated) %>% 
  RMSE(te_treated$BIS_TOTAL_POS)

rf %>% 
  predict(te_treated) %>% 
  RMSE(te_treated$BIS_TOTAL_POS)

# Median absolute error ---------------------------------------------------
#?MLmetrics::MedianAE()

lasso %>% 
  predict(te_treated) %>% 
  MLmetrics::MedianAE(te_treated$BIS_TOTAL_POS)

rf %>% 
  predict(te_treated) %>% 
  MLmetrics::MedianAE(te_treated$BIS_TOTAL_POS)


# Coeficients -------------------------------------------------------------

coef(lasso$finalModel, s = lasso$bestTune$lambda) %>% 
  broom::tidy() %>% 
  mutate(value = round(value, 3)) %>% 
  arrange(desc(value))



# Comparing predictions ---------------------------------------------------


prediction_df <- tibble(
  Pred_lasso = predict(lasso, te_treated),
  Pred_rf = predict(rf, te_treated),
  Actual = te_treated$BIS_TOTAL_POS
) %>% 
  pivot_longer(
    cols = Pred_lasso:Actual,
    names_to = "Condition",
    values_to = "BIS"
  ) 

prediction_df %>% 
  group_by(Condition) %>% 
  summarise(m = mean(BIS), 
            med = median(BIS), 
            sd = sd(BIS))


prediction_df %>% 
  #filter(BEST > 0) %>% 
  ggplot(aes(x = Condition, y = BIS)) +
  stat_summary(fun.data = "mean_cl_boot") +
  geom_jitter(width = 0.1, alpha = 0.2)



imp_rf <- varImp(rf, n = 10)$importance %>% 
  rownames_to_column() %>% 
  arrange(desc(Overall)) %>% 
  head(10) %>% 
  pull(rowname)



mod <- Predictor$new(rf, data = tr_treated, y = "BIS_TOTAL_POS") 

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

plots_rf[[1]] <- plots_rf[[1]] + labs(x = "Non-planning impulsiveness (BIS)")

plots_rf[[2]] <- plots_rf[[2]] + labs(x = "Crying (BDI)")

plots_rf[[3]] <- plots_rf[[3]] + labs(x = "Total impulsiveness (BIS)")

plots_rf[[4]] <- plots_rf[[4]] + 
  labs(x = "I would like to change many things about myself (AAQ)")

plots_rf[[5]] <- plots_rf[[5]] + 
  labs(x = "I pay attention to the impact of my actions on others’ feelings (RFQ)")


plots_rf[[6]] <- plots_rf[[6]] + 
  labs(x = "I need to prove that I am important to people (AAQ)") 

plots_rf[[7]] <- plots_rf[[7]] +
  labs(x = "Dependence scale (SCIID)")

plots_rf[[8]] <- plots_rf[[8]] +
  labs(x = "I keep the other person's point of view in mind (RFQ)")


plots_rf[[9]] <- plots_rf[[9]] +
  labs(x = "Following through with therapy plans to which you agreed (BEST)")




partial_plots <- ggpubr::ggarrange(plotlist = plots_rf[-10])

#ggsave(
#  "partial_plots_bis.png", 
#  plot = partial_plots,
#  dpi = 300, 
#  device = "png", 
#  units = "cm",
#  width = 32, 
#  height = 20
#)



# Pre vs post -------------------------------------------------------------
theme_set(theme_bw(base_size = 10, base_family = "Times") + 
            theme(panel.grid.major = element_blank()))

train_set %>%
  pivot_longer(
    cols = c(BIS_TOTAL_PRE, BIS_TOTAL_POS), 
    names_to = "Tiempo",
    values_to = "BIS_TOTAL"
  ) %>% 
  #dplyr::select(Tiempo, BIS_TOTAL) %>% 
  mutate(Tiempo = factor(Tiempo, levels = c("BIS_TOTAL_PRE", "BIS_TOTAL_POS"))) %>% 
  ggplot(aes(x = Tiempo, y = BIS_TOTAL, col = Tiempo)) +
  geom_line(aes(group = EXP.), alpha = 0.2, col = "black") +
  geom_point(alpha = 0.6) 
 # geom_boxplot(position = position_nudge(x = -0.2, y = 0))
  
 
bis_plot <- train_set %>%
  pivot_longer(
    cols = c(BIS_TOTAL_PRE, BIS_TOTAL_POS), 
    names_to = "Tiempo",
    values_to = "BIS_TOTAL"
  ) %>% 
  #dplyr::select(Tiempo, BIS_TOTAL) %>% 
  mutate(Tiempo = factor(Tiempo, 
                         levels = c("BIS_TOTAL_PRE", "BIS_TOTAL_POS"),
                         labels = c("PRE", "POST"))) %>% 
  ggplot(aes(x = Tiempo, y = BIS_TOTAL)) +
  #geom_line(aes(group = EXP.), alpha = 0.2, col = "black") +
  #geom_point(alpha = 0.6) +
  geom_boxplot(aes(fill = Tiempo), 
               width = .3, 
               position = position_nudge(x = -0.2, y = 0),
               show.legend = F) +
  geom_dotplot(aes(fill = Tiempo),
               binaxis = "y", alpha = 0.5, 
               dotsize = 0.7, binwidth = 2.5,
               show.legend = F) +
  scale_fill_manual(values = c("#DAA520", "#1C86EE")) +
  labs(
    y = "BIS Impulsivity Score",
    x = NULL,
    tag = "A"
  )

#t.test(train_set$BIS_TOTAL_PRE, train_set$BIS_TOTAL_POS, var.equal = T)

#ggsave("bis_plot.png", plot = bis_plot, device = "png", 
#       units = "cm", dpi = 350, width = 12, height = 10)


data_metrics <- res$values %>% 
  dplyr::select(-c(`lasso~Rsquared`, `rf~Rsquared`)) %>% 
  pivot_longer(
    cols = 2:last_col(),
    names_to = "var",
    values_to = "Score"
  ) %>% 
  separate(var, into = c("Model", "Metric"), sep = "~") %>% 
  mutate(Variable = "BIS")



# Importance pvalues rf ---------------------------------------------------
set.seed(12)
imp_rf_pvalues <- importance_pvalues(rf$finalModel, 
                                     method  = "altmann", 
                                     formula = BIS_TOTAL_POS ~ .,
                                     data = tr_treated,
                                     num.permutations = 500)

imp_rf_pvalues %>% 
  data.frame() %>% 
  rownames_to_column() %>% 
  arrange(desc(importance)) %>% 
  filter(pvalue < 0.05) 
#  write_csv("rf_imp_bis.csv")

