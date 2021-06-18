#+ setup, message = FALSE, warning = FALSE, error = FALSE

# Training models ----------------------------------------------------------


# Packages ----------------------------------------------------------------



pacman::p_load(tidyverse, caret, ModelMetrics, vtreat, 
               selectiveInference, glmnet, ranger, vcd, pdp)



# theme -------------------------------------------------------------------

theme_set(theme_bw() + theme(panel.grid = element_blank()))


# Data --------------------------------------------------------------------



train <- read.csv("training_noSession.csv", na.strings = c("NA", "0")) %>% 
  as_tibble() %>% 
  mutate(Educación = as.factor(Educación),
         Ocupación = as.factor(Ocupación), 
         Edo_civil = as.factor(Edo_civil),
         Hijos = replace_na(Hijos, 0)) %>% 
  dplyr::select(EXP., everything())


test <- read.csv("testing_noSession.csv", na.strings = c("NA", "0")) %>% 
  as_tibble() %>% 
  mutate(Educación = as.factor(Educación),
         Ocupación = as.factor(Ocupación), 
         Edo_civil = as.factor(Edo_civil),
         Hijos = replace_na(Hijos, 0)) %>% 
  dplyr::select(EXP., everything())



# Preprocessing -----------------------------------------------------------


treat_plan <- designTreatmentsZ(
  dframe = train,
  varlist = colnames(train) %>%  .[. != c("EXP.", "GPO.") ],
  codeRestriction = c("clean", "lev"),
  missingness_imputation = median
)


tr_treated <- prepare(treat_plan, dframe = train)
te_treated <- prepare(treat_plan, dframe = test)

# To train both models using the exact same folds
set.seed(42)
my_folds <- createFolds(tr_treated$BEST_TOTAL_POS, k = 10)



# Control parameters ------------------------------------------------------

control <- trainControl(method = "cv", 
                        number = 10, 
                        verboseIter = T,
                        index = my_folds)


# Random Forest -----------------------------------------------------------

# ncol(tr_treated) - 1

mtry_param <- seq(2, to = 30, length.out = 6) %>% round()

rf <- train(
  BEST_TOTAL_POS ~ .,
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


lasso <- train(
  BEST_TOTAL_POS ~ .,
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

res <- resamples(models, number = 100)
summary(res)

diff <- diff(res)
summary(diff)

splom(res, metric = "RMSE")
dotplot(res, metric = c("MAE", "RMSE"))
densityplot(res, metric = "MAE")
bwplot(res, metric = c("MAE", "RMSE"))

#data_metrics <- res$values %>% 
#  dplyr::select(-c(`lasso~Rsquared`, `rf~Rsquared`)) %>% 
#  pivot_longer(
#    cols = 2:last_col(),
#    names_to = "var",
#    values_to = "Score"
#  ) %>% 
#  separate(var, into = c("Model", "Metric"), sep = "~") %>% 
#  mutate(Variable = "BEST")



#write_csv(data_metrics, "data_metrics.csv")

# Testing models ----------------------------------------------------------

lasso %>% 
  predict(te_treated) %>% 
  RMSE(te_treated$BEST_TOTAL_POS)

rf %>% 
  predict(te_treated) %>% 
  RMSE(te_treated$BEST_TOTAL_POS)


lasso %>% 
  predict(te_treated) %>% 
  MAE(te_treated$BEST_TOTAL_POS)

rf %>% 
  predict(te_treated) %>% 
  MAE(te_treated$BEST_TOTAL_POS)


# Median absolute error ---------------------------------------------------
#?MLmetrics::MedianAE()

lasso %>% 
  predict(te_treated) %>% 
  MLmetrics::MedianAE(te_treated$BEST_TOTAL_POS)


rf %>% 
  predict(te_treated) %>% 
  MLmetrics::MedianAE(te_treated$BEST_TOTAL_POS)


# Coeficients -------------------------------------------------------------

coef(lasso$finalModel, s = lasso$bestTune$lambda) %>% 
  broom::tidy() %>% 
  mutate(value = round(value, 3)) %>% 
  arrange(desc(value))



# Comparing predictions ---------------------------------------------------


prediction_df <- tibble(
  Pred_lasso = predict(lasso, te_treated),
  Pred_rf = predict(rf, te_treated),
  Actual = te_treated$BEST_TOTAL_POS 
) %>% 
  pivot_longer(
    cols = Pred_lasso:Actual,
    names_to = "Condition",
    values_to = "BEST"
  ) 

prediction_df %>% 
  group_by(Condition) %>% 
  summarise(m = mean(BEST), 
            med = median(BEST), 
            sd = sd(BEST))
  

prediction_df %>% 
  #filter(BEST > 0) %>% 
  ggplot(aes(x = Condition, y = BEST)) +
  stat_summary(fun.data = mean_cl_normal)  +
  geom_jitter(width = 0.1, alpha = 0.2)




## Partial Dependence Plots ------------------------------------------------
#
#
## pdp lasso ---------------------------------------------------------------
#
#
#p1 <- partial(lasso, pred.var = "MF_PRE35_", ice = T, 
#              plot = TRUE, rug = FALSE, alpha = 0.2, plot.engine = "ggplot2", 
#              train = tr_treated) +
#  labs(y = "Predicted BEST")
#
#p2 <- partial(lasso, pred.var = "WQL_PRE22", ice = T,  
#              plot = TRUE, rug = FALSE, alpha = 0.2, plot.engine = "ggplot2",
#              train = tr_treated) +
#  labs(y = NULL)
#
#
#p3 <- partial(lasso, pred.var = c("MF_PRE35_", "SCID_102"),
#              plot = TRUE, chull = TRUE, plot.engine = "ggplot2", 
#              train = tr_treated) 

# Figure 2
#grid.arrange(p1, p2, ncol = 2, top = "lasso")


## pdp rf ------------------------------------------------------------------
#
#  
#f1 <- partial(rf, pred.var = "EOSS_Sec1_PRE", ice = TRUE, 
#              plot = TRUE, rug = FALSE, alpha = 0.1, plot.engine = "ggplot2", 
#              train = tr_treated) +
#  labs(y = "Predicted BEST")
#
#f2 <- partial(rf, pred.var = "CAA_PRE6", ice = TRUE,  
#              plot = TRUE, rug = FALSE, alpha = 0.1, plot.engine = "ggplot2",
#              train = tr_treated) +
#  labs(y = NULL)
#  
#
#grid.arrange(f1, f2, ncol = 2, top = "Random Forest")
  


imp_rf <- varImp(rf, n = 10)$importance %>% 
  rownames_to_column() %>% 
  arrange(desc(Overall)) %>% 
  head(10) %>% 
  pull(rowname)


partial_Said <- function(var = "") {
  partial(rf, pred.var = var, ice = F,  
          plot = TRUE, rug = T, 
          plot.engine = "ggplot2", 
          train = tr_treated) +
    labs(y = "Predicted BEST") 
}


map(imp_rf, partial_Said)

# Pre vs post -------------------------------------------------------------

train %>%
  pivot_longer(
    cols = c(BEST_TOTAL_PRE, BEST_TOTAL_POS), 
    names_to = "Tiempo",
    values_to = "BEST_TOTAL"
  ) %>% 
  dplyr::select(Tiempo, BEST_TOTAL) %>% 
  mutate(Tiempo = factor(Tiempo, levels = c("BEST_TOTAL_PRE", "BEST_TOTAL_POS"))) %>% 
  ggplot(aes(x = Tiempo, y = BEST_TOTAL))  +
  geom_boxplot(width = .3, position = position_nudge(x = -0.2, y = 0)) +
  geom_dotplot(binaxis = "y", alpha = 0.5, dotsize = 0.7, binwidth = 2)



#data %>% 
#  dplyr::select(EXP., BEST_TOTAL_PRE, BEST_TOTAL_POS, 
#         BIS_TOTAL_PRE, BIS_TOTAL_POS) %>% 
#  pivot_longer(
#    cols = BEST_TOTAL_PRE:last_col(),
#    names_to = "Variable",
#    values_to = "Score"
#  ) %>% 
#  mutate(
#    Variable = factor(Variable, 
#                      levels = c("BEST_TOTAL_PRE", "BEST_TOTAL_POS",
#                                 "BIS_TOTAL_PRE", "BIS_TOTAL_POS"),
#                      labels = c("BEST/PRE", "BEST/POST", "BIS/PRE", "BIS/POST"))
#  ) %>% 
#  separate(col = Variable, into = c("Scale", "Time")) %>% 
#  mutate(Time = factor(Time, levels = c("PRE", "POST"))) %>% 
#  filter(!is.na(Score), Score > 0) %>% 
#  ggplot(aes(x = Time, y = Score)) +
#  geom_violin(aes(fill = Time)) +
#  #geom_boxplot(width = .1) +
#  #geom_boxplot(show.legend = F) +
#  #geom_dotplot(aes(fill = Time), 
#  #             binaxis = "y", 
#  #             stackdir = "center",
#  #             binwidth = 7, dotsize = 0.4, ) +
#  #geom_jitter(aes(col = Time), alpha = 0.7, width = 0.5) +
#  stat_summary(fun.data = mean_cl_normal, 
#               geom = "errorbar", width = 0.3) + 
#  scale_fill_manual(values = c("#DAA520", "#1C86EE")) +
#  scale_color_manual(values = c("#DAA520", "#1C86EE")) +
#  facet_wrap(~Scale)



# Importance pvalues rf ---------------------------------------------------
set.seed(12)
imp_rf_pvalues <- importance_pvalues(rf$finalModel, 
                                     method  = "altmann", 
                                     formula = BEST_TOTAL_POS ~ .,
                                     data = tr_treated,
                                     num.permutations = 500)

imp_rf_pvalues %>% 
  data.frame() %>% 
  rownames_to_column() %>% 
  arrange(desc(importance)) %>% 
  filter(pvalue < 0.05) 

# Compare models in testing data set using bootstrap ----------------------


set.seed(156)
bm_rf <- caret::train(BEST_TOTAL_POS ~ .,
               data = te_treated,
               method = "ranger", 
               importance =  "impurity",
               tuneGrid = expand.grid(
                 .mtry = rf$bestTune$mtry,
                 .min.node.size = rf$bestTune$min.node.size,
                 .splitrule = rf$bestTune$splitrule
               )
              )

bm_lasso <- train(
  BEST_TOTAL_POS ~ .,
  data = te_treated,
  method = "glmnet",
  #trControl = control,
  tuneGrid = expand.grid(
    .alpha = lasso$bestTune$alpha,
    .lambda = lasso$bestTune$lambda)
)



rmse_models <- tibble(
  rf = bm_rf$resample[, 1],
  lasso = bm_lasso$resample[, 1]) %>% 
  pivot_longer(
    cols = 1:last_col(), 
    names_to = "Model",
    values_to = "RMSE"
  ) %>% 
  arrange(Model)


t.test(RMSE ~ Model, data = rmse_models)
  

