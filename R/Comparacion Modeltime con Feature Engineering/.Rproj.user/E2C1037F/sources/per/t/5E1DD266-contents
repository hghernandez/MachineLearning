library(modeltime)
library(tidyverse)
library(tidymodels)
library(timetk)

#Objetivo: ver las diferencias de errores con y sin features engineering

#Tomo el dataset de modeltime

df <- m4_monthly %>% filter(id == "M750") %>%
arrange(date)

#Divido el dataset

split <- initial_split(data = df,prop = 0.8)


train <- training(split)

test <- testing(split)

summary(train$date)
View(train)
summary(test$date)
View(test)


#creo la receta

df_rec <- recipe(value ~ date, data= df)%>%
  step_mutate(date_num= as.numeric(date))%>%
  step_rm(date)

df_rec2 <- recipe(value ~ date, data= df)%>%
 step_date(date, features = c("month", "quarter", "year"), ordinal = TRUE)


#Creamos el modelo

prophet_spec <-
  modeltime::prophet_reg() %>%
  set_engine('prophet')

lm_spec <-
  linear_reg() %>%
  set_engine('lm')


#Evaluamos el modelo por bootstrap

boot <- bootstraps(train)


#Creamos el workflow

model_wf <- workflow()%>%
  add_recipe(df_rec2) %>%
  add_model(prophet_spec) %>%
  fit_resamples(
    resamples= boot,
    control = control_resamples(save_pred = TRUE)
  )


model_wf %>%
  collect_metrics()


model_wf %>%
  collect_predictions()

#Creo el modelo final

model_wf_prod <- workflow()%>%
  add_recipe(df_rec2) %>%
  add_model(prophet_spec)

eval <- model_wf_prod %>%
  last_fit(split)

eval %>%
  collect_metrics()

View(eval %>%
  collect_predictions()%>%
    bind_cols(test))


model.refit <- fit(model_wf_prod,df)

pred <- data.frame(date= seq(as.Date('2015-07-01'),as.Date('2015-12-01'),"month"))

prueba <- cbind(predict(model.refit,pred))

predict(model.refit,pred,type = "numeric")

##%######################################################%##
#                                                          #
####         Vemos la proyeccion con modeltime          ####
#                                                          #
##%######################################################%##

library(tidyverse)
library(lubridate)
library(timetk)
library(parsnip)
library(rsample)
library(modeltime)

# Data
m750 <- m4_monthly %>% filter(id == "M750")

# Split Data 80/20
splits <- initial_time_split(m750, prop = 0.9)

# --- MODELS ---

# Model 1: prophet ----

model_fit_prophet <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(value ~ date, data = training(splits))

View(training(splits))
View(testing(splits))

View(train)
# ---- MODELTIME TABLE ----

models_tbl <- modeltime_table(
  model_fit_prophet
)

models_tbl$.model

# ---- CALIBRATE ----

calibration_tbl <- models_tbl %>%
  modeltime_calibrate(new_data = testing(splits))

# ---- ACCURACY ----

calibration_tbl %>%
  modeltime_accuracy()

# ---- FUTURE FORECAST ----

View(calibration_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = m750
  ))

# ---- ALTERNATIVE: FORECAST WITHOUT CONFIDENCE INTERVALS ----
# Skips Calibration Step, No Confidence Intervals

models_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = m750
  )

# ---- KEEP NEW DATA WITH FORECAST ----
# Keeps the new data. Useful if new data has information
#  like ID features that should be kept with the forecast data

calibration_tbl %>%
  modeltime_forecast(
    new_data      = testing(splits),
    keep_data     = TRUE
  )






