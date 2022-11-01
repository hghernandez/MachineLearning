##%######################################################%##
#                                                          #
####                Modelo Random Forest                ####
#                                                          #
##%######################################################%##

#Ejemplo tomado de Juli Silge en https://juliasilge.com/blog/sf-trees-random-tuning/

#Objetivo: predecir que árboles son


library(tidyverse)

sf_trees <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-28/sf_trees.csv")

trees_df <- sf_trees %>%
  mutate(
    legal_status = case_when(
      legal_status == "DPW Maintained" ~ legal_status,
      TRUE ~ "Other" #Todas las no mantenidas por SF las agrupa en otras
    ),
    plot_size = parse_number(plot_size) #Trae el primer valor numerico del string
  ) %>%
  select(-address) %>%
  na.omit() %>% #No imputa datos, omite na
  mutate_if(is.character, factor)



#Armar un mapa de los puntos ----

trees_df %>%
  ggplot(aes(longitude, latitude, color = legal_status)) +
  geom_point(size = 0.5, alpha = 0.4) +
  labs(color = NULL)


# Que relaciones vemos con el cuidador de cada arbol

trees_df %>%
  count(legal_status, caretaker) %>%
  add_count(caretaker, wt = n, name = "caretaker_count") %>%
  filter(caretaker_count > 50) %>%
  group_by(legal_status) %>%
  mutate(percent_legal = n / sum(n)) %>%
  ggplot(aes(percent_legal, caretaker, fill = legal_status)) +
  geom_col(position = "dodge") +
  labs(
    fill = NULL,
    x = "% of trees in each category"
  )

View(trees_df %>%
       count(legal_status, caretaker) %>%
       add_count(caretaker, wt = n, name = "caretaker_count") %>% 
       filter(caretaker_count > 50)
)

trees_df %>%
       group_by(legal_status, caretaker)%>%
       summarise(n= n())%>%
       ungroup()%>%
       group_by(caretaker)%>%
       mutate(total= sum(n))%>%
       filter(total > 50) %>%
       group_by(legal_status) %>%
       mutate(percent_legal = n / sum(n)) %>%
       ggplot(aes(percent_legal, caretaker, fill = legal_status)) +
       geom_col(position = "dodge") +
       labs(
         fill = NULL,
         x = "% of trees in each category"
       )

##%######################################################%##
#                                                          #
####                Contruimos el modelo                ####
#                                                          #
##%######################################################%##

library(tidymodels)

#Dividimos el dataset ----

set.seed(123)

split <- initial_split(trees_df,strata = legal_status)

train <- training(split)

test <- testing(split)

summary(train$legal_status)
summary(test$legal_status)

#Construimos la receta con recipes ----

# Primero, debemos decir recipe()cuál será nuestro modelo (usando una fórmula aquí) y cuáles son nuestros datos de entrenamiento.
# Actualizamos el rol de tree_id, ya que esta es una variable que nos gustaría mantener por conveniencia como identificador de filas, 
# pero no es un predictor ni un resultado.
# usamos step_other()para colapsar los niveles categóricos de especie, cuidador e información del sitio. 
# ¡Antes de este paso, había más de 300 especies!
# La columna date con la fecha en que se plantó cada árbol puede ser útil para ajustar este modelo, pero probablemente no la fecha exacta, 
# dada la lentitud con la que crecen los árboles. Creemos una función de año a partir de la fecha y luego eliminemos la variable original.
# Hay muchos más árboles mantenidos por DPW que no, así que reduzcamos la muestra de los datos para el entrenamiento.

View(trees_df)

unique(trees_df$species)


tree_rec <- recipe(legal_status ~ ., data = train) %>%
  update_role(tree_id, new_role = "ID") %>%
  step_other(species, caretaker, threshold = 0.01) %>% #Agrupa en otras las species con % < a 0.01
  step_other(site_info, threshold = 0.005) %>%  #Agrupa en otras los sites_info con % < a 0.005
  step_dummy(all_nominal(), -all_outcomes()) %>% #One hot encoding todas las nominales
  step_date(date, features = c("year")) %>% #se queda solo con el año
  step_rm(date) #remueve año
  #themis::step_downsample(legal_status) #achica la muestra



# entrenamos el modelo ----

trees_prep <- prep(tree_rec)

# Extraigo los datos entrenados

juiced <- bake(trees_prep,new_data = NULL)

View(juiced)

##%######################################################%##
#                                                          #
####             Especificacion del modelo              ####
#                                                          #
##%######################################################%##

# Especificamos un modelo Random Forest y ajustamos
# mtry cantidad de predictores para muestrear en cada división
# min_n cantidad de observaciones necesarias para seguir dividiendo nodos

tune_spec <- rand_forest(
  mtry = tune(),
  trees = 1000,
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

##%######################################################%##
#                                                          #
####                 Creamos un worflow                 ####
#                                                          #
##%######################################################%##

tune_wf <- workflow() %>%
  add_recipe(tree_rec)%>%
  add_model(tune_spec)
  

##%######################################################%##
#                                                          #
####            Optimizacion hiperparámetros            ####
#                                                          #
##%######################################################%##

# Validacion cruzada ----

set.seed(234)

trees_folds <- vfold_cv(train)

# Entrenamos varios modelos ----

doParallel::registerDoParallel()

set.seed(345)

tune_res <- tune_grid(
  tune_wf,
  resamples = trees_folds,
  grid = 10
)


tune_res

##%######################################################%##
#                                                          #
####                 Métricas de error                  ####
#                                                          #
##%######################################################%##

tune_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, min_n, mtry) %>%
  pivot_longer(min_n:mtry,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC")


# Probamos varias combinaciones de hipeparametros

rf_grid <- grid_regular(
  mtry(range = c(10, 30)),
  min_n(range = c(2, 8)),
  levels = 5
)

rf_grid

set.seed(456)
regular_res <- tune_grid(
  tune_wf,
  resamples = trees_folds,
  grid = rf_grid
)

regular_res


# Vemos las métricas 

regular_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(mtry, mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(y = "AUC")


# Elegimos el mejor modelo

best_auc <- select_best(regular_res, "roc_auc")

final_rf <- finalize_model(
  tune_spec,
  best_auc
)

final_rf

# Vemos la importancia de los preditores en el modelo

library(vip)

final_rf %>%
  set_engine("ranger", importance = "permutation") %>%
  fit(legal_status ~ .,
      data = juice(tree_prep) %>% select(-tree_id)
  ) %>%
  vip(geom = "point")


# Flujo final del modelo ----

final_wf <- workflow() %>%
  add_recipe(tree_rec) %>%
  add_model(final_rf)

final_res <- final_wf %>%
  last_fit(trees_split)

final_res %>%
  collect_metrics()


# Unimos la prediccion con el archivo original

final_res %>%
  collect_predictions() %>%
  mutate(correct = case_when(
    legal_status == .pred_class ~ "Correct",
    TRUE ~ "Incorrect"
  )) %>%
  bind_cols(trees_test) %>%
  ggplot(aes(longitude, latitude, color = correct)) +
  geom_point(size = 0.5, alpha = 0.5) +
  labs(color = NULL) +
  scale_color_manual(values = c("gray80", "darkred"))

