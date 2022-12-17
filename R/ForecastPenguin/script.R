library(tidymodels)
library(dplyr)
library(tidyverse)

#Objetivo implementar un modelo lineal y ver si obtengo las mismas proyecciones
#que las de python

data <- readr::read_csv("C:/Users/usuario/Documents/MachineLearning/python/ForecastPenguin/st_libros.csv")

#Divido el data set en train, test

train <- list()
test <- list()

for(libros in unique(data$TITLE)){

set.seed(123)  

split_data <- initial_split(data[data$TITLE == libros,],prop = 0.8)

train[[libros]] <- training(split_data)
test[[libros]] <- testing(split_data)

}



#Instancio el modelo

lm_spec <- linear_reg() %>%
  set_engine(engine = "lm")

#Ajusto el modelo

lm_fit <- list()

for(libros in unique(data$TITLE)){
  
set.seed(123) 

lm_fit[[libros]] <- lm_spec %>%
  fit(Cantidad ~ Time,
      data = train[[libros]]
)

}


#Obtengo las metricas de error

result_test <- list()

for(libros in unique(data$TITLE)){

result_test[[libros]] <- lm_fit[[libros]] %>%
  predict(new_data = test[[libros]]) %>%
  mutate(
    truth = test[[libros]]$Cantidad,
    model = "lm",
    TITLE = libros
  )
}


result_test_df <- do.call(rbind,result_test)

View(result_test_df  %>%
  group_by(model,TITLE) %>%
  rmse(truth = truth, estimate = .pred))



#Realizo las proyecciones para el mes 37

max(data$Fecha)

pred = data.frame("Fecha" = as.Date('2021-02-01'),
                  "Time" = 37)

View(data[data$TITLE== 'Libro 1',])

lm_fit$`Libro 1` %>%
  predict(new_data = pred)
