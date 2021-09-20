### Upload predictions

library(RODBC)
library(tidyverse)

D <- "DRIVER=ODBC Driver 17 for SQL Server"
S <- "SERVER=<DB_SERVER>"
U <- "<DB_USERNAME>"
P <- Sys.getenv("dbPass")
db <- "dssg"

# CONNECT -----------------------------------------------------------------

connString <- glue::glue('{D};{S};DATABASE={db};UID={U};PWD={P}')

a <- purrr::safely(RODBC::odbcDriverConnect)(connString)

conn <- a$result

## Create uploader function because sqlSave causes a memory fault
uploader <- function(ci, mp, mt, pt){
    a <- "INSERT INTO Predictions (ComplaintId, Model_Type, Prediction, DateTime) values("
    b <- glue::glue("'{ci}', '{mp}', '{mt}', '{pt}')")
    
    ## inherit conn from environment. Not best practice, but will work
    purrr::safely(sqlQuery)(conn, paste0(a, b))
}


## Upload relevance model -----------------------------------------
relevance_model <- read.csv("/home/<USERNAME>/ComplaintModelSMA/src/models/relevance_model_predictions.csv") %>%
  select(-X) %>% unique

relevance_model <- relevance_model %>% 
  select(ComplaintId = complaint_id, Model_Type =  model_type,
    Prediction = model_prediction, DateTime = prediction_timestamp)  %>% 
    mutate(DateTime = lubridate::as_datetime(DateTime))

relevance_model %>% unname %>% pmap(uploader)

## Confirm data is uploaded
relevance_ids <- relevance_model$ComplaintId %>% 
    paste(collapse = "','") %>% paste0("'", ., "'", collapse = "")
relevance_id_returns <- sqlQuery(conn, 
    paste0("Select ComplaintId from Predictions where \"Model_Type\" = 'relevance' AND ComplaintId in (",relevance_ids,")"))

## check to ensure that all the data has been uploaded
if (length(relevance_ids %>% str_split(",") %>% unlist) != length(relevance_id_returns$ComplaintId)) 
  stop("something wrong with relevance models")



## Upload sg model -----------------------------------------
sg_model <- read.csv("/home/<USERNAME>/ComplaintModelSMA/src/models/sanction_model_predictions.csv") %>%
  select(-X) %>% unique

sg_model <- sg_model %>% 
  select(ComplaintId = complaint_id, Model_Type =  model_type,
    Prediction = model_prediction, DateTime = prediction_timestamp)  %>% 
    mutate(DateTime = lubridate::as_datetime(DateTime))

sg_model %>% unname %>% pmap(uploader)

## Confirm data is uploaded
sg_ids <- sg_model$ComplaintId %>% 
    paste(collapse = "','") %>% paste0("'", ., "'", collapse = "")
sg_id_returns <- sqlQuery(conn, 
    paste0("Select ComplaintId from Predictions where \"Model_Type\" = 'sanction' AND ComplaintId in (",sg_ids,")"))

## check to ensure that all the data has been uploaded
if (length(sg_ids%>% str_split(",") %>% unlist) != length(sg_id_returns$ComplaintId)) 
  stop("something wrong with sg models")
