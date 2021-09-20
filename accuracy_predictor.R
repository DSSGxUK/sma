### Evaluate performance of models:

library(RODBC)
library(tidyverse)

D <- "DRIVER=ODBC Driver 17 for SQL Server"
S <- "SERVER=<DB_SERVER>"
U <- "<DB_USERNAME>"
P <- Sys.getenv("dbPass")
db <- "dssg"

connString <- glue::glue('{D};{S};DATABASE={db};UID={U};PWD={P}')
a <- purrr::safely(RODBC::odbcDriverConnect)(connString)
conn <- a$result

predictions <- sqlQuery(conn, 'select * from Predictions')
complaints <- sqlQuery(conn, 'select * from Resumen_Denuncia')

# Relevance model ---------------------------------------------------------

# rel <- predictions %>% filter(Model_Type == "relevance") %>% 
#   select(ComplaintId, Prediction)
rel <- read_csv("src/models/relevance_model_predictions.csv") %>% 
  select(ComplaintId = complaint_id, Prediction = model_prediction)
rel_real <- complaints %>% select(ComplaintId, EndType) %>% 
  filter(!is.na(EndType))

rel_both <- inner_join(rel, rel_real, by = "ComplaintId")

## combine:
rel_both <- rel_both %>% mutate(real = case_when(
  EndType == "Archivo II" | EndType == "Formulación de Cargos" ~ "Relevant",
  EndType == "Archivo I" ~ "Archivo I",
  EndType == "Derivación Total a Organismo Competente" ~ "Derivacion")) %>% 
  select(-EndType, -ComplaintId) %>% mutate_all(as.factor)

yardstick::accuracy(rel_both, Prediction, real)
yardstick::conf_mat(rel_both, Prediction, real)


# Sanction gravity --------------------------------------------------------
# sqlTables(conn) %>% View
sanctions <- sqlQuery(conn, 'select * from Resumen_ProcesoSancion')
c_x_s <- sqlQuery(conn, 'select * from Detalle_DenunciaProcesoSancion')
detailsSan <- sqlQuery(conn, 'select * from Detalle_ProcesoSancionHechoInstrumento') %>% 
  select(SanctionId, InfractionCategory)

## Develop worst sanction possible per complaint. Unite w/ complaints:
detailsSan <- detailsSan %>% left_join(c_x_s, by = "SanctionId")

san_real <- detailsSan %>% filter(!is.na(InfractionCategory)) %>% 
  mutate(InfractionCategory = substr(InfractionCategory, 1,1))  %>% 
  unique %>% mutate(a=1) %>% 
  pivot_wider(id_cols = ComplaintId, names_from = InfractionCategory, 
              values_from = a, values_fn = max) %>% 
  mutate(real = case_when(
    G == 1 ~ "high",
    L == 1 ~ "low")) %>% select(ComplaintId, real)

# san <- predictions %>% filter(Model_Type == "sanction") %>% 
#   select(ComplaintId, Prediction)

san <- read_csv("src/models/sanction_model_predictions.csv") %>% 
  select(ComplaintId = complaint_id, Prediction = model_prediction) %>% 
  mutate(ComplaintId = as.integer(ComplaintId))

san_both <- inner_join(san, san_real, by = "ComplaintId") %>% 
  select(-ComplaintId) %>% mutate_all(as.factor)

yardstick::accuracy(san_both, Prediction, real)
yardstick::conf_mat(san_both, Prediction, real)


                                 