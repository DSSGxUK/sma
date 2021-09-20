### Connect to the SMA db, download results, and create intermediate tables

library(RODBC)
library(tidyverse)

D <- "DRIVER=ODBC Driver 17 for SQL Server"
S <- "SERVER=<DB_SERVER>"
U <- "<DB_USERNAME>"
P <- Sys.getenv("dbPass")
db <- "dssg"

args = commandArgs(trailingOnly=TRUE)

# test if the data path is supplied
if (length(args) < 2) {
  stop("Please supply the data path and the type of run you expect", call.=FALSE)
} else if (length(args) > 2) {
  stop("Please provide only two arguments to ETLer", call.=FALSE)
}

## add trailing slash if required
data_path <-  if(substr(args[1], str_length(args[1]), str_length(args[1])) != "/"){
  args[1] <- paste0(args[1],"/")
} else {
  args[1] <- args[1]
}
  

run_type <- args[2]

# CONNECT -----------------------------------------------------------------

connString <- glue::glue('{D};{S};DATABASE={db};UID={U};PWD={P}')

a <- purrr::safely(RODBC::odbcDriverConnect)(connString)

conn <- a$result

ourTables <- sqlTables(conn) %>% filter(TABLE_TYPE == "TABLE") %>%
  pull(TABLE_NAME) %>% discard(~grepl("trace_",x = .))


# GET DATA ---------------------------------------------------------
csvSaver <- function(item){
  # browser()
  thisTable <- sqlQuery(channel = conn, 
                        query = glue::glue('select * from "{item}"'))
  
  if (item == "Resumen_Denuncia"){
    if (run_type == "filtered"){
      predictions <- sqlQuery(conn, 'select ComplaintId from "Predictions"')
      
      ## disqualify entries for which we have 2 predictions:
      doneComplaintIds <- predictions %>% group_by(ComplaintId) %>% mutate(n = n()) %>% 
        filter(n == 2) %>% pull(ComplaintId)
      
      thisTable <- thisTable %>% filter(!ComplaintId %in% doneComplaintIds)
    } else if (run_type == "partial"){
      thisTable <- sample_n(thisTable, 1000)
      # thisTable <- tail(thisTable, 10)
    } else if (run_type == "full"){
      thisTable <- thisTable
    } else {stop("Don't understand run-type... typo?")}
  }
  
  if (item == "Detalle_DenunciaColumnasExtra") 
    item = "Detalle_DenunciaColumnasExtraI"
  
  write.csv(thisTable, paste0(data_path, substr(item, 1, 31), ".csv"))
  cat(paste(item, "Got", nrow(thisTable), "!\n"))
  return(thisTable)
}

allTables <- ourTables %>% set_names %>% map(csvSaver)
RODBC::odbcClose(conn)

## BUILT JOINT TABLES -----------------------------
allTables <- allTables %>% enframe

## apply left_joins to get joint registries. See commented out entity for 
## exact joint solution, but for now using explicit solution.

## COMPLAINTS
## What is Detalle_DenunciaMateriaAmbienta??
complaints_registry <- allTables %>% filter(grepl("Denuncia", name)) %>% 
  arrange(desc(name)) %>% pull %>% 
  reduce(., left_join, by = "ComplaintId") %>% 
  mutate(District = stringi::stri_trans_general(str = District, id = "Latin-ASCII")) %>% 
  mutate(Region = stringi::stri_trans_general(str = Region, id = "Latin-ASCII"))

complaints_registry %>% 
  select(-FacilityId, -SanctionId, -InspectionId,
  -Latitude, -Longitude, -District, -Region) %>% unique %>% 
  write.csv(paste0(data_path, "processed_r/complaints_registry.csv"))
cat("complaints_registry done!\n")

## SANCTIONS
sanctions_registry <- allTables %>% filter(grepl("Sancion", name)) %>% 
  arrange(desc(name)) %>% pull %>% 
  reduce(., left_join, by = "SanctionId")

sanctions_registry %>% 
  select(-FacilityId, -ComplaintId, -InspectionId) %>% unique %>%
  write.csv(paste0(data_path, "processed_r/sanctions_registry.csv"))
cat("sanctions_registry done!\n")

## FACILITIES SANCTION ID
facilities_registry <- allTables %>% 
  filter(name %in% "Resumen_UnidadFiscalizable") %>% 
  pull %>% pluck(1) %>% 
  left_join(allTables %>% 
    filter(name == "Detalle_ProcesoSancionUnidadFiscalizable") %>% 
  pull %>% pluck(1), by = "FacilityId") %>% 
  mutate(FacilityRegion = stringi::stri_trans_general(str = FacilityRegion, id = "Latin-ASCII"))

facilities_registry %>% 
  write.csv(paste0(data_path, "processed_r/facilities_sanction_id.csv"))
cat("facilities_registry done!\n")

## COMPLAINTS FACILITIES
complaints_facilities_registry <- complaints_registry %>% 
  left_join(facilities_registry %>% select(-SanctionId), by = "FacilityId") %>%
  left_join(allTables %>% filter(name == "Variables_territoriales") %>%
    pull %>% pluck(1), on = "District") %>%
  select(-Health_Impact, -Affected_population, -SanctionId, -InspectionId, 
    -EnvironmentalTopic, -Number, -ComplaintDetail, 
    -ComplaintAnalysis, -Latitude, -Longitude, -Region, 
    -Effect_on_Environment, -Distance_from_event, -Frequency_of_event, 
    -Day_of_event, -Time_of_event) %>% unique %>% 
  mutate(FacilityRegion = stringi::stri_trans_general(str = FacilityRegion, id = "Latin-ASCII")) %>% 
  mutate(FacilityDistrict = stringi::stri_trans_general(str = FacilityDistrict, id = "Latin-ASCII")) %>% 
  mutate(District = stringi::stri_trans_general(str = District, id = "Latin-ASCII"))

complaints_facilities_registry %>% 
  write.csv(paste0(data_path, "processed_r/complaints_facilities_registry.csv"))
cat("complaints_facilities_registry done!\n")

## COMPLAINTS SANCTIONS
complaints_sanctions_registry <- complaints_registry %>% 
  select(ComplaintId, ComplaintStatus, ComplaintType, DateComplaint, 
  DateResponse, EndType, DigitalComplaintId, PreClasification, 
  Clasification, SanctionId) %>% unique %>%
  left_join(sanctions_registry %>% 
  select(SanctionId, SanctionType, SanctionStatus, DateBegining, 
  DateEnding, HasComplianceProgram, MonetaryPenalty) %>% unique, 
  by = "SanctionId")

complaints_sanctions_registry  %>% unique %>% 
  write.csv(paste0(data_path, "processed_r/complaints_sanctions_registry.csv"))
cat("complaints_sanctions_registry done!\n")

# ## CHECK OUTPUTS:
# ## View existing fields (from Collins) to double-check:
# check_list <- list.files('~/data/processed/', full.names = T, pattern = ".csv") %>% 
#   set_names  %>% map(read.csv) %>% map(names) %>% enframe

# my_list <- list.files('~/data/processed_r/', full.names = T, pattern = ".csv") %>% 
#   set_names  %>% map(read.csv) %>% map(names) %>% enframe

# ## ensure that every field in my the data_preprocess script exists in mine.
# bind_rows(check_list %>% mutate(val = 1) %>% 
#             separate(name, c("path","file"), sep = "//"), 
#           my_list %>% mutate(val = 1) %>% 
#             separate(name, c("path","file"), sep = "//")) %>% unnest(value) %>%
#   mutate(path = gsub(".+/", "", path)) %>%
#   pivot_wider(id_cols = c(file, value), names_from = path, values_from = val) %>% 
#   mutate(diff = processed == processed_r) %>% filter(!diff | is.na(diff))

## Clear predictions --------------
purrr::safely(system)("rm ./src/models/relevance_model_predictions.csv")
purrr::safely(system)("rm ./src/models/sanction_model_predictions.csv")
