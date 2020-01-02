

## Adding unique subject IDs
library(tidyverse)
path <- ""
d <- read.csv(paste0(path, "All-Diagnoses-Adults-DK-Triangles.csv"),
              stringsAsFactors = FALSE) %>%
  dplyr::as_tibble()

d <- d %>%
  dplyr::group_by(File, Sub.File, Study, Diagnosis, Subject)

unique_IDs <- dplyr::group_keys(d) %>%
  dplyr::mutate(unique_ID = seq_len(dplyr::n()))

write.csv(unique_IDs, paste0(path, "unique_IDs.csv"))
