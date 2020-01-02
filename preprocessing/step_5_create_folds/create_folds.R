
library(tidyverse)
library(groupdata2)
library(doParallel)

path <- ""
data <- read.csv(paste0(path, "preprocessed_for_tf.csv"), stringsAsFactors = FALSE) %>%
  dplyr::as_tibble() %>% 
  dplyr::mutate(Unique.ID = as.factor(Unique.ID),
                is_control = as.integer(Diagnosis == "Control"))

registerDoParallel(5)

set.seed(1)

data %>%
  dplyr::count(Diagnosis) 

# Balance the classes

upsampled <- balance(data, size="max", cat_col="Diagnosis", 
                     id_col = "Unique.ID", id_method = "distributed")
upsampled %>%
  dplyr::count(Diagnosis)

# N ids
upsampled %>%
  dplyr::count(Diagnosis, Unique.ID) %>% 
  dplyr::count(Diagnosis)

downsampled <- balance(data, size="min", cat_col="Diagnosis", 
                     id_col = "Unique.ID", id_method = "distributed")
downsampled %>%
  dplyr::count(Diagnosis)

# N ids
downsampled %>%
  dplyr::count(Diagnosis, Unique.ID) %>% 
  dplyr::count(Diagnosis)
  
iscontrol_downsampled <- balance(data, size="min", cat_col="is_control", 
                               id_col = "Unique.ID", id_method = "distributed")

iscontrol_downsampled %>%
  dplyr::count(Diagnosis)

# N ids
iscontrol_downsampled %>%
  dplyr::count(Diagnosis, Unique.ID) %>% 
  dplyr::count(Diagnosis)

create_folds <- function(data, k=10, num_fold_cols=20, parallel=TRUE, seed=1, extreme_pairing_levels=2){
  
  set.seed(seed)
  
  folded_data <- data %>%
    groupdata2::fold(
      k = 10,
      cat_col = "Diagnosis",
      id_col = "Unique.ID",
      num_col = "Num.Chars.Split",
      extreme_pairing_levels = extreme_pairing_levels,
      num_fold_cols = 20,
      parallel = TRUE
    )
  
  folded_data_for_stats <- folded_data %>%
    dplyr::select(-c(Transcript, Transcript.Split, X, is_control)) %>%
    tidyr::gather(key="fold_col", value = "fold", 7:26)
  
  fold_stats <- folded_data_for_stats %>%
    dplyr::group_by(fold_col, fold) %>%
    dplyr::summarize(avg_num_chars = mean(Num.Chars.Split),
                     sum_num_chars = sum(Num.Chars.Split),
                     n_transcripts = dplyr::n(),
                     n_IDs = length(unique(Unique.ID))
    )
  
  diagnosis_stats <- folded_data_for_stats %>%
    dplyr::group_by(fold_col, fold) %>%
    dplyr::count(Diagnosis)
  
  diagnosis_stats_deviations <- diagnosis_stats %>%
    dplyr::group_by(fold_col, Diagnosis) %>%
    dplyr::summarize(std_n = sd(n)) %>%
    dplyr::group_by(fold_col) %>%
    dplyr::summarize(sum_std_n_diagnosis = sum(std_n))
  
  stats_deviations <- fold_stats %>%
    dplyr::group_by(fold_col) %>%
    dplyr::summarize(std_avg_num_chars = sd(avg_num_chars),
                     std_sum_num_chars = sd(sum_num_chars),
                     std_n_transcripts = sd(n_transcripts),
                     std_n_ids = sd(n_IDs)
    ) %>%
    dplyr::left_join(diagnosis_stats_deviations, by="fold_col")
  
  
  # Check the relevant stats
  stats_deviations %>%
    dplyr::arrange(std_n_transcripts, 
                   sum_std_n_diagnosis, 
                   std_sum_num_chars) %>% 
    print()
  
  # I want all the stats to be as low as possible, as we want the folds
  # to be as similar as possible
  # The most important standard deviation stats are of
  # 1) the number of transcripts
  # 2) the number of rows per diagnosis
  # 3) the number of characters
  # 4) the number of IDs (which are so close that it doesn't matter)
  # Based on this, I choose .folds_11 for the unsampled dataset
  # and .folds_19 for the upsampled dataset
  
  folded_data
  
}

folded_data <- suppressMessages(create_folds(data, k=10, num_fold_cols=20, parallel=TRUE, seed=1))
folded_upsampled_data <- suppressMessages(create_folds(upsampled, k=10, num_fold_cols=20, parallel=TRUE, seed=1))
folded_downsampled_data <- suppressMessages(create_folds(downsampled, k=10, num_fold_cols=20, parallel=TRUE, seed=1, 
                                                         extreme_pairing_levels=1))
folded_iscontrol_downsampled_data <- suppressMessages(create_folds(iscontrol_downsampled, k=10, num_fold_cols=20, parallel=TRUE, seed=1, 
                                                         extreme_pairing_levels=1))

extract_relevant_cols <- function(folded_data, fold_col=".folds_11"){
  data_to_save <- folded_data[,c("Unique.ID", "Diagnosis", "Observation.ID", "Split.ID",
                                 "Transcript", "Num.Chars","Transcript.Split", "Num.Chars.Split",
                                 fold_col)]
  data_to_save[["Fold"]] <- data_to_save[[fold_col]]
  data_to_save[[fold_col]] <- NULL
  data_to_save
}

data_to_save <- extract_relevant_cols(folded_data, fold_col=".folds_11")
upsampled_data_to_save <- extract_relevant_cols(folded_upsampled_data, fold_col=".folds_19")
downsampled_data_to_save <- extract_relevant_cols(folded_downsampled_data, fold_col=".folds_10")
iscontrol_downsampled_data_to_save <- extract_relevant_cols(folded_iscontrol_downsampled_data, fold_col=".folds_1")

write_csv(data_to_save, paste0(path, "grouped_for_tf.csv") )
write_csv(upsampled_data_to_save, paste0(path, "upsampled_grouped_for_tf.csv"))
write_csv(downsampled_data_to_save, paste0(path, "downsampled_grouped_for_tf.csv"))
write_csv(iscontrol_downsampled_data_to_save, paste0(path, "iscontrol_downsampled_grouped_for_tf.csv"))


folded_data %>% dplyr::group_by(.folds_11) %>%
  dplyr::summarise(num_rows = dplyr::n())

folded_upsampled_data %>% dplyr::group_by(.folds_19) %>%
  dplyr::summarise(num_rows = dplyr::n())
# min is 1477

folded_downsampled_data %>% dplyr::group_by(.folds_10) %>%
  dplyr::summarise(num_rows = dplyr::n()) 
# min is 216

folded_iscontrol_downsampled_data %>% dplyr::group_by(.folds_1) %>%
  dplyr::summarise(num_rows = dplyr::n()) 
# min is 587
