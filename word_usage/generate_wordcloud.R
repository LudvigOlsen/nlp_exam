library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")
library(tidyverse)
library(ggplot2)
# library(vocabular2)

# http://www.sthda.com/english/wiki/text-mining-and-word-cloud-fundamentals-in-r-5-simple-steps-you-should-know

project_path <- ""
data <- read.csv(paste0(project_path, "data/preprocessed/imbalanced_grouped_for_tf.csv"), stringsAsFactors = FALSE)

#### Describe data ####
deduped <- data %>%
  dplyr::as_tibble() %>%
  dplyr::select(Unique.ID, Diagnosis, Observation.ID, Transcript, Num.Chars) %>%
  dplyr::distinct()

deduped %>%
  dplyr::group_by(Diagnosis, Unique.ID) %>%
  dplyr::summarise(n_trials = dplyr::n(),
                   sum_chars = sum(Num.Chars)) %>%
  dplyr::summarise(n_participants = dplyr::n(),
                   avg_n_trials = mean(n_trials),
                   avg_sum_chars = mean(sum_chars),
                   avg_trial_num_chars = mean(sum_chars/n_trials))

#### Word usage analysis ####

data <- data %>%
  dplyr::as_tibble() %>%
  dplyr::select(Diagnosis, Observation.ID, Transcript) %>%
  dplyr::distinct() %>%
  dplyr::group_by(Diagnosis) %>%
  dplyr::summarise(Text = paste0(Transcript, collapse=" ")) %>%
  dplyr::mutate(Text = str_replace_all(Text, " hianden ", " hinanden "),
                Text = str_replace_all(Text, " -agtigt ", " agtigt "),
                Text = str_replace_all(Text, "inaudible", ""),
                Text = str_replace_all(Text, "inaudble", ""))

control_text <- data[["Text"]][[1]]
depression_text <- data[["Text"]][[2]]
schizo_text <- data[["Text"]][[3]]

term_counts <- function(t){
  docs <- Corpus(VectorSource(t))
  docs <- tm_map(docs, removeWords, stopwords("danish"))
  # docs <- tm_map(docs, stemDocument)
  dtm <- TermDocumentMatrix(docs)
  m <- as.matrix(dtm)
  v <- sort(rowSums(m), decreasing=TRUE)
  d <- tibble::tibble(Word = names(v), Count=v)
  d
}

# TODO Extract conditions from colnames

depression_term_counts <- term_counts(depression_text)
control_term_counts <- term_counts(control_text)
schizo_term_counts <- term_counts(schizo_text)

vocab_uni <- compare_vocabs(list(
  "Depression" = depression_term_counts,
  "Control" = control_term_counts,
  "Schizo" = schizo_term_counts
))

doc_stats_depression <- get_doc_stats(vocab_uni, "Depression", remove_zero_counts = TRUE)
doc_stats_control <- get_doc_stats(vocab_uni, "Control", remove_zero_counts = TRUE)
doc_stats_schizo <- get_doc_stats(vocab_uni, "Schizo", remove_zero_counts = TRUE)

doc_stats_collected <- dplyr::bind_rows(doc_stats_depression,
                                        doc_stats_control,
                                        doc_stats_schizo)

# Find correlations between the metrics
metric_correlations <- cor(base_deselect(doc_stats_collected, c("Doc","Word","Count")))
metric_correlations_high <- metric_correlations
metric_correlations_high[abs(metric_correlations_high) < 0.7] <- NA
metric_correlations_low <- metric_correlations
metric_correlations_low[abs(metric_correlations_low) > 0.3] <- NA

metric_rank_correlations <- cor(base_deselect(doc_stats_collected, c("Doc","Word","Count")), method="spearman")

library(doParallel)
registerDoParallel(6)

# Test different values for the beta parameter in Relative TF-NRTF
beta_val_correlations <- function(docs){

  betas <- unique(runif(200, 0.0, 5.0))
  doc_names <- names(docs)

  plyr::ldply(betas, .parallel=TRUE, function(be){
    comp <- compare_vocabs(docs, rel_tf_nrtf_beta = be)
    comp_long <- plyr::ldply(doc_names, function(dn){
      get_doc_stats(comp, dn, remove_zero_counts = TRUE) %>%
        base_select(c("TF_RTF", "TF_NRTF", "TF_MRTF", "REL_TF_NRTF", "REL_TF_MRTF", "TF_IRF"))
    })
    pearson_correlation <- dplyr::as_tibble(cor(comp_long, method = "pearson"), rownames = "By") %>%
      dplyr::mutate("Method" = "pearson")
    kendall_correlation <- dplyr::as_tibble(cor(comp_long, method = "pearson"), rownames = "By") %>%
      dplyr::mutate("Method" = "kendall")
    spearman_correlation <- dplyr::as_tibble(cor(comp_long, method = "spearman"), rownames = "By") %>%
      dplyr::mutate("Method" = "spearman")

    dplyr::bind_rows(pearson_correlation,
                     kendall_correlation,
                     spearman_correlation) %>%
      dplyr::mutate(beta = be)
  })

}

beta_value_corrs <- beta_val_correlations(list(
  "Depression" = depression_term_counts,
  "Control" = control_term_counts,
  "Schizo" = schizo_term_counts
)) %>% dplyr::as_tibble()

betas_long <- beta_value_corrs %>%
  tidyr::gather(key="Metric", value="Correlation", 2:7) %>%
  dplyr::filter(By != Metric,
                Method != "kendall") %>%
  base_rename(before="beta",after = "Beta")

mult <- 0.8
tiff(paste0(project_path, "beta_comparisons_REL_TF_NRTF.tiff"), units="in",
     width=4*mult, height=6*mult, res=300)
betas_long %>%
  dplyr::filter(Metric == "REL_TF_NRTF") %>%
  ggplot(aes(x = Beta, y = Correlation)) +
  geom_line() +
  facet_wrap(Method ~ Metric + By, ncol = 5) +
  theme_light()
dev.off()

tiff(paste0(project_path, "beta_comparisons_REL_TF_MRTF.tiff"), units="in",
     width=4*mult, height=6*mult, res=300)
betas_long %>%
  dplyr::filter(Metric == "REL_TF_MRTF") %>%
  ggplot(aes(x = Beta, y = Correlation)) +
  geom_line() +
  facet_wrap(Method ~ Metric + By, ncol = 5) +
  theme_light()
dev.off()

#### Inspecting metrics with plots ####

doc_stats_collected_long <- doc_stats_collected %>%
  dplyr::select(-c(`In Docs`, idf, Count, Freq)) %>%
  tidyr::gather(key="Metric", value="Score", 3:11) %>%
  dplyr::arrange(Doc, Metric, desc(Score)) %>%
  dplyr::group_by(Doc, Metric) %>%
  dplyr::mutate(ScoreMinMax = minMaxScaler(Score),
                Rank = 1:dplyr::n())


doc_stats_collected_long %>%
  filter(Metric != "IRF") %>%
  ggplot(aes(x = Rank, y=ScoreMinMax, color = Metric)) +
  geom_line() +
  facet_wrap(~Doc) +
  theme_light()

set.seed(8)
metric_colors <- sample(rainbow(10), 6, FALSE)

tiff(paste0(project_path, "metrics_comparison_score_by_rank.tiff"), units="in",
     width=8, height=6, res=300)
doc_stats_collected_long %>%
  filter(Metric != "IRF",
         str_detect(Metric, "_weighted", negate=TRUE)) %>%
  dplyr::group_by(Doc, Metric) %>%
  dplyr::mutate(RankMinMax = minMaxScaler(Rank)) %>%
  dplyr::ungroup() %>%
  ggplot(aes(x = RankMinMax, y = ScoreMinMax, color = Metric)) +
  geom_line() +
  facet_wrap(Doc~.) +
  theme_light() +
  scale_color_brewer(palette = "Dark2") +
  labs(x = "Rank MinMaxScaled", y = "Score MinMaxScaled")
dev.off()

#### Extract most unique words ####

# Get top n most unique for each diagnosis

### TF_rtf
# DEPRESSION
most_unique_depression_rtf <- doc_stats_depression %>%
  dplyr::arrange(dplyr::desc(TF_RTF)) %>%
  head(10) %>%
  mutate_if(is.numeric, list(~round(., 5)))

doc_stats_control[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_depression_rtf[,"Word"], by="Word")
doc_stats_schizo[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_depression_rtf[,"Word"], by="Word")

# CONTROL
most_unique_control_rtf <- doc_stats_control %>%
  dplyr::arrange(dplyr::desc(TF_RTF)) %>%
  head(10) %>%
  mutate_if(is.numeric, list(~round(., 5)))

doc_stats_depression[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_control_rtf[,"Word"], by="Word") %>%
  .[["Freq"]] %>% round(5)
doc_stats_schizo[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_control_rtf[,"Word"], by="Word")

# SCHIZO
most_unique_schizo_rtf <- doc_stats_schizo %>%
  dplyr::arrange(dplyr::desc(TF_RTF)) %>%
  head(10) %>%
  mutate_if(is.numeric, list(~round(., 5)))

doc_stats_depression[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_schizo_rtf[,"Word"], by="Word") %>%
  .[["Freq"]] %>% round(5)
doc_stats_control[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_schizo_rtf[,"Word"], by="Word")


## TF NRTF

# DEPRESSION
most_unique_depression_nrtf <- doc_stats_depression %>%
  dplyr::arrange(dplyr::desc(TF_NRTF)) %>%
  head(10) %>%
  mutate_if(is.numeric, list(~round(., 5)))

doc_stats_control[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_depression_nrtf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)
doc_stats_schizo[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_depression_nrtf[,"Word"], by="Word")  %>%
  .[,"Freq"] %>% round(5)

# CONTROL
most_unique_control_nrtf <- doc_stats_control %>%
  dplyr::arrange(dplyr::desc(TF_NRTF)) %>%
  head(10) %>%
  mutate_if(is.numeric, list(~round(., 5)))

doc_stats_depression[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_control_nrtf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)
doc_stats_schizo[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_control_nrtf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)

# SCHIZO
most_unique_schizo_nrtf <- doc_stats_schizo %>%
  dplyr::arrange(dplyr::desc(TF_NRTF)) %>%
  head(10) %>%
  mutate_if(is.numeric, list(~round(., 5)))

doc_stats_depression[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_schizo_nrtf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)
doc_stats_control[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_schizo_nrtf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)


## Rel TF NRTF

# DEPRESSION
most_unique_depression_rel_tf_nrtf <- doc_stats_depression %>%
  dplyr::arrange(dplyr::desc(REL_TF_NRTF)) %>%
  head(10) %>%
  mutate_if(is.numeric, list(~round(., 5)))

doc_stats_control[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_depression_rel_tf_nrtf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)
doc_stats_schizo[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_depression_rel_tf_nrtf[,"Word"], by="Word")  %>%
  .[,"Freq"] %>% round(5)

# CONTROL
most_unique_control_rel_tf_nrtf <- doc_stats_control %>%
  dplyr::arrange(dplyr::desc(REL_TF_NRTF)) %>%
  head(10) %>%
  mutate_if(is.numeric, list(~round(., 5)))

doc_stats_depression[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_control_rel_tf_nrtf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)
doc_stats_schizo[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_control_rel_tf_nrtf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)

# SCHIZO
most_unique_schizo_rel_tf_nrtf <- doc_stats_schizo %>%
  dplyr::arrange(dplyr::desc(REL_TF_NRTF)) %>%
  head(10) %>%
  mutate_if(is.numeric, list(~round(., 5)))

doc_stats_depression[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_schizo_rel_tf_nrtf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)
doc_stats_control[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_schizo_rel_tf_nrtf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)



## TF MRTF

# DEPRESSION
most_unique_depression_mrtf <- doc_stats_depression %>%
  dplyr::arrange(dplyr::desc(TF_MRTF)) %>%
  head(10) %>%
  mutate_if(is.numeric, list(~round(., 5)))

doc_stats_control[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_depression_mrtf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)
doc_stats_schizo[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_depression_mrtf[,"Word"], by="Word")  %>%
  .[,"Freq"] %>% round(5)

# CONTROL
most_unique_control_mrtf <- doc_stats_control %>%
  dplyr::arrange(dplyr::desc(TF_MRTF)) %>%
  head(10) %>%
  mutate_if(is.numeric, list(~round(., 5)))

doc_stats_depression[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_control_mrtf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)
doc_stats_schizo[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_control_mrtf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)

# SCHIZO
most_unique_schizo_mrtf <- doc_stats_schizo %>%
  dplyr::arrange(dplyr::desc(TF_MRTF)) %>%
  head(10) %>%
  mutate_if(is.numeric, list(~round(., 5)))

doc_stats_depression[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_schizo_mrtf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)
doc_stats_control[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_schizo_mrtf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)


## Rel TF MRTF

# DEPRESSION
most_unique_depression_rel_tf_mrtf <- doc_stats_depression %>%
  dplyr::arrange(dplyr::desc(REL_TF_MRTF)) %>%
  head(10) %>%
  mutate_if(is.numeric, list(~round(., 5)))

doc_stats_control[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_depression_rel_tf_mrtf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)
doc_stats_schizo[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_depression_rel_tf_mrtf[,"Word"], by="Word")  %>%
  .[,"Freq"] %>% round(5)

# CONTROL
most_unique_control_rel_tf_mrtf <- doc_stats_control %>%
  dplyr::arrange(dplyr::desc(REL_TF_MRTF)) %>%
  head(10) %>%
  mutate_if(is.numeric, list(~round(., 5)))

doc_stats_depression[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_control_rel_tf_mrtf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)
doc_stats_schizo[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_control_rel_tf_mrtf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)

# SCHIZO
most_unique_schizo_rel_tf_mrtf <- doc_stats_schizo %>%
  dplyr::arrange(dplyr::desc(REL_TF_MRTF)) %>%
  head(10) %>%
  mutate_if(is.numeric, list(~round(., 5)))

doc_stats_depression[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_schizo_rel_tf_mrtf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)
doc_stats_control[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_schizo_rel_tf_mrtf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)



## TF IRF

# DEPRESSION
most_unique_depression_tf_irf <- doc_stats_depression %>%
  dplyr::arrange(dplyr::desc(TF_IRF)) %>%
  head(10) %>%
  mutate_if(is.numeric, list(~round(., 5)))

doc_stats_control[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_depression_tf_irf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)
doc_stats_schizo[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_depression_tf_irf[,"Word"], by="Word")  %>%
  .[,"Freq"] %>% round(5)

# CONTROL
most_unique_control_tf_irf <- doc_stats_control %>%
  dplyr::arrange(dplyr::desc(TF_IRF)) %>%
  head(10) %>%
  mutate_if(is.numeric, list(~round(., 5)))

doc_stats_depression[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_control_tf_irf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)
doc_stats_schizo[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_control_tf_irf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)

# SCHIZO
most_unique_schizo_tf_irf <- doc_stats_schizo %>%
  dplyr::arrange(dplyr::desc(TF_IRF)) %>%
  head(10) %>%
  mutate_if(is.numeric, list(~round(., 5)))

doc_stats_depression[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_schizo_tf_irf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)
doc_stats_control[,c("Word","Freq")] %>%
  dplyr::right_join(most_unique_schizo_tf_irf[,"Word"], by="Word") %>%
  .[,"Freq"] %>% round(5)



#### Saving plots ####

standard_width = 5
standard_height = 4
standard_res = 300

to_wordcloud <- function(tf, word_col="Word", freq_col = "Freq",
                         scale=c(4,0.4), seed = 1234){
  if (!word_col %in% colnames(tf))
    stop("'word_col' must be in 'tf'")
  if (!freq_col %in% colnames(tf))
    stop("'freq_col' must be in 'tf'")

  tf <- tf[tf[[freq_col]] > 0,]
  set.seed(seed)
  wordcloud(words = tf[[word_col]], freq = tf[[freq_col]], min.freq = 1,
            max.words=100, random.order=FALSE, rot.per=0.35,
            scale = scale,
            fixed.asp=TRUE,
            colors=brewer.pal(8, "Dark2"))

}


# tf

tiff(paste0(project_path, "wordcloud_depression_tf.tiff"), units="in",
     width=standard_width, height=standard_height, res=standard_res)
wordcloud_depression_tf <- to_wordcloud(doc_stats_depression,
                                        freq_col = "Freq",
                                        seed = 1)
dev.off()

tiff(paste0(project_path, "wordcloud_control_tf.tiff"), units="in",
     width=standard_width, height=standard_height, res=standard_res)
wordcloud_control_tf <- to_wordcloud(doc_stats_control,
                                     freq_col = "Freq",
                                     seed = 7)
dev.off()

tiff(paste0(project_path, "wordcloud_schizo_tf.tiff"), units="in",
     width=standard_width, height=standard_height, res=standard_res)
wordcloud_schizo_tf <- to_wordcloud(doc_stats_schizo,
                                    freq_col = "Freq",
                                    seed = 4)
dev.off()


# tf rtf

tiff(paste0(project_path, "wordcloud_depression_tf_rtf.tiff"), units="in",
     width=standard_width, height=standard_height, res=standard_res)
wordcloud_depression_tf_rtf <- to_wordcloud(doc_stats_depression,
                                            freq_col = "TF_RTF",
                                            seed = 1)
dev.off()

tiff(paste0(project_path, "wordcloud_control_tf_rtf.tiff"), units="in",
     width=standard_width, height=standard_height, res=standard_res)
wordcloud_control_tf_rtf <- to_wordcloud(doc_stats_control,
                                         freq_col = "TF_RTF",
                                         seed = 7)
dev.off()

tiff(paste0(project_path, "wordcloud_schizo_tf_rtf.tiff"), units="in",
     width=standard_width, height=standard_height, res=standard_res)
wordcloud_schizo_tf_rtf <- to_wordcloud(doc_stats_schizo,
                                        freq_col = "TF_RTF",
                                        seed = 4)
dev.off()


# tf_nrtf

tiff(paste0(project_path, "wordcloud_depression_tf_nrtf.tiff"), units="in",
     width=standard_width, height=standard_height, res=standard_res)
wordcloud_depression_tf_nrtf <- to_wordcloud(doc_stats_depression,
                                             freq_col = "TF_NRTF",
                                             seed = 1)
dev.off()

tiff(paste0(project_path, "wordcloud_control_tf_nrtf.tiff"), units="in",
     width=standard_width, height=standard_height, res=standard_res)
wordcloud_control_tf_nrtf <- to_wordcloud(doc_stats_control,
                                          freq_col = "TF_NRTF",
                                          seed = 7)
dev.off()

tiff(paste0(project_path, "wordcloud_schizo_tf_nrtf.tiff"), units="in",
     width=standard_width, height=standard_height, res=standard_res)
wordcloud_schizo_tf_nrtf <- to_wordcloud(doc_stats_schizo,
                                         freq_col = "TF_NRTF",
                                         seed = 4)
dev.off()

# rel_tf_nrtf

tiff(paste0(project_path, "wordcloud_depression_rel_tf_nrtf.tiff"), units="in",
     width=standard_width, height=standard_height, res=standard_res)
wordcloud_depression_rel_tf_nrtf <- to_wordcloud(doc_stats_depression,
                                             freq_col = "REL_TF_NRTF",
                                             seed = 1)
dev.off()

tiff(paste0(project_path, "wordcloud_control_rel_tf_nrtf.tiff"), units="in",
     width=standard_width, height=standard_height, res=standard_res)
wordcloud_control_tf_nrtf <- to_wordcloud(doc_stats_control,
                                          freq_col = "REL_TF_NRTF",
                                          seed = 7)
dev.off()

tiff(paste0(project_path, "wordcloud_schizo_rel_tf_nrtf.tiff"), units="in",
     width=standard_width, height=standard_height, res=standard_res)
wordcloud_schizo_tf_nrtf <- to_wordcloud(doc_stats_schizo,
                                         freq_col = "REL_TF_NRTF",
                                         seed = 5)
dev.off()


# tf irf

tiff(paste0(project_path, "wordcloud_depression_tf_irf.tiff"), units="in",
     width=standard_width, height=standard_height, res=standard_res)
wordcloud_depression_tf_irf <- to_wordcloud(doc_stats_depression,
                                            freq_col = "TF_IRF",
                                            seed = 1, scale = c(2, 0.3))
dev.off()

tiff(paste0(project_path, "wordcloud_control_tf_irf.tiff"), units="in",
     width=standard_width, height=standard_height, res=standard_res)
wordcloud_control_tf_irf <- to_wordcloud(doc_stats_control,
                                         freq_col = "TF_IRF",
                                         seed = 7, scale = c(2, 0.3))
dev.off()

tiff(paste0(project_path, "wordcloud_schizo_tf_irf.tiff"), units="in",
     width=standard_width, height=standard_height, res=standard_res)
wordcloud_schizo_tf_irf <- to_wordcloud(doc_stats_schizo,
                                        freq_col = "TF_IRF",
                                        seed = 4, scale = c(2, 0.3))
dev.off()




