install.packages("ggplot2")
install.packages("dplyr")
library(ggplot2)
library(dplyr)

#get a list of al the behavioral files and read them in
file_list <- list.files(
  path = "behav",
  pattern = "\\.csv$",
  full.names = TRUE
)
data_list <- lapply(file_list, read.csv)

#cleaning and combining the dataframes
data_list <- lapply(data_list, function(df) {
  df <- df[, !names(df) %in% "...13", drop = FALSE]
  df$subject <- as.character(df$subject)
  df
})
combined_df <- dplyr::bind_rows(data_list)
combined_df <- combined_df %>%
  mutate(
    subjective_response_group = ifelse(subjective_response == 1, "unaware", "aware")
  )

#creating accuracy column
combined_df$accuracy <- ifelse(
  combined_df$target_type == combined_df$objective_response,
  1,
  0
)

#calculating group level mean accuracy and 95% confidence intervals
plot_data <- combined_df %>%
  group_by(subjective_response_group) %>%
  summarise(
    n = sum(!is.na(accuracy)),
    accuracy = mean(accuracy, na.rm = TRUE),
    se = sqrt(accuracy * (1 - accuracy) / n),
    lower = accuracy - 1.96 * se,
    upper = accuracy + 1.96 * se,
    .groups = "drop"
  )

#calculating participant level mean accuracy
participant_means <- combined_df %>%
  group_by(subject, subjective_response_group) %>%
  summarise(
    accuracy = mean(accuracy, na.rm = TRUE),
    .groups = "drop"
  )

ggplot(plot_data, aes(
  x = subjective_response_group,
  y = accuracy
)) +
  geom_col(fill = "steelblue", width = 0.6) +
  geom_errorbar(
    aes(ymin = lower, ymax = upper),
    width = 0.2
  ) +
  geom_point(
    data = participant_means,
    aes(x = subjective_response_group, y = accuracy),
    colour = "red",
    size = 1.5,
    position = position_jitter(width = 0.2)
  ) +
  geom_hline(
    yintercept = 0.5,
    linetype = "dotted",
    colour = "black"
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.1)
  ) +
  labs(
    x = "Subjective response",
    y = "Proportion correct"
  ) +
  theme_classic()