library(ggplot2)
library(scales)
library(patchwork)
library(dplyr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


# single_model
loss_single_model <- data.frame(
  sm_single = numeric(0),
  sm_n = numeric(0),
  sm_n_post = numeric(0)
)
for (task_id in 0:9){
  df = read.csv(paste0("check_loss/loss_task", task_id, "_init.csv"))
  loss_single_model = rbind(loss_single_model, df)
}
colnames(loss_single_model) <- c("loss_1", "loss_n", "loss_n_post")


# single_model_C
loss_single_model_C <- data.frame(
  sm_single = numeric(0),
  sm_n = numeric(0),
  sm_n_post = numeric(0)
)
for (task_id in 0:9){
  df = read.csv(paste0("check_loss/loss_task", task_id, "_fisher_noDeb.csv"))
  loss_single_model_C = rbind(loss_single_model_C, df)
}
colnames(loss_single_model_C) <- c("loss_1", "loss_n", "loss_n_post")


# single_model_D
loss_single_model_D <- data.frame(
  sm_single = numeric(0),
  sm_n = numeric(0),
  sm_n_post = numeric(0)
)
for (task_id in 0:9){
  df = read.csv(paste0("check_loss/loss_task", task_id, "_initDebReg.csv"))
  loss_single_model_D = rbind(loss_single_model_D, df)
}
colnames(loss_single_model_D) <- c("loss_1", "loss_n", "loss_n_post")

# single_model_DC
loss_single_model_DC <- data.frame(
  sm_single = numeric(0),
  sm_n = numeric(0),
  sm_n_post = numeric(0)
)
for (task_id in 0:9){
  df = read.csv(paste0("check_loss/loss_task", task_id, "_fisher_cross.csv"))
  loss_single_model_DC = rbind(loss_single_model_DC, df)
}
colnames(loss_single_model_DC) <- c("loss_1", "loss_n", "loss_n_post")
colMeans(loss_single_model_DC)

# n_model
loss_n_model <- data.frame(
  sm_single = numeric(0),
  sm_n = numeric(0),
  sm_n_post = numeric(0)
)
for (task_id in 0:9){
  df = read.csv(paste0("nmodel/check_loss/loss_task", 
                       task_id, "_trainsize100000_init.csv"))
  loss_n_model = rbind(loss_n_model, df)
}
colnames(loss_n_model) <- c("loss_1", "loss_n", "loss_n_post")
colMeans(loss_n_model)

# n_model_C
loss_n_model_C <- data.frame(
  sm_single = numeric(0),
  sm_n = numeric(0),
  sm_n_post = numeric(0)
)
for (task_id in 0:9){
  df = read.csv(paste0("nmodel/check_loss/loss_task", 
                       task_id, "_trainsize100000_fisher.csv"))
  loss_n_model_C = rbind(loss_n_model_C, df)
}
colnames(loss_n_model_C) <- c("loss_1", "loss_n", "loss_n_post")
colMeans(loss_n_model_C)


# n_model_C_5x
loss_n_model_C_5x <- data.frame(
  sm_single = numeric(0),
  sm_n = numeric(0),
  sm_n_post = numeric(0)
)
for (task_id in 0:9){
  df = read.csv(paste0("nmodel/check_loss/loss_task", 
                       task_id, "_trainsize500000_fisher.csv"))
  loss_n_model_C_5x = rbind(loss_n_model_C_5x, df)
}
colnames(loss_n_model_C_5x) <- c("loss_1", "loss_n", "loss_n_post")
colMeans(loss_n_model_C_5x)


# check the data organization is correct
rbind(colMeans(loss_single_model),
      colMeans(loss_single_model_D),
      colMeans(loss_single_model_DC),
      colMeans(loss_n_model),
      colMeans(loss_n_model_C),
      colMeans(loss_n_model_C_5x))


###########
# Final plot
###########

my_theme <- theme_bw(base_size = 16) +  # increase base font size
  theme(
    axis.title = element_text(face = "bold", size = 24),
    axis.text = element_text(size = 22),
    strip.text = element_text(face = "bold", size = 22),
    axis.line = element_line(size = 0.6),
    panel.grid = element_line(size = 0.3, linetype = "dashed"),
    legend.position = "none",
    text = element_text(family = "Times New Roman")
  )


### loss_1
df1 <- data.frame(loss = loss_single_model$loss_1, model = "model-1")
df2 <- data.frame(loss = loss_single_model_D$loss_1, model = "model-1D")
df3 <- data.frame(loss = loss_single_model_DC$loss_1, model = "model-1DC")
df4 <- data.frame(loss = loss_n_model$loss_1, model = "model-n")
df5 <- data.frame(loss = loss_n_model_C$loss_1, model = "model-nC")
df6 <- data.frame(loss = loss_single_model_C$loss_1, model = "model-1C")

df8 <- data.frame(loss = loss_n_model_C_5x$loss_1, model = "model-nC5")

# Combine them
df_loss_1 <- rbind(df1, df2, df3, df4, df5, df6, df8)

p1 = ggplot(df_loss_1, aes(x = model, y = loss, fill = model)) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Set2") +
  ylab("loss_1") +
  xlab("") +
  ggtitle("") +
  my_theme




### loss_n
df1 <- data.frame(loss = loss_single_model$loss_n, model = "model-1")
df2 <- data.frame(loss = loss_single_model_D$loss_n, model = "model-1D")
df3 <- data.frame(loss = loss_single_model_DC$loss_n, model = "model-1DC")
df4 <- data.frame(loss = loss_n_model$loss_n, model = "model-n")
df5 <- data.frame(loss = loss_n_model_C$loss_n, model = "model-nC")
df6 <- data.frame(loss = loss_single_model_C$loss_n, model = "model-1C")

df8 <- data.frame(loss = loss_n_model_C_5x$loss_n, model = "model-nC5")

# Combine them
df_loss_n <- rbind(df1, df2, df3, df4, df5, df6, df8)

p2 = ggplot(df_loss_n, aes(x = model, y = loss, fill = model)) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Set2") +
  scale_y_log10(
    breaks = 10^(3:5),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  ylab("loss_n") +
  xlab("") +
  ggtitle("") +
  my_theme


### loss_n_post
df1 <- data.frame(loss = loss_single_model$loss_n_post, model = "model-1")
df2 <- data.frame(loss = loss_single_model_D$loss_n_post, model = "model-1D")
df3 <- data.frame(loss = loss_single_model_DC$loss_n_post, model = "model-1DC")
df4 <- data.frame(loss = loss_n_model$loss_n_post, model = "model-n")
df5 <- data.frame(loss = loss_n_model_C$loss_n_post, model = "model-nC")
df6 <- data.frame(loss = loss_single_model_C$loss_n_post, model = "model-1C")

df8 <- data.frame(loss = loss_n_model_C_5x$loss_n_post, model = "model-nC5")

# Combine them
df_loss_n_post <- rbind(df1, df2, df3, df4, df5, df6, df8)

p3 = ggplot(df_loss_n_post, aes(x = model, y = loss, fill = model)) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Set2") +
  scale_y_log10(
    labels = trans_format("log10", math_format(10^.x))
  ) +
  ylab("loss_n_post") +
  xlab("model") +
  ggtitle("") +
  my_theme +
  theme(
    axis.title.x = element_text(margin = margin(t = 15))
  )



###########
# Final plot: facet_wrap
###########

my_theme <- theme_bw(base_size = 16) +  # increase base font size
  theme(
    axis.title = element_text(face = "bold", size = 24),
    axis.text = element_text(size = 22),
    strip.text = element_text(face = "bold", size = 22),
    axis.line = element_line(size = 0.6),
    panel.grid = element_line(size = 0.3, linetype = "dashed"),
    legend.position = "none",
    text = element_text(family = "Times New Roman")
  )

df_all <- bind_rows(
  df_loss_1   %>% mutate(loss_type = "loss-1"),
  df_loss_n   %>% mutate(loss_type = "loss-n"),
  df_loss_n_post %>% mutate(loss_type = "loss-n-posterior")
)


df_all <- df_all %>%
  mutate(
    loss_plot = ifelse(loss_type %in% c("loss-n", "loss-n-posterior"), log10(loss), loss)
  )

# Custom labeller that adapts to facet type
facet_labeller <- function(variable, value) {
  if (value %in% log_facets) {
    return(scales::math_format(10^.x))
  } else {
    return(identity)
  }
}

df_all <- df_all %>%
  mutate(model = ifelse(model == "model-1", "1-model", model)) %>%
  mutate(model = ifelse(model == "model-1C", "1-model-C", model)) %>%
  mutate(model = ifelse(model == "model-1D", "1-model-D", model)) %>%
  mutate(model = ifelse(model == "model-1DC", "1-model-DC", model)) %>%
  mutate(model = ifelse(model == "model-n", "n-model", model)) %>%
  mutate(model = ifelse(model == "model-nC", "n-model-C", model)) %>%
  mutate(model = ifelse(model == "model-nC5", "n-model-C5", model)) 


final_plot <- ggplot(df_all, aes(x = model, y = loss_plot, fill = model)) +
  geom_boxplot() +
  facet_wrap(~loss_type, scales = "free_y", ncol = 1,
             labeller = labeller(loss_type = label_value)) +
  scale_fill_brewer(palette = "Set2") +
  scale_y_continuous(labels = function(x) {
    if (max(x, na.rm = TRUE) <= log10(max(df_all$loss, na.rm = TRUE)) &&
        min(x, na.rm = TRUE) >= 0 &&
        length(unique(x[!is.na(x)])) > 1) {
      parse(text = paste0("10^", x))
    } else {
      x
    }
  }) +
  labs(y = "loss", x = "model") +
  my_theme


ggsave("loss_boxplot/score_loss_plot.pdf",
       plot = final_plot,
       width = 14,
       height = 12,
       units = "in",
       device = cairo_pdf)






###########
# Subset plot: only loss-1 and loss-n, and only 1-model / 1-model-D / n-model
###########

df_subset <- df_all %>%
  filter(
    loss_type %in% c("loss-1", "loss-n"),
    model %in% c("1-model", "1-model-D", "n-model")
  ) %>%
  mutate(
    loss_type = factor(loss_type, levels = c("loss-1", "loss-n")),
    model = factor(model, levels = c("1-model", "1-model-D", "n-model"))
  )

subset_plot <- ggplot(df_subset, aes(x = model, y = loss_plot, fill = model)) +
  geom_boxplot() +
  facet_wrap(
    ~loss_type,
    scales = "free_y",
    ncol = 2,
    labeller = labeller(loss_type = label_value)
  ) +
  scale_fill_brewer(palette = "Set2") +
  scale_y_continuous(labels = function(x) {
    if (max(x, na.rm = TRUE) <= log10(max(df_all$loss, na.rm = TRUE)) &&
        min(x, na.rm = TRUE) >= 0 &&
        length(unique(x[!is.na(x)])) > 1) {
      parse(text = paste0("10^", x))
    } else {
      x
    }
  }) +
  labs(y = "loss", x = "model") +
  my_theme

ggsave("loss_boxplot/score_loss_plot_subset.pdf",
       plot = subset_plot,
       width = 12,
       height = 5,
       units = "in",
       device = cairo_pdf)






