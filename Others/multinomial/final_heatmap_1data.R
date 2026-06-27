library(ggplot2)
library(dplyr)
library(tidyr)
library(MASS)
library(RColorBrewer)

df <- read.csv("rela_error_1data.csv")

lims <- range(c(df$rela_error, df$pen_rela_error), na.rm = TRUE)

df_long <- df %>%
  pivot_longer(
    cols = c(rela_error, pen_rela_error),
    names_to = "type",
    values_to = "error"
  ) %>%
  mutate(
    type = factor(
      type,
      levels = c("rela_error", "pen_rela_error"),
      labels = c("Without curvature matching", "With curvature matching")
    )
  )

u1 <- sort(unique(df$theta1))
u2 <- sort(unique(df$theta2))
dx <- median(diff(u1))
dy <- median(diff(u2))

x_sum <- as.matrix(read.csv("x_sum.csv", header = FALSE))
MLE <- data.frame(theta1 = x_sum[1]/1000, theta2 = x_sum[2]/1000, mark = "MLE")

x_breaks <- c(0.25, 0.30, 0.35, 0.40)

theta_post <- as.matrix(read.csv("theta_post.csv", header = FALSE))[1:5000, ]
stopifnot(ncol(theta_post) == 2)

xlim <- range(df$theta1, na.rm = TRUE)
ylim <- range(df$theta2, na.rm = TRUE)

n_grid <- 280
bw_fac <- c(1.5, 1.5)

h0 <- c(bandwidth.nrd(theta_post[,1]), bandwidth.nrd(theta_post[,2]))
k <- MASS::kde2d(
  theta_post[,1], theta_post[,2],
  n = n_grid,
  h = bw_fac * h0,
  lims = c(xlim[1], xlim[2], ylim[1], ylim[2])
)

z <- as.vector(k$z)
dx_k <- diff(k$x[1:2])
dy_k <- diff(k$y[1:2])

hdr_cutoff <- function(alpha) {
  z_sorted <- sort(z, decreasing = TRUE)
  cum_mass <- cumsum(z_sorted) * dx_k * dy_k
  idx <- which(cum_mass >= alpha)[1]
  z_sorted[idx]
}

alphas <- c(0.25, 0.50, 0.75, 0.95)
cuts <- sapply(alphas, hdr_cutoff)

dens_df <- data.frame(expand.grid(x = k$x, y = k$y), z = z)

p <- ggplot(df_long, aes(x = theta1, y = theta2, fill = error)) +
  geom_tile(width = dx, height = dy) +
  facet_wrap(~ type, nrow = 1) +
  coord_equal(expand = FALSE) +
  scale_fill_gradientn(
    colours = rev(RColorBrewer::brewer.pal(11, "RdYlGn")),
    limits  = lims,
    name    = "Relative Score Error"
  ) +
  geom_contour(
    data = dens_df,
    aes(x = x, y = y, z = z, linetype = "contours"),
    breaks = as.numeric(cuts),
    inherit.aes = FALSE,
    color = "black",
    linewidth = 0.7,
    show.legend = TRUE
  ) +
  scale_linetype_manual(
    name   = "25, 50, 75, 95%\ntrue posterior contours",
    values = c(contours = "dashed"),
    breaks = "contours",
    labels = " "
  ) +
  geom_point(
    data = MLE,
    aes(x = theta1, y = theta2, color = mark, shape = mark),
    inherit.aes = FALSE,
    size = 2.5,
    stroke = 0
  ) +
  scale_color_manual(
    name   = NULL,
    values = c(MLE = "blue"),
    breaks = "MLE",
    labels = "MLE"
  ) +
  scale_shape_manual(
    name   = NULL,
    values = c(MLE = 15),
    breaks = "MLE",
    labels = "MLE"
  ) +
  scale_x_continuous(
    breaks = x_breaks,
    labels = function(x) {
      ifelse(abs(x - 0.25) < 1e-12,
             "",
             format(x, nsmall = 2))
    }
  ) +
  labs(x = expression(theta[1]), y = expression(theta[2])) +
  theme_classic(base_size = 14) +
  theme(
    strip.background = element_blank(),
    strip.text = element_text(size = 14),
    panel.spacing.x = grid::unit(1.0, "lines"),
    axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 0.5, size = 11),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12),
    legend.key = element_rect(fill = "transparent", color = NA),
    legend.background = element_rect(fill = "transparent", color = NA),
    legend.box.background = element_rect(fill = "transparent", color = NA),
  ) +
  guides(
    fill     = guide_colorbar(order = 1),
    color    = guide_legend(order = 2, override.aes = list(linetype = 0, shape = 15, size = 3)),
    linetype = guide_legend(
      order = 3,
      keywidth  = grid::unit(2.4, "cm"),
      keyheight = grid::unit(0.5, "cm"),
      override.aes = list(color = "black", linewidth = 0.7)
    ),
    shape    = "none"
  )

print(p)
