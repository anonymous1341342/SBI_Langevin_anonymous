setwd("...") # fill in the path to your working directory

x_sum        <- as.matrix(read.csv("x_sum.csv", header = FALSE))
theta_post   <- as.matrix(read.csv("theta_post.csv", header = FALSE))[1:5000, ]
est_post     <- as.matrix(read.csv("est_post.csv", header = FALSE))
pen_est_post <- as.matrix(read.csv("pen_est_post.csv", header = FALSE))

plot_marginal_densities <- function(theta_post, est_post, pen_est_post,
                                    x_sum = NULL, add_mle = TRUE,
                                    adjust = 1.0,
                                    n = 512,
                                    title1 = expression(paste("Marginal density of ", theta[1])),
                                    title2 = expression(paste("Marginal density of ", theta[2]))) {
  stopifnot(requireNamespace("ggplot2", quietly = TRUE))

  df <- rbind(
    data.frame(value = theta_post[,1], dim = "theta1", model = "True posterior"),
    data.frame(value = est_post[,1], dim = "theta1", model = "Without curvature matching"),
    data.frame(value = pen_est_post[,1], dim = "theta1", model = "With curvature matching"),
    data.frame(value = theta_post[,2], dim = "theta2", model = "True posterior"),
    data.frame(value = est_post[,2], dim = "theta2", model = "Without curvature matching"),
    data.frame(value = pen_est_post[,2], dim = "theta2", model = "With curvature matching")
  )

  df$model <- factor(df$model, levels = c("True posterior", "Without curvature matching", "With curvature matching"))

  xlim1 <- range(df$value[df$dim == "theta1"], finite = TRUE)
  xlim2 <- range(df$value[df$dim == "theta2"], finite = TRUE)
  pad <- function(r) { d <- diff(r); c(r[1] - 0.03*d, r[2] + 0.03*d) }
  xlim1 <- pad(xlim1); xlim2 <- pad(xlim2)

  cols <- c("True posterior" = "#000000",
            "Without curvature matching" = "#0072B2",
            "With curvature matching" = "#D55E00")

  base_plot <- function(which_dim, xlim_use, title_use, xlab_expr) {
    p <- ggplot2::ggplot(df[df$dim == which_dim, ], ggplot2::aes(x = value, color = model)) +
      ggplot2::geom_density(linewidth = 1.1, adjust = adjust, n = n) +
      ggplot2::scale_color_manual(values = cols, name = NULL) +
      ggplot2::coord_cartesian(xlim = xlim_use, expand = FALSE) +
      ggplot2::labs(title = title_use, x = NULL, y = "Density") +
      ggplot2::theme_classic(base_size = 14) +
      ggplot2::theme(
        plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"),
        legend.position = "right"
      )

    if (add_mle && !is.null(x_sum) && length(x_sum) >= 2) {
      mle_val <- as.numeric(x_sum[ifelse(which_dim == "theta1", 1, 2)]) / 1000
      mle_df <- data.frame(x = mle_val, key = "MLE")

      p <- p +
        ggplot2::geom_vline(
          data = mle_df,
          ggplot2::aes(xintercept = x, linetype = key),
          color = "#D62728",
          linewidth = 0.9,
          inherit.aes = FALSE
        ) +
        ggplot2::scale_linetype_manual(values = c("MLE" = "solid"), name = NULL) +
        ggplot2::guides(
          color = ggplot2::guide_legend(order = 1),
          linetype = ggplot2::guide_legend(order = 2)
        )
    }

    p
  }

  p1 <- base_plot("theta1", xlim1, title1, expression(theta[1]))
  p2 <- base_plot("theta2", xlim2, title2, expression(theta[2]))

  list(p_theta1 = p1, p_theta2 = p2)
}

res <- plot_marginal_densities(theta_post, est_post, pen_est_post,
                               x_sum = x_sum, add_mle = TRUE,
                               adjust = 1.0)

print(res$p_theta1)
print(res$p_theta2)

plot_marginals_1row_shared_legend <- function(theta_post, est_post, pen_est_post,
                                              x_sum = NULL, add_mle = TRUE,
                                              adjust = 1.0, n = 512,
                                              pad_frac = 0.03) {
  stopifnot(requireNamespace("ggplot2", quietly = TRUE))
  stopifnot(requireNamespace("patchwork", quietly = TRUE))

  df <- rbind(
    data.frame(value = theta_post[,1], dim = "theta1", model = "True posterior"),
    data.frame(value = est_post[,1], dim = "theta1", model = "Without curvature matching"),
    data.frame(value = pen_est_post[,1], dim = "theta1", model = "With curvature matching"),
    data.frame(value = theta_post[,2], dim = "theta2", model = "True posterior"),
    data.frame(value = est_post[,2], dim = "theta2", model = "Without curvature matching"),
    data.frame(value = pen_est_post[,2], dim = "theta2", model = "With curvature matching")
  )
  df$model <- factor(df$model, levels = c("True posterior", "Without curvature matching", "With curvature matching"))

  cols <- c("True posterior" = "#000000",
            "Without curvature matching" = "#0072B2",
            "With curvature matching" = "#D55E00")

  pad_range <- function(r) {
    d <- diff(r)
    if (!is.finite(d) || d == 0) return(r)
    c(r[1] - pad_frac * d, r[2] + pad_frac * d)
  }

  xlim1 <- pad_range(range(df$value[df$dim == "theta1"], finite = TRUE))
  xlim2 <- pad_range(range(df$value[df$dim == "theta2"], finite = TRUE))

  build_one <- function(which_dim, xlim_use, title_expr,
                        mle_val = NULL, y_title = "Density") {
    dsub <- df[df$dim == which_dim, , drop = FALSE]

    ymax <- max(stats::density(dsub$value, adjust = adjust, n = n)$y, na.rm = TRUE)

    p <- ggplot2::ggplot(dsub, ggplot2::aes(x = value, color = model)) +
      ggplot2::geom_density(linewidth = 1, adjust = adjust, n = n) +
      ggplot2::scale_color_manual(values = cols, name = NULL) +
      ggplot2::coord_cartesian(xlim = xlim_use, ylim = c(0, 1.06 * ymax)) +
      ggplot2::labs(title = title_expr, x = NULL, y = y_title) +
      ggplot2::theme_classic(base_size = 14) +
      ggplot2::theme(
        plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"),
        legend.position = "bottom"
      )

    if (add_mle && !is.null(mle_val)) {
      mle_df <- data.frame(x = mle_val, key = "MLE")
      p <- p +
        ggplot2::geom_vline(
          data = mle_df,
          ggplot2::aes(xintercept = x, linetype = key),
          color = "#000000",
          linewidth = 0.8,
          inherit.aes = FALSE
        ) +
        ggplot2::scale_linetype_manual(values = c("MLE" = "dashed"), name = NULL) +
        ggplot2::guides(
          color = ggplot2::guide_legend(order = 1),
          linetype = ggplot2::guide_legend(order = 2)
        ) +
        ggplot2::theme(
          legend.key.width = grid::unit(1.8, "lines")
        )
    }

    p
  }

  mle1 <- if (add_mle && !is.null(x_sum)) as.numeric(x_sum[1]) / 1000 else NULL
  mle2 <- if (add_mle && !is.null(x_sum)) as.numeric(x_sum[2]) / 1000 else NULL

  p1 <- build_one("theta1", xlim1,
                  title_expr = expression(paste("Marginal density of ", theta[1])),
                  mle_val = mle1, y_title = "Density")

  p2 <- build_one("theta2", xlim2,
                  title_expr = expression(paste("Marginal density of ", theta[2])),
                  mle_val = mle2, y_title = "")

  (p1 | p2) +
    patchwork::plot_layout(guides = "collect") &
    ggplot2::theme(legend.position = "bottom")
}

p <- plot_marginals_1row_shared_legend(theta_post, est_post, pen_est_post,
                                       x_sum = x_sum, add_mle = TRUE,
                                       adjust = 1.0)

print(p)

ggplot2::ggsave("curv_plots/marginal_density.pdf", p,
                device = cairo_pdf,
                width = 9.0, height = 3.8, units = "in", dpi = 300)
