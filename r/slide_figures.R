# Slide figures for the off-market (list-blind) deck, drawn with ggplot2. From the repo folder:
#
#   python r/slide_figure_data.py     numbers from the models -> outputs/figure_data/*.csv
#   Rscript r/slide_figures.R         charts -> figures/slides/*.png
#
# Also reads outputs/backtest_predictions.csv (python run.py backtest): the real predictions and
# sale prices behind the error charts. Packages: ggplot2, scales, ragg.
suppressPackageStartupMessages({ library(ggplot2); library(scales) })

data_dir <- file.path("outputs", "figure_data")
out_dir <- file.path("figures", "slides")
for (f in c(file.path(data_dir, "index.csv"), file.path("outputs", "backtest_predictions.csv")))
  if (!file.exists(f)) stop(f, " is missing: run this from the repo folder after python r/slide_figure_data.py")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
navy <- "grey20"; blue <- "#00BFC4"; accent <- "#F8766D"; grey <- "grey60"; paper <- "white"   # ggplot2 default hues

theme_deck <- function(base = 17) {
  theme_grey(base_size = base) + theme(legend.position = "top")
}
# w, h: size on the 1920 x 1080 slide in px; files are drawn at twice that
save <- function(p, name, w, h) {
  ggsave(file.path(out_dir, paste0(name, ".png")), p, width = w / 100, height = h / 100,
         dpi = 200, device = ragg::agg_png, bg = paper)
}
d <- function(f) read.csv(file.path(data_dir, f), stringsAsFactors = FALSE)

# 1. market index -------------------------------------------------------------
ix <- d("index.csv"); ix$month <- as.Date(ix$month)
p <- ggplot(ix, aes(month, 100 * (exp(level) - 1), linetype = kind, colour = kind)) +
  geom_hline(yintercept = 0, colour = grey, linewidth = 0.4) +
  geom_vline(xintercept = as.Date("2026-05-31"), colour = grey, linewidth = 0.4) +
  geom_line(linewidth = 1.3) +
  annotate("text", x = as.Date("2022-03-01"), y = 18.5, label = "Peak, March 2022", hjust = 0, size = 5.5, colour = navy) +
  annotate("text", x = as.Date("2025-12-01"), y = -15, label = "Low, Dec 2025", hjust = 1, size = 5.5, colour = navy) +
  annotate("text", x = as.Date("2026-05-15"), y = 17, label = "data ends", hjust = 1, size = 5, colour = "grey30") +
  scale_colour_manual(values = c(fitted = navy, projected = blue), labels = c("Index from the data", "Projected Jun-Sep 2026"), name = NULL) +
  scale_linetype_manual(values = c(fitted = "solid", projected = "22"), labels = c("Index from the data", "Projected Jun-Sep 2026"), name = NULL) +
  scale_y_continuous(labels = function(x) paste0(ifelse(x > 0, "+", ""), x, "%")) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  labs(x = NULL, y = "Market level vs average") + theme_deck()
save(p, "index", 1600, 580)

# 2. one house and its 10 nearest detached sales --------------------------------
mp <- d("map.csv")
home <- subset(mp, role == "home"); nb <- subset(mp, role == "neighbour")
xlim <- range(c(nb$lon, home$lon)) + c(-0.0012, 0.0012); ylim <- range(c(nb$lat, home$lat)) + c(-0.0008, 0.0006)
others <- subset(mp, role == "other" & lon > xlim[1] & lon < xlim[2] & lat > ylim[1] & lat < ylim[2])
bar <- 0.2 / (111 * cos(mean(ylim) * pi / 180))   # 200 m in degrees of longitude
p <- ggplot() +
  geom_point(data = others, aes(lon, lat), colour = "grey70", size = 3) +
  geom_segment(data = nb, aes(x = home$lon, y = home$lat, xend = lon, yend = lat), colour = blue, alpha = 0.4, linewidth = 0.7) +
  geom_point(data = nb, aes(lon, lat), colour = blue, size = 5) +
  geom_point(data = home, aes(lon, lat), colour = accent, size = 8, shape = 18) +
  annotate("segment", x = xlim[2] - 0.0004 - bar, xend = xlim[2] - 0.0004, y = ylim[1] + 0.0003, yend = ylim[1] + 0.0003, linewidth = 1.2, colour = navy) +
  annotate("text", x = xlim[2] - 0.0004 - bar / 2, y = ylim[1] + 0.0005, label = "200 m", size = 5, colour = navy) +
  coord_quickmap(xlim = xlim, ylim = ylim, expand = FALSE) +
  labs(x = NULL, y = NULL) + theme_deck() +
  theme(axis.text = element_blank(), panel.grid.major = element_line(colour = "grey92"),
        panel.background = element_rect(fill = "#FFFFFF", colour = "grey70"))
save(p, "map", 700, 560)
p <- ggplot(nb, aes(ppsf, 0)) +
  geom_vline(xintercept = median(nb$ppsf), colour = accent, linewidth = 1.2) +
  geom_point(colour = blue, size = 6, position = position_jitter(width = 0, height = 0.25, seed = 1)) +
  annotate("text", x = median(nb$ppsf) + 8, y = 0.75, label = paste0("median $", round(median(nb$ppsf))), hjust = 0, size = 5.5, colour = accent) +
  scale_x_continuous(labels = dollar, limits = c(550, 950)) + scale_y_continuous(limits = c(-1, 1), breaks = NULL) +
  labs(x = "Price per sq ft, constant dollars", y = NULL) + theme_deck()
save(p, "strip", 860, 190)

# 3-8 (split, boosting, steps, recency, one tree, importance) are drawn by r/slide_figures_matplotlib.py

# 9. the Elmbank house's neighbours' misses ---------------------------------------------------
ms <- nb[order(nb$miss), ]; ms$rank <- factor(seq_len(nrow(ms)))
p <- ggplot(ms, aes(100 * miss, rank, fill = miss > 0)) +
  geom_col(width = 0.7) + geom_vline(xintercept = 0, colour = navy) +
  geom_text(aes(label = sprintf("%+.1f%%", 100 * miss), hjust = ifelse(miss > 0, -0.15, 1.15)), size = 5.3, colour = navy) +
  scale_fill_manual(values = c(`TRUE` = blue, `FALSE` = grey), guide = "none") +
  scale_x_continuous(limits = c(-10, 36), labels = function(x) paste0(x, "%")) + scale_y_discrete(labels = NULL) +
  labs(x = "How far each neighbour sold above (+) or below its honest guess", y = "10 nearest sales") + theme_deck()
save(p, "misses", 960, 500)

# 10. what the local fix buys --------------------------------------------------------------------
cg <- d("compsgain.csv"); cg$type <- factor(cg$type, levels = cg$type[order(cg$without)])
p <- ggplot(cg) +
  geom_segment(aes(x = without, xend = with, y = type, yend = type), colour = "#C77CFF", linewidth = 1.6) +
  geom_point(aes(without, type), colour = grey, size = 5) + geom_point(aes(with, type), colour = blue, size = 5) +
  geom_text(aes(pmax(without, with), type, label = sprintf("%.2f%% → %.2f%%", without, with)), hjust = -0.2, size = 5.5, colour = navy) +
  scale_x_continuous(labels = function(x) paste0(x, "%"), limits = c(5, 11.5)) +
  labs(x = "Typical miss, 2024 summer:  grey = without the local fix,  blue = with it", y = NULL) + theme_deck(19)
save(p, "compsgain", 1664, 480)

# 11. where it fails, 2025 held-back summer (from the real predictions) ------------------------
bt <- read.csv(file.path("outputs", "backtest_predictions.csv"), stringsAsFactors = FALSE)
off <- subset(bt, track == "off_market")
off$err <- 100 * (off$predicted_price / off$sold_price - 1)
s25 <- subset(off, season == 2025)
houses <- c("Detached", "Semi-Detached", "Freehold Townhouse")
bytype <- aggregate(err ~ house_type_name, s25, function(e) median(abs(e)))
bytype$n <- as.vector(table(s25$house_type_name)[bytype$house_type_name])
bytype$lab <- paste0(bytype$house_type_name, " (", comma(bytype$n), ")")
bytype$lab <- factor(bytype$lab, levels = bytype$lab[order(bytype$err)])
bytype$grp <- ifelse(bytype$house_type_name %in% houses, "house", ifelse(bytype$house_type_name == "Co-Op Apt", "coop", "condo"))
p <- ggplot(bytype, aes(err, lab, fill = grp)) +
  geom_col(width = 0.65) + geom_text(aes(label = sprintf("%.1f%%", err)), hjust = -0.15, size = 5.5, colour = navy) +
  scale_fill_manual(values = c(house = accent, condo = blue, coop = "#C77CFF"), guide = "none") +
  scale_x_continuous(limits = c(0, 11), expand = c(0, 0), labels = function(x) paste0(x, "%")) +
  labs(x = "Typical miss (MdAPE)", y = NULL) + theme_deck()
save(p, "failures_type", 800, 440)
s25$decile <- cut(rank(s25$predicted_price, ties.method = "first"), 10, labels = 1:10)
bydec <- aggregate(err ~ decile, s25, function(e) median(abs(e)))
p <- ggplot(bydec, aes(decile, err, fill = as.integer(decile) > 5)) +
  geom_col(width = 0.7) + geom_text(aes(label = sprintf("%.1f", err)), vjust = -0.5, size = 5.3, colour = navy) +
  scale_fill_manual(values = c(`TRUE` = accent, `FALSE` = blue), guide = "none") +
  scale_y_continuous(limits = c(0, 12), expand = c(0, 0), labels = function(x) paste0(x, "%")) +
  labs(x = "Predicted-price tenth (1 = cheapest)", y = "Typical miss") + theme_deck()
save(p, "failures_decile", 800, 440)

# 12. error spreads by summer, every home --------------------------------------------------------
early <- read.csv(file.path(data_dir, "early_summers.csv"), stringsAsFactors = FALSE)
early$err <- 100 * (early$predicted_price / early$sold_price - 1)
sp <- rbind(early[, c("season", "err")], off[, c("season", "err")])
qs <- do.call(rbind, lapply(split(sp$err, sp$season), function(e)
  data.frame(p10 = quantile(e, .1), p25 = quantile(e, .25), p50 = median(e), p75 = quantile(e, .75), p90 = quantile(e, .9), mdape = median(abs(e)))))
qs$season <- factor(rownames(qs), levels = rev(rownames(qs))); qs$hot <- qs$season == "2022"
p <- ggplot(qs, aes(y = season)) +
  geom_vline(xintercept = 0, colour = grey) +
  geom_linerange(aes(xmin = p10, xmax = p90, colour = hot), linewidth = 9, alpha = 0.35) +
  geom_linerange(aes(xmin = p25, xmax = p75, colour = hot), linewidth = 9) +
  geom_point(aes(x = p50), shape = 124, size = 12, colour = navy) +
  geom_text(aes(x = 31, label = sprintf("MdAPE %.1f%%  lean %+.1f%%", mdape, p50)), hjust = 0, size = 5.5, colour = navy) +
  scale_colour_manual(values = c(`TRUE` = accent, `FALSE` = blue), guide = "none") +
  scale_x_continuous(limits = c(-20, 50), breaks = seq(-20, 30, 10), labels = function(x) paste0(ifelse(x > 0, "+", ""), x, "%")) +
  labs(x = "Estimate vs sale price:  pale = middle 80% of homes,  dark = middle 50%,  tick = median", y = NULL) + theme_deck(19)
save(p, "spreads", 1664, 520)
print(qs)

# 13. which summer month, by listing date -------------------------------------------------------
wd <- d("window.csv")
wd$month <- factor(wd$month, levels = c("June", "July", "August", "September")); wd$listing <- factor(wd$listing, levels = unique(wd$listing))
p <- ggplot(wd, aes(month, draws, fill = listing)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75) +
  geom_text(aes(label = draws), position = position_dodge(width = 0.8), vjust = -0.4, size = 5.3, colour = navy) +
  scale_fill_manual(values = c(navy, blue, "#C77CFF"), name = NULL) +
  scale_y_continuous(limits = c(0, 62), expand = c(0, 0)) +
  labs(x = NULL, y = "Guesses (of 100) landing in that month") + theme_deck(19)
save(p, "window", 1664, 420)

# 14. projecting summer 2024 ----------------------------------------------------------------------
pj <- d("projex.csv"); pj$month <- factor(month.name[pj$month], levels = month.name[5:9])
pj$is_med <- grepl("Median", pj$year)
p <- ggplot(pj, aes(month, change, group = year, colour = year, linewidth = is_med, linetype = is_med)) +
  geom_hline(yintercept = 0, colour = grey) + geom_line() + geom_point(size = 3.5) +
  scale_colour_manual(values = c("2021" = "#C77CFF", "2022" = "#7CAE00", "2023" = "#00BFC4", "Median (2024 projection)" = navy), name = NULL) +
  scale_linewidth_manual(values = c(`TRUE` = 2.2, `FALSE` = 1.1), guide = "none") +
  scale_linetype_manual(values = c(`TRUE` = "22", `FALSE` = "solid"), guide = "none") +
  scale_y_continuous(labels = function(x) paste0(ifelse(x > 0, "+", ""), x, "%")) +
  labs(x = NULL, y = "Change since May") + theme_deck(19)
save(p, "projex", 1664, 440)

# 15. predicted vs actual, every out-of-sample prediction we have ---------------------------------
# five summer backtests (each trained only on sales before June 1) and the month-by-month check
va <- subset(read.csv(file.path("outputs", "validation_predictions.csv"), stringsAsFactors = FALSE), track == "off_market")
cols <- c("house_type_name", "sold_price", "predicted_price")
pa <- rbind(early[, cols], off[, cols], va[, cols])
pa$group <- factor(ifelse(pa$house_type_name %in% houses, "Houses", "Condos and co-ops"), levels = c("Condos and co-ops", "Houses"))
pa <- pa[order(pa$group), ]
pa$err <- 100 * (pa$predicted_price / pa$sold_price - 1)
lims <- range(c(pa$sold_price, pa$predicted_price))
short <- function(v) ifelse(v >= 1e6, paste0("$", v / 1e6, "M"), paste0("$", v / 1e3, "k"))
p <- ggplot(pa, aes(sold_price, predicted_price, colour = group)) +
  geom_point(size = 0.45, alpha = 0.18) +
  geom_abline(slope = 1, intercept = 0, colour = navy, linewidth = 0.9) +
  geom_abline(slope = 1, intercept = log10(c(0.9, 1.1)), colour = navy, linetype = "22", linewidth = 0.6) +
  annotate("text", x = 8e6, y = 1.25e7, label = "+10%", size = 5, colour = navy) +
  annotate("text", x = 1.3e7, y = 8e6, label = "-10%", size = 5, colour = navy) +
  scale_x_log10(limits = lims, oob = squish, breaks = c(2e5, 5e5, 1e6, 2e6, 5e6, 1e7), labels = short) +
  scale_y_log10(limits = lims, oob = squish, breaks = c(2e5, 5e5, 1e6, 2e6, 5e6, 1e7), labels = short) +
  scale_colour_manual(values = c("Condos and co-ops" = "#00BFC4", "Houses" = accent), name = NULL) +
  guides(colour = guide_legend(override.aes = list(size = 5, alpha = 1))) +
  coord_equal() + labs(x = "Actual sale price (log scale)", y = "Predicted price (log scale)") + theme_deck(19)
save(p, "predactual", 900, 680)
cat(sprintf("predicted vs actual: %d homes, MdAPE %.2f%%, within 10%% %.1f%%, within 20%% %.1f%%, houses %.2f%%, condos %.2f%%\n",
            nrow(pa), median(abs(pa$err)), 100 * mean(abs(pa$err) <= 10), 100 * mean(abs(pa$err) <= 20),
            median(abs(pa$err[pa$group == "Houses"])), median(abs(pa$err[pa$group != "Houses"]))))

# 16. five folds: every training sale is guessed by a model that never saw it ---------------------
fd <- expand.grid(part = 1:5, round = 1:5)
fd$role <- ifelse(fd$part == fd$round, "guessed, never seen", "trained on")
p <- ggplot(fd, aes(part, factor(round, levels = 5:1), fill = role)) +
  geom_tile(colour = paper, linewidth = 2.5) +
  scale_fill_manual(values = c("trained on" = "#C77CFF", "guessed, never seen" = accent), name = NULL) +
  scale_x_continuous(breaks = 1:5, labels = paste("Part", 1:5), expand = c(0, 0)) +
  scale_y_discrete(labels = paste("Round", 5:1)) +
  labs(x = "The training sales, split into five parts", y = NULL) + theme_deck(19) +
  theme(panel.grid.major = element_blank(), legend.position = "bottom")
save(p, "folds", 1000, 470)

# 17-18. the two small trees, drawn from the fitted splits ---------------------------------------
money <- function(v) ifelse(v >= 1e6, sprintf("$%.2fM", v / 1e6), sprintf("$%.0fk", v / 1e3))
tree_plot <- function(tr, name, w, h, path = FALSE) {
  tr$depth <- 0
  for (i in seq_len(nrow(tr))) if (tr$left[i] >= 0) tr$depth[c(tr$left[i], tr$right[i]) + 1] <- tr$depth[i] + 1
  leaf <- tr$left < 0
  tr$x <- NA; tr$x[leaf] <- seq_len(sum(leaf))                 # leaves left to right
  for (i in rev(seq_len(nrow(tr)))) if (!leaf[i]) tr$x[i] <- mean(tr$x[c(tr$left[i], tr$right[i]) + 1])
  tr$y <- -tr$depth
  question <- ifelse(tr$feature == "nbr_value", paste0("Neighbours' value\nunder ", money(tr$threshold), "?"),
              ifelse(tr$feature == "sqft", paste0("Under ", comma(round(tr$threshold)), " sq ft?"),
              ifelse(tr$feature == "is_condo", "A house?", paste0("Text score\nunder ", round(tr$threshold, 3), "?"))))
  tr$label <- ifelse(leaf, paste0(money(tr$value), "\n", comma(tr$n), " sales"),
              ifelse(tr$depth == 0, paste0(question, "\n", comma(tr$n), " sales"), question))
  tr$width <- ifelse(leaf, 0.86, pmin(0.9 * 2^(max(tr$depth) - tr$depth), 3.2))
  tr$on <- if (path) as.logical(tr$on_path) else rep(FALSE, nrow(tr))
  kids <- do.call(rbind, lapply(which(!leaf), function(i) data.frame(
    x = tr$x[i], y = tr$y[i] - 0.3, xend = tr$x[c(tr$left[i], tr$right[i]) + 1], yend = tr$y[i] - 0.7,
    side = c("yes", "no"), on = tr$on[i] & tr$on[c(tr$left[i], tr$right[i]) + 1])))
  p <- ggplot() +
    geom_segment(data = subset(kids, !on), aes(x, y, xend = xend, yend = yend), colour = "grey60", linewidth = 0.8) +
    geom_segment(data = subset(kids, on), aes(x, y, xend = xend, yend = yend), colour = accent, linewidth = 2.2) +
    geom_text(data = kids, aes((x + xend) / 2 + ifelse(side == "yes", -0.12, 0.12), (y + yend) / 2, label = side),
              size = 4.8, colour = "grey30") +
    geom_tile(data = tr, aes(x, y, width = width, height = 0.6, fill = interaction(leaf, on)), colour = navy, linewidth = 0.6) +
    geom_text(data = tr, aes(x, y, label = label, colour = on), size = ifelse(leaf, 5.6, 5.4), lineheight = 0.95) +
    scale_fill_manual(values = c("FALSE.FALSE" = "#FFFFFF", "TRUE.FALSE" = "grey90", "FALSE.TRUE" = accent, "TRUE.TRUE" = accent), guide = "none") +
    scale_colour_manual(values = c(`FALSE` = navy, `TRUE` = "#FFFFFF"), guide = "none") +
    scale_x_continuous(expand = expansion(add = 0.5)) + scale_y_continuous(expand = expansion(add = 0.4)) +
    theme_void() + theme(plot.background = element_rect(fill = paper, colour = NA))
  save(p, name, w, h)
}
tree_plot(d("tree2.csv"), "tree2", 1664, 560)
tree_plot(d("tree3.csv"), "tree3", 1664, 600, path = TRUE)
cat("done\n")
