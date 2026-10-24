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
navy <- "#14213D"; blue <- "#1F5FBF"; orange <- "#C2551F"; grey <- "#8C877A"; paper <- "#F7F5EF"

theme_deck <- function(base = 17) {
  theme_minimal(base_size = base) +
    theme(plot.background = element_rect(fill = paper, colour = NA),
          panel.grid.minor = element_blank(),
          panel.grid.major = element_line(colour = "#E3DFD3"),
          text = element_text(colour = navy),
          axis.text = element_text(colour = "#4A5568"),
          axis.title = element_text(colour = "#4A5568"),
          legend.position = "top", legend.justification = "left",
          plot.margin = margin(10, 16, 10, 10))
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
  annotate("text", x = as.Date("2026-05-15"), y = 17, label = "data ends", hjust = 1, size = 5, colour = "#4A5568") +
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
  geom_point(data = others, aes(lon, lat), colour = "#C9C4B5", size = 3) +
  geom_segment(data = nb, aes(x = home$lon, y = home$lat, xend = lon, yend = lat), colour = blue, alpha = 0.4, linewidth = 0.7) +
  geom_point(data = nb, aes(lon, lat), colour = blue, size = 5) +
  geom_point(data = home, aes(lon, lat), colour = orange, size = 8, shape = 18) +
  annotate("segment", x = xlim[2] - 0.0004 - bar, xend = xlim[2] - 0.0004, y = ylim[1] + 0.0003, yend = ylim[1] + 0.0003, linewidth = 1.2, colour = navy) +
  annotate("text", x = xlim[2] - 0.0004 - bar / 2, y = ylim[1] + 0.0005, label = "200 m", size = 5, colour = navy) +
  coord_quickmap(xlim = xlim, ylim = ylim, expand = FALSE) +
  labs(x = NULL, y = NULL) + theme_deck() +
  theme(axis.text = element_blank(), panel.grid.major = element_line(colour = "#ECE8DC"),
        panel.background = element_rect(fill = "#FFFFFF", colour = "#DDD9CC"))
save(p, "map", 700, 560)
p <- ggplot(nb, aes(ppsf, 0)) +
  geom_vline(xintercept = median(nb$ppsf), colour = orange, linewidth = 1.2) +
  geom_point(colour = blue, size = 6, position = position_jitter(width = 0, height = 0.25, seed = 1)) +
  annotate("text", x = median(nb$ppsf) + 8, y = 0.75, label = paste0("median $", round(median(nb$ppsf))), hjust = 0, size = 5.5, colour = orange) +
  scale_x_continuous(labels = dollar, limits = c(550, 950)) + scale_y_continuous(limits = c(-1, 1), breaks = NULL) +
  labs(x = "Price per sq ft, constant dollars", y = NULL) + theme_deck()
save(p, "strip", 860, 190)

# 3. how a tree picks its question ------------------------------------------------
sp <- d("split.csv")
best <- do.call(rbind, lapply(split(sp, sp$feature), function(x) x[which.max(x$drop), ]))
best$label <- ifelse(best$feature == "Size", paste0(comma(round(best$cut)), " sq ft: ", round(best$drop), "%"),
                     paste0(dollar(round(best$cut, -3)), ": ", round(best$drop), "%"))
p <- ggplot(sp, aes(100 * q, drop, colour = feature)) +
  geom_line(linewidth = 1.4) + geom_point(data = best, size = 4) +
  geom_text(data = best, aes(label = label), vjust = -1, size = 5.5, show.legend = FALSE) +
  scale_colour_manual(values = c("Neighbours' value" = blue, "Size" = orange), name = NULL) +
  scale_y_continuous(labels = function(x) paste0(x, "%"), limits = c(0, 60)) +
  labs(x = "Where the cut falls (% of sales below it)", y = "Error removed") + theme_deck()
save(p, "split", 1120, 540)

# 4. accuracy as trees are added --------------------------------------------------
bo <- d("boosting.csv")
p <- ggplot(bo, aes(trees, mdape)) +
  geom_line(colour = blue, linewidth = 1.4) + geom_point(colour = blue, size = 3.5) +
  geom_text(data = subset(bo, trees %in% c(1, 100, 2000)), aes(label = paste0(mdape, "%")), vjust = -1.1, size = 5.5, colour = navy) +
  scale_x_log10(breaks = c(1, 10, 100, 1000, 2000), labels = comma) +
  scale_y_continuous(labels = function(x) paste0(x, "%"), limits = c(0, 40)) +
  labs(x = "Trees added (log scale)", y = "Typical miss (MdAPE)") + theme_deck()
save(p, "boosting", 1600, 560)

# 5. the Elmbank house, tree by tree ------------------------------------------------
st <- d("steps.csv")
lab <- subset(st, trees %in% c(0, 50, 2000))
lab$txt <- c("start: the average home", "50 trees: pulled up by its neighbours", "2,000 trees")
p <- ggplot(st, aes(trees, value)) +
  geom_line(colour = blue, linewidth = 1.4) + geom_point(colour = blue, size = 3.5) +
  geom_point(data = lab, colour = orange, size = 5) +
  geom_text(data = lab, aes(label = paste0(dollar(round(value, -3)), "  ", txt)), hjust = c(0, 0.5, 1), vjust = c(1.9, -1.2, 1.9), size = 5.5, colour = navy) +
  scale_x_continuous(trans = "log1p", breaks = c(0, 1, 10, 100, 1000, 2000), labels = comma) +
  scale_y_continuous(labels = dollar, limits = c(870000, 1000000)) +
  labs(x = "Trees added so far (log scale)", y = "Value, constant dollars") + theme_deck()
save(p, "steps", 1600, 520)

# 6. recency weights -----------------------------------------------------------------
rc <- data.frame(months = 0:64); rc$weight <- 0.5^(rc$months / 12)
mk <- subset(rc, months %in% c(0, 12, 24, 36, 64))  # 64 months back = January 2021 in the final model
p <- ggplot(rc, aes(months, weight)) +
  geom_line(colour = blue, linewidth = 1.4) + geom_point(data = mk, colour = c(blue, blue, blue, blue, orange), size = 4) +
  geom_text(data = mk, aes(label = c("1.0 last month", "0.5 a year ago", "0.25 two years", "0.125 three years", "0.025 Jan 2021")),
            hjust = -0.1, vjust = -0.8, size = 5.3, colour = navy) +
  scale_x_continuous(breaks = seq(0, 60, 12), limits = c(0, 80)) + scale_y_continuous(limits = c(0, 1.12), breaks = seq(0, 1, 0.25)) +
  labs(x = "Months before the data ends", y = "Weight in training") + theme_deck()
save(p, "recency", 1120, 500)

# 7. one big tree vs boosting -----------------------------------------------------------
ot <- d("onetree.csv"); ot$model <- factor(ot$model, levels = rev(ot$model))
p <- ggplot(ot, aes(mdape, model, fill = model == "Four boosted models, averaged")) +
  geom_col(width = 0.65) + geom_text(aes(label = paste0(mdape, "%")), hjust = -0.15, size = 6, colour = navy) +
  scale_fill_manual(values = c(`TRUE` = navy, `FALSE` = blue), guide = "none") +
  scale_x_continuous(labels = function(x) paste0(x, "%"), limits = c(0, 13), expand = c(0, 0)) +
  labs(x = "Typical miss (MdAPE), 2024 summer", y = NULL) + theme_deck(19)
save(p, "onetree", 1664, 400)

# 8. permutation importance ---------------------------------------------------------------
im <- d("importance.csv"); im$family <- factor(im$family, levels = im$family[order(im$added)])
p <- ggplot(im, aes(added, family)) +
  geom_col(fill = blue, width = 0.65) + geom_text(aes(label = sprintf("+%.1f pts", added)), hjust = -0.15, size = 6, colour = navy) +
  scale_x_continuous(limits = c(0, 19), expand = c(0, 0)) +
  labs(x = "Points added to the typical miss when the family is scrambled (base 6.9%)", y = NULL) + theme_deck(19)
save(p, "importance", 1664, 420)

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
  geom_segment(aes(x = without, xend = with, y = type, yend = type), colour = "#B8B3A3", linewidth = 1.6) +
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
  scale_fill_manual(values = c(house = orange, condo = blue, coop = "#8C97AA"), guide = "none") +
  scale_x_continuous(limits = c(0, 11), expand = c(0, 0), labels = function(x) paste0(x, "%")) +
  labs(x = "Typical miss (MdAPE)", y = NULL) + theme_deck()
save(p, "failures_type", 800, 440)
s25$decile <- cut(rank(s25$predicted_price, ties.method = "first"), 10, labels = 1:10)
bydec <- aggregate(err ~ decile, s25, function(e) median(abs(e)))
p <- ggplot(bydec, aes(decile, err, fill = as.integer(decile) > 5)) +
  geom_col(width = 0.7) + geom_text(aes(label = sprintf("%.1f", err)), vjust = -0.5, size = 5.3, colour = navy) +
  scale_fill_manual(values = c(`TRUE` = orange, `FALSE` = blue), guide = "none") +
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
  scale_colour_manual(values = c(`TRUE` = orange, `FALSE` = blue), guide = "none") +
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
  scale_fill_manual(values = c(orange, blue, navy), name = NULL) +
  scale_y_continuous(limits = c(0, 62), expand = c(0, 0)) +
  labs(x = NULL, y = "Guesses (of 100) landing in that month") + theme_deck(19)
save(p, "window", 1664, 420)

# 14. projecting summer 2024 ----------------------------------------------------------------------
pj <- d("projex.csv"); pj$month <- factor(month.name[pj$month], levels = month.name[5:9])
pj$is_med <- grepl("Median", pj$year)
p <- ggplot(pj, aes(month, change, group = year, colour = year, linewidth = is_med, linetype = is_med)) +
  geom_hline(yintercept = 0, colour = grey) + geom_line() + geom_point(size = 3.5) +
  scale_colour_manual(values = c("2021" = "#8C97AA", "2022" = orange, "2023" = "#5F86C9", "Median (2024 projection)" = navy), name = NULL) +
  scale_linewidth_manual(values = c(`TRUE` = 2.2, `FALSE` = 1.1), guide = "none") +
  scale_linetype_manual(values = c(`TRUE` = "22", `FALSE` = "solid"), guide = "none") +
  scale_y_continuous(labels = function(x) paste0(ifelse(x > 0, "+", ""), x, "%")) +
  labs(x = NULL, y = "Change since May") + theme_deck(19)
save(p, "projex", 1664, 440)
cat("done\n")
