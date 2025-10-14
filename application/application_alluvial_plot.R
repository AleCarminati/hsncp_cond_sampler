library(ggplot2)
library(ggalluvial)
library(readr)
library(showtext)
showtext_auto()
showtext_opts(dpi = 300)
font_add(family = "LM Roman 10", regular = "C:/Users/EC/AppData/Local/Microsoft/Windows/Fonts/lmroman10-regular-webfont.ttf")

df <- read_csv("alluvialdf.csv")

# make sure categorical
df$bestclus <- factor(df$bestclus)
df$bestclusHDP <- factor(df$bestclusHDP)
df$group <- factor(df$group)
df$group_end <- factor(df$group)

p1 <- ggplot(df,
       aes(axis1 = group,
           axis2 = bestclusHDP,
           axis3 = bestclus,
           axis4 = group_end)) +
  geom_alluvium(aes(fill = bestclusHDP)) +
  geom_stratum() +
  # geom_text(stat = "stratum", aes(label = after_stat(stratum))) +
  scale_x_discrete(limits = c("Groups_start", "Clustering - HDP", "Clustering - HSNCP", "Groups_end"),
                   labels = c("Groups", "Clustering - HDP", "Clustering - HSNCP", "Groups"),
                   expand = c(.05, .05)) +
  theme_minimal() +
  theme(
    legend.position = "none",
    panel.grid = element_blank(),
    text = element_text(family = "LM Roman 10", size= 12),
    axis.text.y = element_blank(),  # remove y-axis text
    axis.text.x  = element_text(),  # tick labels
    axis.ticks = element_blank(),   # remove tick marks
    axis.title = element_blank()
  )

# save to PDF
ggsave("plots/sloan_alluvial.png", p1, width = 6.4, height = 6.4*0.75)
