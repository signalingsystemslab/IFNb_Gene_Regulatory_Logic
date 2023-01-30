library(tidyverse)
library(pheatmap)

tb = bind_cols(
  irf_tlr=c(0.1,0.25,0.01,NA),
  nfkb_tlr=c(1,0,1,NA),
  ifnb_tlr=c(0.4,0.2,0,NA),
  irf_rlr=c(0.75,1,0.075,0),
  nfkb_rlr=c(0.5,0,0.5,0.5),
  ifnb_rlr=c(1,1,0.4,0)
)

pheatmap(
  tb,
  border_color = "black",
  # color = c(rev(colorRampPalette(c("white", "navy", "midnightblue"), bias = 0.45)(50)), colorRampPalette(c("white", "firebrick3", "firebrick"), bias = 0.45)(50)),
  # breaks = seq(0, 1, length.out = 100),
  scale = "none",
  cellwidth = 60,
  cellheight = 40,
  cluster_cols = F,
  cluster_rows = F,
  show_rownames = F,
  show_colnames=F,
  display_numbers = as.matrix(tb),
  number_color="black",
  # number_format="%f",
  legend = F,
  fontsize = 20,
  filename = "heatmap_gen_obs.png")

# Export heatmap palette as RGB
pal = colorRampPalette(rev(brewer.pal(n = 7, name = "RdYlBu")))(100)
pal = hex2RGB(pal)
r = pal@coords[,1]
g = pal@coords[,2]
b = pal@coords[,3]
pal = bind_cols(r = r, g=g, b=b)

write.csv(pal, file = "../data/colormap.csv")
