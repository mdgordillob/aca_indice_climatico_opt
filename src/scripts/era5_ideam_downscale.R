library(ColOpenData)
library(dplyr)
library(sf)
library(leaflet)
library(ggplot2)
lat <- c(4.263744, 4.263744, 4.078156, 4.078156, 4.263744)
lon <- c(-75.042067, -74.777022, -74.777022, -75.042067, -75.042067)
polygon <- st_polygon(x = list(cbind(lon, lat))) %>% st_sfc()
roi <- st_as_sf(polygon)
leaflet(roi) %>%
  addProviderTiles("OpenStreetMap") %>%
  addPolygons(
    stroke = TRUE,
    weight = 2,
    color = "#2e6930",
    fillColor = "#2e6930",
    opacity = 0.6
  )