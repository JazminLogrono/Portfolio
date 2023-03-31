################################################
#
# Author: Jazmin Logroño
# Purpose: Poster Project, Exploring California Wildfires Danger zones 
#
################################################

#loading tidyverse library
library(tidyverse)
library(ggplot2)
library(wordcloud)
library(RColorBrewer)
library(quanteda)
library(maps)
#reading in the wildfire dataset -UPDATE FILE PATH
my.dir <- "/Users/jazminlogrono/Desktop/Syracuse University/Information Visualization - IST 719/Poster Project/data & Rscript/"
fileName <- "California_Fire_Incidents.csv"
wildFires <- read.csv(file = paste0(my.dir,fileName), 
                      sep = ",",stringsAsFactors = FALSE, header = TRUE)

################################################
#
# Data Clean up
#
################################################

wildFires <- wildFires %>% filter(Counties != "State of Nevada") %>% 
  filter(Counties != "State of Oregon") %>% filter(Counties != "Mexico")
wildFires <- wildFires[!is.na(wildFires$AcresBurned),]



################################################
#
# WIP plots
#
################################################

#What are some high-risk counties based on historical acres burned?
acresCounties <- aggregate(wildFires$AcresBurned, 
                           list(County = wildFires$Counties, Longitude =  wildFires$Longitude, Latitude = wildFires$Latitude),sum)
colnames(acresCounties) <- c('Counties', 'Longitude', 'Latitude', 'AcresBurned')
acresCounties <- acresCounties[complete.cases(acresCounties),]
acresCounties <- acresCounties %>% filter(Longitude != 0) %>%
  filter(Latitude != 0)

map(database = 'state', regions = 'California')
symbols(acresCounties$Longitude, acresCounties$Latitude,
        circles = sqrt(acresCounties$AcresBurned), add = TRUE,
        inches = 0.15, bg ="#9B180B" )

quantile(acresCounties$AcresBurned)


#Which counties have suffered the most structures damaged or structures destroyed due to wildfires?
structuresDamaged <- wildFires[!is.na(wildFires$StructuresDamaged),]
structuresDestroyed <- wildFires[!is.na(wildFires$StructuresDestroyed),]

damaged <- aggregate(structuresDamaged$StructuresDamaged, list(Counties = structuresDamaged$Counties),
                     sum)
destroyed <- aggregate(structuresDestroyed$StructuresDestroyed, list(Counties = structuresDestroyed$Counties),
                       sum)
colnames(damaged) <- c('Counties', "StructuresDamaged")
colnames(destroyed) <- c('Counties', "StructuresDestroyed")
structuresDF <- full_join(damaged, destroyed, by = "Counties")
structuresDF <- structuresDF %>% filter(StructuresDamaged != 0) %>% filter(StructuresDestroyed != 0)



ggplot(structuresDF) + aes(reorder(x = Counties,StructuresDamaged), y = StructuresDamaged) + 
  geom_bar(stat = "identity", fill = ifelse(structuresDF$StructuresDamaged > mean(structuresDF$StructuresDamaged),'#E54631', '#E69936')) + coord_flip() + 
  ggtitle("Structures Damaged By Wildfires, 2013-2019", subtitle = "Counties with zero structures damaged by wildfires are not shown.") +
  xlab('California Counties') + ylab('Number of Structures Damaged') +
  geom_text(aes(label = StructuresDamaged),vjust = 1.5, color = "#9E250D")



ggplot(structuresDF) + aes(reorder(x = Counties,StructuresDestroyed), y = StructuresDestroyed) + 
  geom_bar(stat = "identity", fill = ifelse(structuresDF$StructuresDestroyed > mean(structuresDF$StructuresDestroyed),'#E54631', '#E69936')) + coord_flip() + 
  ggtitle("Structures Destroyed By Wildfires, 2013-2019", subtitle = "Only showing Counties with Structures Destroyed by Wildfires, 2013-2019.") +
  xlab('California Counties') + ylab('Number of Structures Destroyed') +
  geom_text(aes(label = StructuresDestroyed),vjust = 1.5, color = "#9E250D")

#acres burned - yearly trend
yearlyAcresTrend <- aggregate(wildFires$AcresBurned, list(wildFires$ArchiveYear), sum)
colnames(yearlyAcresTrend) <- c('Year', 'AcresBurned')

ggplot(yearlyAcresTrend) + aes(x = Year, y = AcresBurned) + geom_line(stat = "identity", color = "#CC5500") + 
  ggtitle("California Acres Burned by Wildfires, 2013-2019", subtitle = "2017 and 2018 were anomalous relative to previous years.") + 
  ylab("Total Acres Burned") 
  
#wordCloud of location column

wordcloud(wildFires$Location, min.freq = 7, 
          random.color = FALSE, random.order = FALSE)


#What are some high-risk counties based on Injuries and Deaths?
peopleHurt <- aggregate(list(wildFires$Injuries, wildFires$Fatalities), 
                           list(County = wildFires$Counties, Longitude =  wildFires$Longitude, Latitude = wildFires$Latitude),sum)
colnames(peopleHurt) <- c('Counties', 'Longitude', 'Latitude', 'Injuries', 'Fatalities')
peopleHurt <- peopleHurt[complete.cases(peopleHurt),]


map(database = 'state', regions = 'California')
symbols(peopleHurt$Longitude, peopleHurt$Latitude,
        circles = sqrt(peopleHurt$Injuries), add = TRUE,
        inches = 0.15, bg = "#6c584c" )
symbols(peopleHurt$Longitude, peopleHurt$Latitude,
        circles = sqrt(peopleHurt$Fatalities), add = TRUE,
        inches = 0.15, bg = "#43281c" )
title(main = "Fatalities and Injuries in California Counties due to Wild Fires")


quantile(peopleHurt$Injuries)
quantile(peopleHurt$Fatalities)
