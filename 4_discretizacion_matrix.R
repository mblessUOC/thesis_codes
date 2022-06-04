
#------------------------------------------------------------------------------
# Discretization of data in specific interval for image generation
#------------------------------------------------------------------------------
setwd("/home/max/Dropbox/Master/Materias/Trabajo Final de Master/Descarga y creacion datos")

library(TCGAbiolinks)
library(beepr)
projects <- TCGAbiolinks:::getGDCprojects()
listProjectCancer<-projects[grep("TCGA", projects$id),c("id","tumor","name")]
# listProjectCancer$tumor
typeData="Exp"
# typeData="Met"
# typeData="miRNA"
# ------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Librerias
#------------------------------------------------------------------------------
library(ggpubr)
library(dplyr)
library(tidyr)
library(shotGroups)
library(reshape2)
library(ggplot2)
library(svMisc)
library(reticulate)

#------------------------------------------------------------------------------
# Loop configuration
#------------------------------------------------------------------------------
# graph_Discretization=TRUE
graph_Discretization=TRUE
# saveNumpy=TRUE
saveNumpy=FALSE
counter=0
start=1
end=1
# end=length(listProjectCancer$tumor)

folderPath=paste("plots/discre_",typeData,"/",sep="")
# Create python folder
pythonFolder<-paste("data_",typeData,"/python",sep="")
dir.create(pythonFolder)
#------------------------------------------------------------------------------
# Información de la discretización
# dimension de la matriz para cada muestra
#------------------------------------------------------------------------------
# RNA and methylation
nDim1=nDim2=256

# miRNA
# nDim1=nDim2=64

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------
# Progress bar function
timerBar<-function(int,total){
  width <- options()$width*0.8
  cat("\r",paste0(round(int / total * 100,2), '% completed |'))
  cat(paste0(rep('=', int / total * width), collapse = ''),">",sep="")
  # cat(paste0(rep(' ', total * width-1), collapse = ''),"|",sep="")
  # Sys.sleep(.05)
  # if (int == total) cat('\014Done')
  # else cat('\014')
}
beeper<-function(times=1,freq=0.5,type=10){
  for (i in 1:times){
    beep(type)
    message("*beep*")
    Sys.sleep(freq)
  }
}
beeper(1)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Loop
#------------------------------------------------------------------------------
start.time.total<-proc.time()[3]
for (project in listProjectCancer$tumor[start:end]){
  beeper(1)
  #------------------------------------------------------------------------------
  # Title
  #------------------------------------------------------------------------------
  # project = listProjectCancer$tumor[start:end];project
  counter=counter+1
  message      ("\n*-----------------------------------*")
  message(paste(" - DISCRETIZATION OF ",project,sep=""))
  #message      ("*-----------------------------------*")
  message(paste(" - Total: ",counter,"/",length(listProjectCancer$tumor[start:end]),sep=""))
  message(paste(" - Dimensiones matriz: ",nDim2," x ", nDim2,sep=""))
  message      ("*-----------------------------------*")
  
  #------------------------------------------------------------------------------
  # Loading data
  #------------------------------------------------------------------------------
  # i=3
  loadFile = NULL
  loadFile[1] = paste("data_",typeData,"/data_rot",typeData,"_",project,".Rdata",sep="")
  loadFile[2] = paste("data_",typeData,"/metadata",typeData,"_",project,".Rdata",sep="")
  loadFile[3] = paste("data_",typeData,"/data",typeData,"_",project,".Rdata",sep="")
  # loadFile[4] = paste("data_",typeData,"/PCA",typeData,"_",project,".Rdata",sep="")
  message(" - Loading - " )
  for(i in 1:length(loadFile)){
    message(paste("          ",project," - Uploaded file: ",loadFile[i],sep=""))
    load(file = loadFile[i])
  }
  message(paste("   *Done* ",sep=""))
  #------------------------------------------------------------------------------
  # metadata #information data of each samples
  # pc_scores #values after PCA
  # data #raw data (FKPM or beta value)
  # data_rot #PC values but rotated for minimum square alligned to margins
  #------------------------------------------------------------------------------
  
  #Resumen PC1 y PC2
  summary(data_rot[,1])
  summary(data_rot[,2])
  
  # data_rot[1:20,]
  # data[1:20,1:4]
  # metadata
  class(data)
  class(data_rot)
  class(metadata)
  #------------------------------------------------------------------------------
  
  # Datos ejemplo
  # pc_score <- matrix(rnorm(1000, 5, 2), ncol=2)
  # pc_score <-abs(pc_score)
  # plot(data.frame(pc_score))
  #------------------------------------------------------------------------------
  
  #------------------------------------------------------------------------------
  # Array de datos
  # #-------
  # # Información de la discretización
  # # dimension de la matriz para cada muestra
  # nDim1=nDim2=64  
  # #-------
  
  # Cantidad de muestras
  tot.muestras<-ncol(data)
  message(" - Information - Samples to discretize: ",tot.muestras,sep="")
  muestra<-colnames(data)
  
  # Creación de array con NA
  dataArray<-array(NA,dim=c(nDim1,nDim2,tot.muestras))
  # dim(dataArray)
  
  # Rango de los valores de cada componente principal
  range.PC1 <- c(min(data_rot[,1]) , max(data_rot[,1]))
  range.PC2 <- c(min(data_rot[,2]) , max(data_rot[,2]))
  range.PC1
  range.PC2
  
  breaks.PC1 <- seq(range.PC1[1], range.PC1[2], length = nDim1)
  breaks.PC2 <- seq(range.PC2[1], range.PC2[2], length = nDim2)
  # breaks.PC1
  # breaks.PC2
  # length(breaks.PC1)
  # length(breaks.PC2)
  
  
  # Genero los intervalos para cada valor de cada PC. Las variables group.PCX son 
  # vectores que indican en qué intervalo debiera estar la CPX
  ?cut
  group.PC1 <- cut(data_rot[,1],
                   breaks=breaks.PC1,
                   include.lowest=TRUE,
                   right=FALSE
                    )
  group.PC2 <- cut(data_rot[,2], 
                     breaks=breaks.PC2, 
                     include.lowest=TRUE, 
                     right=FALSE
                     )
  
  # Inspecciono frecuencia de cada CPX en cada intervalo
  summary(group.PC1)
  summary(group.PC2)
  # sum(summary(group.PC1)) -> tiene que coincidir con nrow(BRCA_rot)
  # sum(summary(group.PC2)) -> tiene que coincidir con nrow(BRCA_rot)
  
  # group_number<-as.numeric(group_tagsX)
  
  # En la variable "intervalo.PC" ubico las coordenadas de cada celda de la matriz de cada componente
  # Al factor group.PC1 al aplicarle as.numeric(), transformo el factor en la posición del factor. esta
  # posicion es la coordenada para ubicar cada CP en el array
  
  intervalo.PC<-data.frame("pos.PC1"=as.numeric(group.PC1),
                           "pos.PC2"=as.numeric(group.PC2),
                           row.names=row.names(data_rot))
  # Esta variable indica la posición del intervalo en la matriz dataArray para cada gen
  intervalo.PC[1:3,]
  
  #------------------------------------------------------------------------------
  # Ejemplo para revisar si las PC fueron organizadas en los intervalos correctos
  data.frame("Intervalo"=as.character(group.PC1[1:3]),"PC1"=data_rot[1:3,1],"Coord_array_X"=intervalo.PC[1:3,1])
  data.frame("Intervalo"=as.character(group.PC2[1:3]),"PC2"=data_rot[1:3,2],"Coord_array_Y"=intervalo.PC[1:3,2])
  #------------------------------------------------------------------------------
  
  #Data frame con los nombres de la muestra y la identificación en la 3ra dimension del array
  df.muestras=NULL
  df.muestras<-data.frame("Matriz"=paste("Muestra",1:tot.muestras,sep="."),
                          "Muestra"=muestra,
                          "TipoTejido"=as.character(metadata$sample_type))
    
  head(df.muestras)
  
  # En el siguiente loop lo que hago es ubicar las coordenadas de cada gen y colocar las expresiones
  # para cada muestra
  
  message(" - Information - Generating Discretization of: ",nDim1,"x",nDim2,sep="")
  start.time<-proc.time()[3]
  for (row in 1:nrow(intervalo.PC)) {
    # row=1
    pos<-as.numeric(intervalo.PC[row,]) #extraigo las coordenadas de CP de un gen y lo ubico en el array
    # pos
    # dataArray[pos[1],pos[2],] <- NA
    # dataArray[pos[1],pos[2],]
    # as.numeric(BRCA_log[row,])
    # Calculo el promedio de expresión en dicho intervalo para todos los genes
    dataArray[pos[1],pos[2],] <- rowMeans(cbind(dataArray[pos[1],pos[2],],
                                                as.numeric(data[row,])),
                                          na.rm=TRUE)
    # dataArray[pos[1],pos[2],]
    # progress(row, 
    #          max.value = nrow(intervalo.PC),
    #          progress.bar = TRUE,
    #          init = (row == 0)
    #          )
    timerBar(row,nrow(intervalo.PC))
  }
  message(paste("\n   *Done* - Time: ",round(proc.time()[3]-start.time,3),"s",sep=""))
  
  # reemplazo los NA por 0
  dataArray[is.na(dataArray)] <- 0
  
  # Nombro las columnas, filas y las diferentes matrices correspondientes a cada muestra
  colnames(dataArray) <- 1:nDim1
  rownames(dataArray) <- 1:nDim2
  dimnames(dataArray)[[3]] <- paste("Muestra",1:tot.muestras,sep=".")# dataArray[,,1]
  dimnames(dataArray)[[3]] <- 1:tot.muestras
  
  #------------------------------------------------------------------------------
  #Guardo datos para rstudio
  #------------------------------------------------------------------------------
  # save(dataArray, file = "dataArray_BRCA.Rdata")
  filename = paste("data_",typeData,"/dataArray",typeData,"_",project,".Rdata",sep="")
  message(paste(" - Saving - dataArray file:",filename))
  save(dataArray, file = filename)
  dim(dataArray)
  message(paste("   *Done* ",sep=""))
  
  #-----------------------------------------------------------------------------
  #Guardo datos para python
  #------------------------------------------------------------------------------
  
  dim(dataArray) #100x100x1222
  
  if(saveNumpy){
    np <- import("numpy")
    filename = paste("data_",typeData,"/python/dataArray",typeData,"_",project,sep="")
    message(paste(" - Saving - dataArray in numpy: ",filename,".npy",sep=""))
    numpy_dataArray<-array(dataArray, c(nDim1,nDim2,tot.muestras))
    # np$save("numpy_dataArray", numpy_dataArray)
    np$save(filename, numpy_dataArray)
    message(paste("   *Done* ",sep=""))
  }else{
    message(" - Information - Not saving dataArray in numpy")
  }

  
  #------------------------------------------------------------------------------
  #Otra manera de hacer un plot con valores en una matriz
  #------------------------------------------------------------------------------
  # dataArray[49:51,49:51,1]
  # data<-dataArray[,,1]
  # data[data==0]<-NA
  # ?matplot
  # matplot(data,
  #         pch=c(19,19),
  #         col="black",
  #         cex=c(0.5,0.5)
  #         )
  
  
  
  #------------------------------------------------------------------------------
  # DISCRETIZATION PLOTS
  #------------------------------------------------------------------------------
  
  if(graph_Discretization){
    message(" - Information - Generating plots")
    
    # Convierto el array 3D en una matriz 2D
    # longData<-melt(dataMatrix)
    longData<-melt(dataArray)
    # dim(dataArray)
    # dim(longData)
    # head(longData)
    # Var1: dim1; Var2: dim2; Var3: sample; value: expresión
    # summary(longData)
    
    longData<-longData[longData$value!=0,]
    # dim(longData)
    # summary(longData)
    
    # df.muestras
    # matrixName<-df.muestras[1:3,]
    
    #------------------------------------------------------------------------------
    # GENERACION DE PLOT PARA REVISION VISUAL DE EXPRESION PARA CADA MUESTRA
    #------------------------------------------------------------------------------
    
    # Extraigo mayor valor de expresión
    maxExpression<-ceiling(max(longData$value))
    
    # Extraigo tipos de tumores entre los datos
    cancers <- as.matrix(unique(df.muestras[3]))
    cancerName <- df.muestras[3]
    message(      "   * Type of sample:")
    message(paste("     **",cancers,"\n"))
    table(cancerName)
    
    # calculo la cantidad de muestras según la cantidad de tumores
    n=4*length(cancers)
    # índices de las muestras a graficar
    randomSample<-NULL
    
    # typeCancer
    # Selecciono las filas de diferentes clases de tumores. Tambien se analiza 
    # si hay 4 muestras por tumor. Si no lo hay se agrega NA
    # j=1
    for (typeCancer in cancers){
      # typeCancer<-cancers[j];typeCancer
      
      # Cantidad de muestras según el tipo de tumor
      typeCancer_num<-sum(( df.muestras[3] == typeCancer ))
      
      # Analizo si tengo 4 muestras o si lo repito
      if (typeCancer_num > 3){
        indexTemp <- sample(which ( df.muestras[3] == typeCancer ) ,4)
      }else{
        indexTemp<-c(which(df.muestras[3]==typeCancer),
                     rep(NA,4-typeCancer_num))
      }
      # en randomSample guardo los índices de las muestras utilizadas
      randomSample<-c(randomSample,indexTemp)
      # print(randomSample)
    }
    # randomSample
    # df.muestras[randomSample,3]
    # folderPath=paste("plots/discre_",typeData,"/",sep="")
    dir.create(folderPath,
               recursive = TRUE)
    
    #------------------------------------------------------------------------------
    #Loop que va generando plots segun las filas en el dataframe randomSample
    #------------------------------------------------------------------------------
    
    message(      "   * Creating plots")
    #Genero listado para guardar los plots generados
    plot_list = list()
    index<-0
    
    for (k in randomSample){
      # k <- randomSample[1];k
      # print(df.muestras[k,3])
      index <- index+1
      # print(k)
      
      if(is.na(k)){
        # print("Hacer plot SIN datos")
        g <- ggplot(NULL)
        plot_list[[index]] <- g
      }else{
        # print("Hacer plot CON datos")
        # Selecciono los valores de una muestra específica
        data.muestra<-longData[longData$Var3==k,]
    
        g<-ggplot(data.muestra, aes(x = Var1, y = Var2)) +
          geom_tile(aes(fill=value)) +
          scale_fill_gradient(low="white", high="red", limits = c(0,maxExpression)) +
          labs(x="Interval PC1", y="Interval PC2",
               title=paste0("Muestra: \n",df.muestras[k,2]),
               subtitle = df.muestras[k,3]) +
          theme_bw() +
          theme(axis.text.x=element_text(size=9, angle=0, vjust=0.3),
                axis.text.y=element_text(size=9),
                plot.title=element_text(size=11),
                plot.margin = unit(c(1,1,1,1), "cm")) +
          theme(legend.position="top",
                legend.justification="right",
                legend.margin=margin(0,0,0,0),
                legend.box.margin=margin(-10,-10,-10,-10)) +
          xlim(0, nDim1) +
          ylim(0, nDim2)
        plot_list[[index]] = g
      # print(g)
      # print(plot_list[[9]])
      }
    }
    #------------------------------------------------------------------------------
    # Del listado de plot voy agrupando en cuadriculas y guardo un pdf
    #------------------------------------------------------------------------------
    message(        "   * Saving plots in pdf: ")
    counterPlot=0
    for (typeCancer in cancers){
      # typeCancer=cancers[1]
      message(paste("     **",typeCancer))
      # print(counterPlot)
      plot <- ggarrange(plot_list[[counterPlot+1]], plot_list[[counterPlot+2]],
                        plot_list[[counterPlot+3]], plot_list[[counterPlot+4]],
                         ncol = 2, nrow = 2)  
      # print(plot)
      namePlot=paste(folderPath,typeData,"_",project,"_",typeCancer,".pdf",sep="")
      ggsave(namePlot,
             plot,
             scale = 1,
             dpi = 1000,
             width = 210, height = 297, units = "mm")
      counterPlot=counterPlot+4
    }
    # plot1 <- ggarrange(plot_list[[1]], plot_list[[2]], plot_list[[3]], plot_list[[4]],
    #                    ncol = 2, nrow = 2)
    # # print(plot1)
    # namePlot=paste(folderPath,"plots/",typeData,"_",project,"_",cancers[1],".pdf",sep="")
    # ?ggsave
    # ggsave(namePlot,
    #        plot1,
    #        scale=1,
    #        dpi=1000,
    #        width = 210, height = 297, units = "mm")
    # 
    # plot2 <- ggarrange(plot_list[[5]], plot_list[[6]], plot_list[[7]], plot_list[[8]],
    #                    ncol = 2, nrow = 2)
    # print(plot2)
    # namePlot=paste("plots/",typeData,"_",project,"_",cancers[2],".pdf",sep="")
    # ggsave(namePlot,
    #        plot2,
    #        scale=1,
    #        dpi=1000,
    #        width = 210, height = 297, units = "mm")
    # 
    # plot3 <- ggarrange(plot_list[[9]], plot_list[[10]], plot_list[[11]], plot_list[[12]],
    #                    ncol = 2, nrow = 2)
    # print(plot3)
    # namePlot=paste("plots/",typeData,"_",project,"_",cancers[3],".pdf",sep="")
    # ggsave(namePlot,
    #        plot3,
    #        scale=1,
    #        dpi=1000,
    #        width = 210, height = 297, units = "mm")
    # 
    # # barplot((table(df.muestras$TipoTejido)),
    #         # col = rainbow(3),
    #         # main = "Class Distribution")
    # }
  }else{
    message(" - Information - No generation of sample plots")
  }
}
beeper(1,type=3)

message(      "             ********")
message(paste("\n   *FINISH Discretization* - Time: ",round(proc.time()[3]-start.time.total,3),"s",sep=""))
message(      "             ********")

