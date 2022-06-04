#------------------------------------------------------------------------------
# Function to look for the minimum rectangle including all the points and rotate the point
# to have the rectangle straight
#------------------------------------------------------------------------------
setwd("/home/max/Dropbox/Master/Materias/Trabajo Final de Master/Descarga y creacion datos")

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
library(ggpubr)
library(dplyr)
library(tidyr)
library(shotGroups)
library(ggplot2)
library(TCGAbiolinks)

#------------------------------------------------------------------------------
# Variable configuration
#------------------------------------------------------------------------------
projects <- TCGAbiolinks:::getGDCprojects()
listProjectCancer<-projects[grep("TCGA", projects$id),c("id","tumor","name")]
# listProjectCancer$tumor

typeData="Exp"
# typeData="Met"
# typeData="miRNA"

#------------------------------------------------------------------------------
# Loop configuration
#------------------------------------------------------------------------------
graph_Rotaction=FALSE
counter=0
start=1
# end=3
end=length(listProjectCancer$tumor)
 # dataPCA is to store information about samples and PC explaned variance
start.time.total<-proc.time()[3]
#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------
# Definition of function that performs rotacion of the minimum area rectangle
# that includes all points

beeper<-function(times=1,freq=0.5,type=10){
  for (i in 1:times){
    beep(type)
    message("*beep*")
    Sys.sleep(freq)
  }
}
beeper(1)

selection_rotacion<-function(data,graph=FALSE,numbers=FALSE){
  # data<-data.frame(data)
  # genero rectánculo que contiene los putnos con la mínima area
  if(!numbers){
    symbols="p"
  }else{
    symbols="n"
  }
  minRect <- getMinBBox(data)  
  
  # convex hull
  H  <- chull(data)                          
  
  # centro rectánculo
  center<- c((minRect$pts[1,1]+minRect$pts[3,1])/2, (minRect$pts[1,2]+minRect$pts[3,2])/2)
  
  # visualización gráfica del rectángulo con menor área
  if(graph==TRUE){
    # plot original points, convex hull, and minimum bounding box
    plot(data, 
         xlim=range(c(data[,1], minRect$pts[,1])),
         ylim=range(c(data[,2], minRect$pts[,2])), 
         asp=1,
         # pch=16,
         type=symbols,
         col="blue",
         ylab="",yaxt="n",
         xlab="",xaxt="n",
         main=""
    )
    
    # Tengo con los puntos 
    if(numbers){
      text(data[,1],data[,2],
           labels=seq(1,nrow(data)),
           cex=1,
           font=1,
           col="blue")  
    }
    
    
    # show convex hull
    # polygon(data[H, ], col=NA)
    
    # Dibujo retángulo
    drawBox2(minRect$pts, fg='blue', colCtr='blue', pch=4, cex=2)
  }
  
  # Información rectángulo 
  area<-minRect$width*minRect$height;area # area rectangulo
  angle<-minRect$angle; angle             # orientación del rectangulo
  
  #----------------
  # Condiciones para elegir el menor angulo de rotación para que la imagen cuadre
  if(angle<180 && angle>=135){
    rotacion <- deg2rad(180-angle)  
    
  }else if(angle<135 && angle>=90) {
    rotacion <- deg2rad(90-angle)  
    
  }else if(angle<90 && angle>=45) {
    rotacion <- deg2rad(90-angle)
    
  } else { # angle<45 && angle>=0
    rotacion <- deg2rad(0-angle)
    
  }
  #----------------
  # Explicación sobre como rotar puntos https://en.wikipedia.org/wiki/Rotation_matrix
  # Matriz de rotación
  matrizRotacion <- matrix( c(cos(rotacion), -sin(rotacion),
                              sin(rotacion), cos(rotacion)),
                            nrow = 2, ncol = 2,
                            byrow = TRUE)
  # La rotación se hace sobre el origen (0,0), por lo tanto hay que centralizar los 
  # datos al centro del rectángulo y después desplazarlo al orgin:
  
  # 1) desplazo los puntos para que el origen sea el centro del rectángulo
  data_centered<-t(apply(data, 1 , function(x) x-center))
  
  # 2) genero rotación según ángulo calculado
  data_centered_rot <- t(matrizRotacion %*% t(data_centered)) 
  
  # 3) desplazo los puntos para volver al origen del comienzo
  data_rot<-t(apply(data_centered_rot, 1 , function(x) x+center))
  
  
  
  minRect_rot <- getMinBBox(data_rot)
  H_rot  <- chull(data_rot)
  center2<- c((minRect_rot$pts[1,1]+minRect_rot$pts[3,1])/2, (minRect_rot$pts[1,2]+minRect_rot$pts[3,2])/2)
  
  if(graph==TRUE){
    par(new=TRUE)
    plot(data_rot, 
         # xlim=range(c(data_rot[ , 1], minRect_rot$pts[ , 1])),
         # ylim=range(c(data_rot[ , 2], minRect_rot$pts[ , 2])), 
         xlim=range(c(data[ , 1], minRect$pts[ , 1])),
         ylim=range(c(data[ , 2], minRect$pts[ , 2])),
         asp=1,
         # pch=16,
         type=symbols,
         col="red",
         ylab="",yaxt="n",
         xlab="",xaxt="n",
         main=""
    )
    if(numbers){
      text(data_rot[,1],data_rot[,2],
           labels=seq(1,nrow(data_rot)),
           cex=1, 
           font=1,
           col="red")
    }
    
    legend(max(data[,1]*1.01),max(data[,2]*1.01),
           legend=c("Initial", "Final"),
           col=c("blue","red"),
           lty=c(1,1),
           box.lty=0
    )
    # polygon(data_rot[H_rot, ], col=NA) # show convex hull
    drawBox2(minRect_rot$pts, fg='red', colCtr='red', pch=3, cex=2)
  }
  
  
  
  #Comparo si los centros de los rectángulos se mantuvieron en el mismo lugar
  if( any(round(center,0)!=round(center2,0)) ){
    warning("Center of the initial values do not matchwith the one after rotation") 
  }
  
  #area rotada
  area_rot<-minRect_rot$width * minRect_rot$height; area_rot # rotated box area
  
  #Comparo si las areas de los rectángulos se mantuvieron iguales
  if( round(area,0)!=round(area_rot,0) ){
    warning("Initial area do not match with the one after rotation") 
  }
  
  
  #angulo final
  angle_rot<-minRect_rot$angle; angle_rot
  # print(paste0("Angulo inicial: ",angle))
  # print(paste0("Angulo finaal: ",angle_rot))
  rownames(data_rot)<-rownames(data)
  return_list <- list("data_rotated" = data_rot,
                      "rowNames" = rownames(data),
                      "angles" = data.frame("Initial"=angle,"Final"=angle_rot))
  return(return_list)
}
# función para pasar de grados a radianes
deg2rad <- function(deg) {(deg * pi) / (180)}
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Loop
#------------------------------------------------------------------------------

for (project in listProjectCancer$tumor[start:end]){
  beeper(1)
  # project="BRCA"
  #------------------------------------------------------------------------------
  # Title
  #------------------------------------------------------------------------------
  
  counter=counter+1
  message      ("\n*-----------------------------------*")
  message(paste(" - INITIATING SELECTION AND ROTATION OF ",project,sep=""))
  #message      ("*-----------------------------------*")
  message(paste(" - Total: ",counter,"/",length(listProjectCancer$tumor),sep=""))
  message      ("*-----------------------------------*")
  loadFile=paste("data_",typeData,"/PCA",typeData,"_",project,".Rdata",sep="")


  #------------------------------------------------------------------------------
  # Loading files
  #------------------------------------------------------------------------------

  message(paste(" - Loading - ",project," - Uploaded file: ",loadFile,sep=""))
  load(file = loadFile)
  message(paste("   *Done* ",sep=""))
  
  # pc_scores
  # BRCA_log[1:20,1:4]
  # pc_scores[1:20,]
  #------------------------------------------------------------------------------
  # # función para pasar de grados a radianes
  # deg2rad <- function(deg) {(deg * pi) / (180)}
  # 
  # message(paste("Information - ",project," - Definiendo funcion: selection_rotacion()",sep=""))
  
  #---------------------
  # Funcion selection_rotacion
  #---------------------
  # selection_rotacion<-function(data,graph=FALSE,numbers=FALSE){
  #   data<-data.frame(data)
  #   # genero rectánculo que contiene los putnos con la mínima area
  #   if(!numbers){
  #     symbols="p"
  #   }else{
  #     symbols="n"
  #   }
  #   minRect <- getMinBBox(data)
  # 
  #   # convex hull
  #   H  <- chull(data)
  # 
  #   # centro rectánculo
  #   center<- c((minRect$pts[1,1]+minRect$pts[3,1])/2, (minRect$pts[1,2]+minRect$pts[3,2])/2)
  # 
  #   # visualización gráfica del rectángulo con menor área
  #   if(graph==TRUE){
  #     # plot original points, convex hull, and minimum bounding box
  #     plot(data,
  #          xlim=range(c(data[,1], minRect$pts[,1])),
  #          ylim=range(c(data[,2], minRect$pts[,2])),
  #          asp=1,
  #          # pch=16,
  #          type=symbols,
  #          col="blue",
  #          ylab="",yaxt="n",
  #          xlab="",xaxt="n",
  #          main=""
  #     )
  # 
  #     # Tengo con los puntos
  #     if(numbers){
  #       text(data[,1],data[,2],
  #            labels=seq(1,nrow(data)),
  #            cex=1,
  #            font=1,
  #            col="blue")
  #     }
  # 
  # 
  #     # show convex hull
  #     # polygon(data[H, ], col=NA)
  # 
  #     # Dibujo retángulo
  #     drawBox2(minRect$pts, fg='blue', colCtr='blue', pch=4, cex=2)
  #   }
  # 
  #   # Información rectángulo
  #   area<-minRect$width*minRect$height;area # area rectangulo
  #   angle<-minRect$angle; angle             # orientación del rectangulo
  # 
  # 
  #   # Condiciones para elegir el menor angulo de rotación para que la imagen cuadre
  #   if(angle<180 && angle>=135){
  #     rotacion <- deg2rad(180-angle)
  # 
  #   }else if(angle<135 && angle>=90) {
  #     rotacion <- deg2rad(90-angle)
  # 
  #   }else if(angle<90 && angle>=45) {
  #     rotacion <- deg2rad(90-angle)
  # 
  #   } else { # angle<45 && angle>=0
  #     rotacion <- deg2rad(0-angle)
  # 
  #   }
  # 
  #     # Explicación sobre como rotar puntos https://en.wikipedia.org/wiki/Rotation_matrix
  #   # Matriz de rotación
  #   matrizRotacion <- matrix( c(cos(rotacion), -sin(rotacion),
  #                               sin(rotacion), cos(rotacion)),
  #                             nrow = 2, ncol = 2,
  #                             byrow = TRUE)
  #   # La rotación se hace sobre el origen (0,0), por lo tanto hay que centralizar los
  #   # datos al centro del rectángulo y después desplazarlo al orgin:
  # 
  #   # 1) desplazo los puntos para que el origen sea el centro del rectángulo
  #   data_centered<-t(apply(data, 1 , function(x) x-center))
  # 
  #   # 2) genero rotación según ángulo calculado
  #   data_centered_rot <- t(matrizRotacion %*% t(data_centered))
  # 
  #   # 3) desplazo los puntos para volver al origen del comienzo
  #   data_rot<-t(apply(data_centered_rot, 1 , function(x) x+center))
  # 
  # 
  # 
  #   minRect_rot <- getMinBBox(data_rot)
  #   H_rot  <- chull(data_rot)
  #   center2<- c((minRect_rot$pts[1,1]+minRect_rot$pts[3,1])/2, (minRect_rot$pts[1,2]+minRect_rot$pts[3,2])/2)
  # 
  #   if(graph==TRUE){
  #     par(new=TRUE)
  #     plot(data_rot,
  #          # xlim=range(c(data_rot[ , 1], minRect_rot$pts[ , 1])),
  #          # ylim=range(c(data_rot[ , 2], minRect_rot$pts[ , 2])),
  #          xlim=range(c(data[ , 1], minRect$pts[ , 1])),
  #          ylim=range(c(data[ , 2], minRect$pts[ , 2])),
  #          asp=1,
  #          # pch=16,
  #          type=symbols,
  #          col="red",
  #          ylab="",yaxt="n",
  #          xlab="",xaxt="n",
  #          main=""
  #     )
  #       if(numbers){
  #       text(data_rot[,1],data_rot[,2],
  #            labels=seq(1,nrow(data_rot)),
  #            cex=1,
  #            font=1,
  #            col="red")
  #       }
  # 
  #     legend(max(data[,1]*1.01),max(data[,2]*1.01),
  #            legend=c("Initial", "Final"),
  #            col=c("blue","red"),
  #            lty=c(1,1),
  #            box.lty=0
  #     )
  #     # polygon(data_rot[H_rot, ], col=NA) # show convex hull
  #     drawBox2(minRect_rot$pts, fg='red', colCtr='red', pch=3, cex=2)
  #   }
  # 
  # 
  # 
  #   #Comparo si los centros de los rectángulos se mantuvieron en el mismo lugar
  #   if( any(round(center,0)!=round(center2,0)) ){
  #     warning("Center of the initial values do not matchwith the one after rotation")
  #   }
  # 
  #   #area rotada
  #   area_rot<-minRect_rot$width * minRect_rot$height; area_rot # rotated box area
  # 
  #   #Comparo si las areas de los rectángulos se mantuvieron iguales
  #   if( round(area,0)!=round(area_rot,0) ){
  #     warning("Initial area do not match with the one after rotation")
  #   }
  # 
  # 
  #   #angulo final
  #   angle_rot<-minRect_rot$angle; angle_rot
  #   # print(paste0("Angulo inicial: ",angle))
  #   # print(paste0("Angulo finaal: ",angle_rot))
  #   rownames(data_rot)<-rownames(data)
  #   return_list <- list("data_rotated" = data_rot,
  #                       "rowNames" = rownames(data),
  #                       "angles" = data.frame("Initial"=angle,"Final"=angle_rot))
  #   return(return_list)
  # }
  #---------------------
  
  #-------------
  # Ejemplo de la función selection_rotacion(data=,graph=FALSE)
  #-------------
  # Creación datos aleatorios para evaluar rendimiento algoritmo
  # xy <- matrix(round(rnorm(1000, 100, 15)), ncol=2)
  # selection_rotacion(data=xy,graph=TRUE,numbers=FALSE)
  
  #El mismo gráfico, pero los puntos se encuentran enumerados
  # selection_rotacion(data=xy,graph=TRUE,numbers=TRUE)
  #-------------
  
  
  # Rotación del PCA
  message(paste("Information - ",project," - Rotacion datos del PCA",sep=""))
  start.time<-proc.time()[3]
  if(graph_Rotaction){
    plotGraph=TRUE
    message(paste("Information - ",project," - Generando gráfico de rotación",sep=""))
  }else{
    plotGraph=FALSE
  }
  results<-selection_rotacion(data=pc_scores,graph=plotGraph)
  data_rot<-results$data_rotated
  end.time<-Sys.time()
  message(paste("   *Done* - Time: ",round(proc.time()[3]-start.time,3),"s",sep=""))
  
  filename = paste("data_",typeData,"/data_rot",typeData,"_",project,".Rdata",sep="")
  message(paste(" - Saving - Datos rotados file:",filename))
  save(data_rot, file = filename)

  
}
beeper(1,type=3)

message(      "         ********")
message(paste("\n   *FINISH Rotacion* - Time: ",round(proc.time()[3]-start.time.total,3),"s",sep=""))
message(      "         ********")

