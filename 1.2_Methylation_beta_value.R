#-------------------------------------------------------
# Código que calcula los beta values para las muestras
# de metilación descargadas en los archivos .idat.
#-------------------------------------------------------

# Explicación de cómo utilizar minfi
# https://www.bioconductor.org/help/course-materials/2015/BioC2015/methylation450k.html

# TCGAanalyze_Preprocessing: Preprocessing of Gene Expression data (IlluminaHiSeq_RNASeqV2)
# You can easily search TCGA samples, download and prepare a matrix of gene expression.
setwd("/home/max/Dropbox/Master/Materias/Trabajo Final de Master/Descarga y creacion datos")
library(SummarizedExperiment)
library(TCGAbiolinks)
library(tidyverse)
library(beepr)
library(sesame)

library(minfi)
library(minfiData)
library(sva)


projects <- TCGAbiolinks:::getGDCprojects()
listProjectCancer <- projects[grep("TCGA", projects$id),c("id","tumor","name")]
# listProjectCancer$id

dataFiltered<-data.frame(Project=rep(NA,length(listProjectCancer$id)),
                         Sample.Size=rep(NA,length(listProjectCancer$id)),
                         Init.Probe.Size=rep(NA,length(listProjectCancer$id)),
                         Filt.SNP=rep(NA,length(listProjectCancer$id)),
                         Filt.Chr=rep(NA,length(listProjectCancer$id)),
                         Fin.Gene.Size=rep(NA,length(listProjectCancer$id))
)

#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
beeper<-function(times=1,freq=0.5,type=10){
  for (i in 1:times){
    beep(type)
    message("*beep*")
    Sys.sleep(freq)
  }
}
beeper(1)


start=32
end=33
counter=0

for (project in listProjectCancer$id[start:end]){
  global.start.time<-proc.time()[3]
  # project = listProjectCancer$id[start:end]
  # project="TCGA-TGCT"
  counter=counter+1
  dataFiltered$Project[counter] = project
  # beeper(1)
  message      ("\n*---------------------------------------------------------------------*")
  message(paste("   INITIATING BETA VALUES OF ",project,sep=""))
  message(paste("   Total projects: ",counter,"/",length(listProjectCancer$id[start:end]),sep=""))
  message      ("*---------------------------------------------------------------------*\n ")
  
  
  # project="TCGA-BRCA"
  #nombre de la base de datos de cancer
  cancer.Name<-sub("TCGA-","",project)
  
  # define a variable with missing values so it won't look for samples barcodes
  listSamples <- rlang::missing_arg()
  
  #-----------------------------------
  # STEP 1: Search, download, prepare |
  #-----------------------------------
  # 1.1 - DNA methylation
  # ----------------------------------
  # DNA methylation aligned to hg38
  
  # ----------------------------------------------------------------------------
  # GDCquery - Configuro consulta de los datos a descargar
  # ----------------------------------------------------------------------------
  start.time<-proc.time()[3]
  message("\n - Information - GDCquery function\n")
  check <- 1
  attempt <- 0
  while( check==1  && attempt < 3 ) {
    check <<- 1 
    attempt <- attempt + 1
    message(paste("    * Attempt GDCDownload: ",attempt," attempt",sep=""))
    check<-tryCatch({
      
      #--- Original code
      query_met <- GDCquery(project = project,
                            data.category = "DNA Methylation",
                            data.type = "Masked Intensities", 
                            legacy = FALSE,
                            file.type = ".idat",
                            platform = "Illumina Human Methylation 450"
                            # barcode =  listSamples
      )
      # getResults(query_met)
      # getResults(query_met,cols=c("cases","file_name"))
      check<<-0
      
    }, 
    error=function(e){
      #--- Code to repeate if error appears
      message("\n ******* Original error message:\n")
      message(e)
      message("\n Waiting 15 seconds before trying again...")
      Sys.sleep(15)
      # return(1)
      check<<-1
    }
    )
    # print(paste("Print check -",check,"-",sep=""))
    if(attempt>=3){
      dataFiltered[counter,2:5]<-"Error GDC query" 
      next
    }
  }
  message(paste(" * Done * Time:",round(proc.time()[3]-start.time,2)))
  
  # ----------------------------------------------------------------------------
  # Creo metadatos a partir del query.met
  # ----------------------------------------------------------------------------
  metadata<-as.data.frame(getResults(query_met))
  
  
  message(paste("** minfi analysis **\n"))
  message(paste(" - Information - Samples to analyze according metadata: ",length(unique(metadata$cases)),"\n",sep=""))
  

  #--------------
  # minfi analysis
  #--------------

  #directorio en donde encontré los datos
  # saveIdat <- "/data/GDCdata"
  
  #directorio en donde encontré los datos
  baseDir <- paste(getwd(),"/data/GDCdata/",project, "/harmonized/DNA_Methylation/Masked_Intensities",sep="")
  baseDir
  destDir<- paste(getwd(),"/data/GDCdata/",project, "/harmonized/DNA_Methylation/All_Idat_Files",sep="")
  destDir
  message(paste(" - Information - Moving .idat files one folder up"))
  message(paste("    * Base folder:\n",baseDir,sep=))
  message(paste("    * Destination folder:\n",destDir,sep=))
  
  base.files.path<-list.files(path = baseDir,
                              pattern = ".idat",
                              full.names = F, recursive = T,
                              include.dirs = F)
  
  # base.files.path
  freqIdat<-length(base.files.path)
  if(is.null(freqIdat)){
    freqIdat=0
  }
  message(paste("    * ",freqIdat,".idat files will be copied to the base folder (Green and Red)"))   
  
  # nuevo path de los archivos a copiar
  if(!dir.exists(destDir)){
    message(paste("    * Creating destination file"))   
    dir.create(destDir)  
  }
  # table(metadata$cases)
  
  #-----------------------------------------------------------------------------
  # Genero listado con los paths de los archivos destino 
  #-----------------------------------------------------------------------------
  
  # Elimino cases duplicados en metadata
  metadata <- metadata[!duplicated(metadata$cases),]
  
  # Genero listado de archivos .idat para Grn y Red
  # https://stackoverflow.com/questions/16143700/pasting-two-vectors-with-combinations-of-all-vectors-elements
  dist.file.name<-paste(rep(substr(metadata$file_name, start = 1, stop = 41), each = 2),
                        c("_Grn.idat","_Red.idat"), 
                        sep ="")
  
  # table(metadata$file_name %in% dist.file)
  dest.file.path<-paste(destDir,"/",dist.file.name,sep="")
  
  start.time<-proc.time()[3]
  # length(base.files.path)
  # length(dest.file.path)
  # copio archivos en cada carpeta en la carpeta superior "Masked_Intensities"
  file.copy(base.files.path,dest.file.path,
            overwrite=TRUE)
  
  #Condicion para ver si todos los archivos fueron copiados correctamente
  if(!all(file.exists(dest.file.path))){
    message("WARNING, something wrong with the copying of files")
  }

  message(paste("    * Done * - Time:",round(proc.time()[3]-start.time,2)))
  
  file.empty.folders<-paste(baseDir,"/",metadata$id,sep="")
  file.remove(file.empty.folders,recursive=TRUE)
  if(any(file.exists(base.files.path))){
    message(paste("    * WARNING - One file was not deleted and still in the folder"))
  }
  message(paste("\n - Information - Uploading .idat files for analysis"))
  dataFileCase=data.frame(Basename=substr(metadata$file_name, start = 1, stop = 41),
                          Cases=metadata$cases)
  
  # codigo para ver si las filas considen con los datos
  # pos=12
  # dataFileCase[pos,]
  # metadata[metadata$cases==dataFileCase$Cases[pos],c("file_name","cases")]
  
  message(paste("    * Total .idat file:",length(dest.file.path),""))
  message(paste("    * Total samples:",nrow(dataFileCase),"\n"))
  
  dataFiltered$Sample.Size[counter]=nrow(dataFileCase)
  
  #intervalo máximo
  sample_interval = 200
  # dataFileCase<-matrix(NA,nrow=901,ncol=10)
  
  #Calculo intervalo inferior
  intervalosInf<-seq(1,nrow(dataFileCase),by = sample_interval)
  if(nrow(dataFileCase)<sample_interval){
    intervalosSup<-nrow(dataFileCase)
  }else{
    intervalosSup<-c(seq(sample_interval,nrow(dataFileCase),by = sample_interval),nrow(dataFileCase))
  }
  
  # intervalosSup
  if(length(intervalosSup)>length(intervalosInf)){
    intervalosSup<-intervalosSup[-length(intervalosSup)]
    intervalos<-rbind(intervalosInf,intervalosSup)
  }else{
    intervalos<-rbind(intervalosInf,intervalosSup)
  }
  
  # listado en donde coloco los valores de beta_value por intervalo
  data.met<-list()
  
  # manifestList guarda datos de los tipos de posiciones de metilación
  manifestList<-list()
  message(paste(" - Information - Generating beta values"))
  message(paste("    * Creating the following intervals : Inf-Sup"))
  message(paste("                                        ",intervalos[1,],"-",intervalos[2,],"\n"))
  # i=1
  #-----------------------------------------------------------------------------
  # Loop to calculate beta_Value per batches
  #-----------------------------------------------------------------------------
  for (i in 1:ncol(intervalos)){
    message(      "    *----------------------------------------------")   
    message(paste("    - Calculation beta value for samples: ",intervalos[1,i]," a ",intervalos[2,i],sep=""))
    message(paste("    - Total: ",i,"/",ncol(intervalos),sep=""))
    message(      "    *----------------------------------------------\n")
    
    # ?read.metharray.exp
  # RGSet <- read.metharray.exp(baseDir,data.frame(Basename=dataFileCase$File))
    start.time<-proc.time()[3]

    message(paste("       ** Information - Loading batch ",i," of .idat files into variable RedGrn.set",sep=""))
    RedGrn.Set <- read.metharray.exp(base=destDir,
                                     targets=dataFileCase[intervalos[1,i]:intervalos[2,i],],
                                     recursive=TRUE)

    
    message(paste("          *Done* - Time: ",round(proc.time()[3]-start.time,2),"\n",sep=""))
    
    # phenoData <- pData(RedGrn.Set)
    # phenoData
    colnames(RedGrn.Set)<-dataFileCase$Cases[intervalos[1,i]:intervalos[2,i]]
    message(paste("       ** Information - RedGrn.Set and manifest",sep=""))
    
    # RedGrn.Set
    manifest <- getManifest(RedGrn.Set)
    manifestList[[i]]<-manifest
    manifestList[[i]]
    # dataFiltered$Init.Probe.Size[counter]=nrow(rowData(data.met))
    # head(getProbeInfo(manifest))


  # preprocessIllumina() -> with preprocessing and normalization
  # Convert a RGChannelSet to a MethylSet by implementing the preprocessing 
  # choices as available in Genome Studio: background subtraction and control
  # normalization. Both of them are optional and turning them off is equivalent 
  # to raw preprocessing (preprocessRaw):
    
    message(paste("       ** Information - Normalization of data",sep=""))
    start.time<-proc.time()[3]
    Meth.Set <- preprocessIllumina(RedGrn.Set,
                                   bg.correct = TRUE,
                                   normalize = "controls")
    
    message(paste("          *Done* - Time: ",round(proc.time()[3]-start.time,2),"\n",sep=""))
    
    dataFiltered$Init.Probe.Size[counter]=nrow(getMeth(Meth.Set))
  # colnames(Meth.Set)
  # dim(getMeth(Meth.Set))
  # dim(getUnmeth(Meth.Set))
  # head(getMeth(Meth.Set))
  # head(getUnmeth(Meth.Set))

    if(FALSE){
      message(paste("       ** Information - Quality check graphs\n",sep=""))
      # QualIty check of probes per sample
      qc <- getQC(Meth.Set)
      plotQC(qc)
      densityPlot(Meth.Set, sampGroups = phenoData$Cases)
      densityBeanPlot(Meth.Set, sampGroups = phenoData$Cases)
    }
  
  


  # ratioConvert()
  # Converting methylation data from methylation and unmethylation channels, to
  # ratios (Beta and M-values).
  
    message(paste("       ** Information - Conversion to ratios (beta, M and CNN)",sep=""))
    start.time<-proc.time()[3]
    Ratio.Set <- ratioConvert(Meth.Set,
                            what = "beta",
                            keepCN = TRUE,
                            type="Illumina" # hay que agregarle este parámetro para que calcule
                                     # el betaValue = M/(M+U+100), sino no agrega 100
                                     # y aparecen valor NA
                            )
    
    message(paste("          *Done* - Time: ",round(proc.time()[3]-start.time,2),"\n",sep=""))
  
  # Ratio.Set

  # assay(Ratio.Set)

  #The functions getBeta, getM and getCN return respectively the Beta value matrix, M value matrix and the Copy Number matrix.
  # betaValue_raw <- getBeta(Ratio.Set)
  # dim(betaValue_raw)
  # mVAlue <- getM(RSet)
  # CN <- getCN(RSet)
  # summary(betaValue_raw)

  # summary(mVAlue)
  # summary(CN)
  # mapToGenome()
  # The function mapToGenome applied to a RatioSet object will add genomic coordinates 
  # to each probe together with some additional annotation information. The output object 
  # is a GenomicRatioSet (class holding M or/and Beta values together with associated genomic
  # coordinates). It is possible to merge the manifest object with the genomic locations
  # by setting the option mergeManifest to TRUE.
    message(paste("       ** Information - Mapping probes",sep=""))
    start.time<-proc.time()[3]
  
    GenRatio.set <- mapToGenome(Ratio.Set)
    # GenRatio.set
    
    message(paste("          *Done* - Time: ",round(proc.time()[3]-start.time,2),"\n",sep=""))
  # 
    # To access the full annotation, one can use the command getAnnotation:
    # annotation <- getAnnotation(GenRatio.set)
  # names(annotation)

  # Genetic variants and cell type composition
  # presence of SNPs inside the probe body or at the nucleotide extension can 
  # have important consequences on the downstream analysis, minfi offers the
  # possibility to remove such probes. 

  # snps <- getSnpInfo(GenRatio.set)
  # head(snps,10)
  # dim(snps)
  # he function addSnpInfo will add to the GenomicRanges of the GenomicRatioSet
  # the 6 columns shown in snps
    message(paste("       ** Information - Adding SNP information",sep=""))
    start.time<-proc.time()[3]
    GenRatio.set <- addSnpInfo(GenRatio.set)
    
  # The function dropLociWithSnps allows to drop the corresponding probes with
  # SNP present inside.
    message(paste("                      - Elimnation of SNP",sep=""))
    
    GenRatio.set<-dropLociWithSnps(GenRatio.set, snps=c("SBE","CpG"), maf=0)
    # beta<-getBeta(GenRatio.set)
    # summary(GenRatio.set)
    # nrow(GenRatio.set)
    dataFiltered$Filt.SNP[counter]=dataFiltered$Init.Probe.Size[counter]-nrow(GenRatio.set)
    
    message(paste("          *Done* - Time: ",round(proc.time()[3]-start.time,2),"\n",sep=""))

    # Filtering by probes inside sexual chromosomes (chrX and chrY)
    # List of chromosomes
    # unique(as.character(seqnames(GRset)))
    message(paste("       ** Information - Elimination of probes in chrX and chrY",sep=""))
    start.time<-proc.time()[3]
    GenRatio.set_chr<-subset(GenRatio.set,subset = !as.character(seqnames(GenRatio.set)) %in% c("chrX","chrY"))
    # GenRatio.set
    # GenRatio.set_chr
    
    message(paste("          *Done* - Time: ",round(proc.time()[3]-start.time,2),"\n",sep=""))
    
    # annotation_chr <- getAnnotation(GenRatio.set_chr)
    # names(annotation_chr)
    
    dataFiltered$Filt.Chr[counter]<- nrow(GenRatio.set) - nrow(GenRatio.set_chr)
    dataFiltered$Fin.Gene.Size[counter] = nrow(GenRatio.set_chr)
    # dataFiltered[1,]
    
    # beta_final<-getBeta(GenRatio.set_chr)
    # class(beta_final)
    # data.frame(beta_final)
    message(paste("       ** Information - Saving partial beta_value in data.met",sep=""))
    data.met[[i]]<-getBeta(GenRatio.set_chr)
    start.time<-proc.time()[3] #beta_final
    message(paste("                      - Partial total amount of samples with beta_value: ",sum(sapply(data.met,ncol)),sep=""))
    beeper(1)
    # cantidad de veces que NA aparece en la matriz beta_final
    # sum(is.nan(beta_final))
    # table(is.nan(beta_final))

    # resumen y tamaño de los valores de beta_value
    # dim(beta_final)
    # summary(beta_final)
  #-----------------------------------------------------------------------------
  # En of loop for calculation beta_values per batch
  #-----------------------------------------------------------------------------
  }  
  message(paste("    - Batch calculation of beta_value for ",cancer.Name,": DONE",sep=""))
  message(      "    *----------------------------------------------\n")
  
  message(paste(" - Information - Combining all beta_value in data"))
  data<-NULL
  for (j in 1:length(data.met)){
      data <- cbind(data,data.met[[j]])
  }  
  filenameData = paste("data_Met/dataMet_",cancer.Name,".Rdata",sep="")
  message(paste(" - Information - Final dimension of beta_value ",cancer.Name,": ",nrow(data)," x ",ncol(data),sep=""))
  
  start.time<-proc.time()[3]
  message(paste("\n - Saving -",cancer.Name,"- Data file:",filenameData))
  save(data, file = filenameData)
  message(paste(" *Done* - Time: ",round(start.time<-proc.time()[3]-start.time,2),"\n",sep=""))
  beeper(1,type=2)
  
  #Condicion para ver si todos los archivos fueron copiados correctamente y borro la carpeta
  if(all(file.exists(dest.file.path))){
    message(paste(" - Deleting originals.idat files and folder in:\n",baseDir))
    file.remove(base.files.path)
    file.remove(baseDir)
  }else{
    message("WARNING: someting happened with the deletion of old values")
  }
  write.table(dataFiltered, file = "methylation_beta_value_log.txt")
  
  # message(paste(names(dataFiltered)," | ",sep=""))
  # message(paste(paste(dataFiltered$Project,"   ",dataFiltered$Sample.Size,"       ",dataFiltered$Init.Probe.Size,"           "
                      # ,dataFiltered$Filt.SNP,"     ",dataFiltered$Filt.Chr,"     ",dataFiltered$Fin.Gene.Size,"\n")))
  
  message(paste( "\n ** DONE ** ",project," - ",counter,"/",length(listProjectCancer$id[start:end]),
                   " - Loop time: " , round(proc.time()[3]-global.start.time)) 
          ) 
  #-----------------------------------------------------------------------------
  # Eliminating .idat specifics columns and saving metadata file
  #-----------------------------------------------------------------------------
  
  filenameMeta = paste("data_Met/metadataMet_",cancer.Name,".Rdata",sep="")
  metadata<-metadata[,-which(names(metadata) %in% c("file_name","file_id","channel"))]
  message(paste("\n - Saving -",cancer.Name,"- Metadata file:",filenameMeta,"\n"))
  rownames(metadata)<-seq(1:nrow(metadata))
  # metadata<-as.data.frame(getResults(query_met))
  save(metadata, file = filenameMeta)
  message(paste(" - Information - Names in metadata and data are match?",all((metadata$cases)==colnames(data)),"\n"))

  #-------------------------------------------------------------------------------
  # Elimination of big size variable to release some space in RAM
  #-------------------------------------------------------------------------------
  
  rm(list=c("metadata","data.met","data",
            "RedGrn.set","Meth.Set",
            "GenRatio.set","GenRatio.set_chr"))

  }
#-----------------------------------------------------------------------------
# End of master loop through all cancers
#-----------------------------------------------------------------------------

message(paste( "\n ** DONE METHYLATION CALCULATION ** "))
beeper(1,type=3)       
