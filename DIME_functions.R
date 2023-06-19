
# DIME functions
# Created by: Fernan Rodrigo Perez Galvez
# Date: April 13 2023

# libraries required
library(changepoint)
library(ggplot2)
library(gridExtra)


# handle directories of files
rename_file <- function(input_path, suffix){
  
  # get the directory of the input file
  output_dir <- file.path(dirname(input_path))
  # If the input file is in the current working directory, set output_dir to an empty string
  if (output_dir == getwd()) {
    output_dir <- ""
  }
  # get name of file
  input_file_name <- file.path(basename(input_path))
  # add stamp
  new_data_name <- gsub("\\.csv$", suffix, input_file_name)
  output_path <- file.path(output_dir, new_data_name)
  
  return(output_path)
}



# score DataFrame files using a median or mean threshold
score_median <- function(data_file, meanThresh = FALSE, reverse_flag = FALSE){
  
  # read dataframe
  datos <- read.csv(data_file)
  
  # extract dimentions
  colNum <- dim(datos)[2]
  # column names
  colNames <- colnames(datos)
  
  # initialize results dataframe
  # three columns in dataframe are descriptive: index, seconds, minutes
  results <- data.frame(matrix(NA, nrow = colNum-3, ncol = 3))
  colnames(results) <- c('ID', 'threshold', 'estimate')
  
  # retrieve minutes vector
  tempTime <- datos[,colNum]
  
  # cycle through PIC data columns, from 2 to N-2
  for(j in 2:(colNum-2)){
    
    # extract PIC vector
    tempSeries <- datos[,j]
    tempData <- as.data.frame(cbind(tempTime,tempSeries))
    tempData$minutes <- tempData$tempTime
    
    # extract PIC values different from zero
    serie_nonzero <- tempData[tempData$tempSeries != 0,]
    # remove NaNs
    serie_nonzero <- serie_nonzero[complete.cases(serie_nonzero),]
    
    # in case the values are all zero fill row with NA
    if(dim(serie_nonzero)[1]==0){
      threshold <- NA
      estimate <- NA
      results[(j-1),] <- c(colNames[j], threshold, estimate)
      
    }else{
      # compute threshold
      if(meanThresh){
        threshold <- mean(serie_nonzero$tempSeries, na.rm = TRUE)
      }else{
        threshold <- median(serie_nonzero$tempSeries, na.rm = TRUE)
      }
      
      # if threshold is empty return NA
      if(is.na(threshold)){
        threshold <- NA
        estimate <- NA
        results[(j-1),] <- c(colNames[j], threshold, estimate)
      }else{
        
        # calculate endpoint
        if(reverse_flag){
          endpoint <- head(serie_nonzero[serie_nonzero$tempSeries>=threshold,], n=1)
        }else{
          endpoint <- tail(serie_nonzero[serie_nonzero$tempSeries>=threshold,], n=1)
        }
        
        # if endpoint is empty return NA
        if(dim(endpoint)[1]==0){
          threshold <- NA
          estimate <- NA
          results[(j-1),] <- c(colNames[j], threshold, estimate)
        }else{
          # include estimate results
          estimate <- endpoint$minutes
          results[(j-1),] <- c(colNames[j], threshold, estimate)
        }
      }
    }
  }
  
  # manage directories
  output_path <- rename_file(data_file, "_TKD_median.csv")
  # save data
  write.csv(results, output_path)
  
  return(results)
}


MacLean_iterations <- function (data_frame_results, threshold, trimming = 0, decreasing = FALSE, reverse = FALSE){
  
  # extract dimentions
  col_num <- dim(data_frame_results)[2]
  # column names
  colNames_data <- colnames(data_frame_results)
  # retrieve minutes vector
  tempTime <- data_frame_results[,col_num]
  
  # initialize results dataframe
  # three columns in dataframe are descriptive: index, seconds, minutes
  results_iteration <- data.frame(matrix(NA, nrow = col_num-3, ncol = 2))
  colnames(results_iteration) <- c('ID', 'estimate')
  
  # cycle through PIC data columns, from 2 to N-2
  for(j in 2:(col_num-2)){
    # add ID
    results_iteration$ID[j-1] <- colNames_data[j]
    # extract PIC vector
    tempSeries <- data_frame_results[,j]
    tempData <- as.data.frame(cbind(tempTime, tempSeries))

    # reverse dataframe if required
    if (reverse){
      tempData <- tempData[rev(row.names(tempData)),]
    }
    # if desired, trim the end of the dataframe by 'trimming'
    tempData <- tempData[1:(nrow(tempData)-trimming),]
    
    # NA PICs to zero
    tempData$tempSeries[is.na(tempData$tempSeries)] <- 0
    # set PIC values less or equal to threshold to zero
    tempData$tempSeries[tempData$tempSeries < threshold] <- 0
    serie_modified <- tempData[tempData$tempSeries > 0,]
    # if the last distance is less than threshold transform to zero
    if (tempData$tempSeries[length(tempData$tempSeries)] >0){
      TKD <- NA
      results_iteration$estimate[j-1] <- TKD
    }else{
      if (decreasing){
      # find indexes where the distance reduces
        indices <- which((serie_modified$tempSeries[-1]-serie_modified$tempSeries[-length(serie_modified)]) <0)
        
        #set the time where PIC decreases
        decreasing_index <- indices[length(indices)] + 1
        TKD <- serie_modified$tempTime[decreasing_index]

      }else{
      # or compute just the last value
        TKD <- serie_modified$tempTime[length(serie_modified$tempTime)]
      }

      # if TKD is empty set to 0
      if(length(TKD) == 0){
        TKD <- 0
        results_iteration$estimate[j-1] <- TKD
      }else{
        results_iteration$estimate[j-1] <- TKD
      }
    }
  }
  #print(TKD)
  return(results_iteration)
}



# score DataFrame files with MacLean 2022 algorithm
score_MacLean2022 <- function(data_file, threshold_initial = 0.0, threshold_final = 10.0, threshold_steps = 5, reverse_flag = FALSE){
  
  # read dataframe
  datos <- read.csv(data_file)
  # calculate thresholds
  thresholds <- seq(from = threshold_initial, to = threshold_final, length.out = threshold_steps)

  # extract dimentions
  col_num <- dim(datos)[2]
  # column names
  well_names <- colnames(datos)
  # retrieve minutes vector
  temp_time <- datos[,col_num]

  # initialize results dataframe
  # three columns in dataframe are descriptive: index, seconds, minutes
  all_iterations <- data.frame(matrix(NA, nrow = col_num-3, ncol = threshold_steps + 1))
  results_names <- list()
  for (i in seq_along(thresholds)){
    estimate_name <- paste0('estimate_', round(thresholds[i], 2))
    results_names[[i]] <- estimate_name
  }
  colnames(all_iterations) <- c('ID', results_names)

  # compute the TDK for every threshold
  for (i in seq_along(thresholds)){

    # set current threshold
    threshold_current = thresholds[i]
    # get comparison iteration results_temp  
    current_iteration <- MacLean_iterations(datos, threshold_current, reverse = reverse_flag)
    all_iterations[,i+1] <- current_iteration$estimate
    # add comlumns
    if(i == 1){
      all_iterations[,i] <- current_iteration$ID
    }
  }
  #print(head(all_iterations))
  # create a dataframe for regression
  # number of regressions is number of thresholds - 1
  # rows have the list of comparisons
  # columns are Comparison, First threshold, F-statistic, Pval, R2
  regression_estimates <- data.frame(matrix(NA, nrow = threshold_steps -1, ncol = 4))
  colnames(regression_estimates) <- c('ID','thresh', 'F-statistic', 'R2')
  

  # create a list of comparision names
  results_names <- list()
  for (i in seq_along(thresholds[-length(thresholds)])){
    comparisons <- paste0('estimate_', round(thresholds[i], 2), "_vs_", round(thresholds[i+1], 2))
    regression_estimates$ID[i] <- comparisons
  }
  # plot list
  plot_list <- list()
  nrow_plot <- floor(sqrt(threshold_steps-1))
  ncol_plot <- ceiling((threshold_steps-1)/nrow_plot)

  # compute regression table
  for (i in seq_along(thresholds[-length(thresholds)])){
    # get series
    series_initial <- all_iterations[,i+1]
    series_contrast <- all_iterations[,i+2]
    series_df <- as.data.frame(cbind(series_initial, series_contrast))
    # compute lineal model
    regression <- lm(series_contrast ~ series_initial, data = series_df)
    test_summary <- summary(regression)
    
    # set parameters
    regression_estimates$thresh[i] <- thresholds[i]
    regression_estimates$`F-statistic`[i] <- test_summary$fstatistic[1]
    #regression_estimates$Pval[i] <- NA
    regression_estimates$R2[i] <- round(test_summary$r.squared, 2)

    # plot
    # Plot the raw data and the regression line
    p <- ggplot(data = series_df) +
      geom_point(aes(x = series_initial, y = series_contrast), shape = 20) +
      labs(title = paste("R2 =", round(test_summary$r.squared, 2)),
           x = paste0("Min threshold:", round(thresholds[i], 2)),
           y = paste0("New Threshold:", round(thresholds[i+1], 2))) +
      geom_abline(intercept = 0, slope = 1, color = "blue", linetype = "dashed") +
      theme_bw(base_size = 8)
    
    # add the plot to the list
    plot_list[[i]] <- p
  }
  
  # manage directories
  label <- paste0("ti", threshold_initial, "tf", threshold_final, "ts", threshold_steps)
  dataset_path <- rename_file(data_file, paste0("_TKD_",label,"_MacLean2022.csv"))
  regression_path <- rename_file(data_file, paste0("_Regression_",label, "_MacLean2022.csv"))
  plot_path <- rename_file(data_file, paste0("_Regression_",label,"_MacLean2022.png"))
  # save data
  write.csv(all_iterations, dataset_path)
  write.csv(as.data.frame(regression_estimates), regression_path)
  
  # create the grid of plots
  my_grid <- grid.arrange(grobs = plot_list, ncol = ncol_plot, nrow = nrow_plot)
  # save the grid to a file
  ggsave(plot_path, plot = my_grid, width = 6, height = 4, dpi = 300)

  return(list(df1 = all_iterations, df2 = regression_estimates))
}

# change point function
score_changePoint <- function(data_file){
  
  # read dataframe
  datos <- read.csv(data_file)
  
  # extract dimentions
  colNum <- dim(datos)[2]
  # column names
  colNames <- colnames(datos)
  
  # initialize results dataframe
  # three columns in dataframe are descriptive: index, seconds, minutes
  results <- data.frame(matrix(NA, nrow = colNum-3, ncol = 3))
  colnames(results) <- c('ID', 'confidence', 'estimate')
  
  # retrieve minutes vector
  tempTime <- datos[,colNum]
  
  # cycle through PIC data columns, from 2 to N-2
  for(j in 2:(colNum-2)){
    
    # extract PIC vector
    tempSeries <- datos[,j]
    tempSeries <- tempSeries[complete.cases(tempSeries)]
    
    # calculate change point
    cp_model<- cpt.mean(tempSeries, method = "AMOC", class = FALSE)
    temp_index <- cp_model[1]
    time_KD <- tempTime[temp_index]
    confidence <- cp_model[2]
    # if object is empty
    if(length(cp_model) == 0){
      # empty endpoint
      estimate <- NA
    }else{
      # calculate endpoint
      estimate<-time_KD
    }
    # append iteration results
    threshold <- NA
    results[(j-1),] <- c(colNames[j], confidence, estimate)
  }
  
  # manage directories
  output_path <- rename_file(data_file, "_TKD_changePoint.csv")
  # save data
  write.csv(results, output_path)
  
  return(results)
}

fast_DIME_functions <- function(data_input){
  
  # read dataframe
  datos <- read.csv(data_input)
  
  #run functions
  score_MacLean2022(datos)
  score_median(datos)
  score_changePoint(datos)
}

