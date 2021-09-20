START=$(date +%s)

  # do something
  cd /home/<USERNAME>/ComplaintModelSMA
  Rscript ETLer.R /home/<USERNAME>/data filtered > pr_ETLer.log 2>&1
  echo "-------------------- ETLER DONE"
  echo "-------------------------------"
  
  cd /home/<USERNAME>/ComplaintModelSMA/src/models
  conda activate base; python relevance_prediction.py --data /home/<USERNAME>/data/processed_r/ > /home/<USERNAME>/ComplaintModelSMA/pr_relevance_prediction.log 2>&1; conda deactivate
  echo "-------------------- RELEVANCE PREDICTION DONE"
  echo "-------------------------------"

  conda activate base; python sanction_gravity_prediction.py --data /home/<USERNAME>/data/processed_r/ > /home/<USERNAME>/ComplaintModelSMA/pr_sanction_gravity_prediction.log 2>&1; conda deactivate
  echo "-------------------- SG PREDICTION DONE"
  echo "-------------------------------"

  cd /home/<USERNAME>/ComplaintModelSMA
  Rscript prediction_uploader.R > pr_uploader.log 2>&1
  echo "-------------------- UPLOADER DONE"
  echo "-------------------------------"

  # timer
  END=$(date +%s)
  DIFF=$(( $END - $START ))
  echo "XXX predictionCycle $START $DIFF"

  
