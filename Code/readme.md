Explains what each piece of code does

# data_clean.ipynb
- Used to transform the NumberRecall data into IRT data ie participant, item and response
- Uses the NumberRecall_UserScores.xlsx dataset.
- The main idea is:
  * Treat each level attempted as one item.
  *  Keep only three columns in the output: participant_id = AccountId, item_id = 1..TotalResponses, response ∈ {0,1} (1 = the participant got that item right (correct), 0 = the participant got that item wrong (incorrect))
  
