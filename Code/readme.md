Explains what each piece of code does

# data_clean.ipynb
- Used to transform the NumberRecall data into IRT data ie participant, item and response
- Uses the NumberRecall_UserScores.xlsx dataset.
- The main idea is:
  * Treat each level attempted as one item.
  *  Keep only three columns in the output: participant_id = AccountId, item_id = 1..TotalResponses, response ∈ {0,1} (1 = the participant got that item right (correct), 0 = the participant got that item wrong (incorrect))
  *  Set Level = CorrectResponses if they differ.
- Produces a new dataframe with the right IRT format

  
- **Note: An assumption made is With only aggregates (eg TotalResponses=23, CorrectResponses=20), we don’t know the exact order of right/wrong trials.For a clean IRT-style table, we adopt a deterministic convention: the first 20 item_ids are marked correct (1) and the last 3 are wrong (0)**
  
