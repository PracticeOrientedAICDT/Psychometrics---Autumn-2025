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
- **Total Responses = Correct Responses + up to 3 because of the three lives a user has**
  
# NumberRecallMIRT.R

- Using the IRT cleaned data, fits a 3-PL model with MIRT and IRT package in R
- Input: NumberRecall_MIRT_Format.csv
- Produces two files ie items_params.csv and abilities_csv
- Fits a 3-parameter logistic model: discrimination (a), difficulty (b), guessing (c).

- **Note: Removes items that are all 0 or all 1 (cannot be estimated in IRT).**
