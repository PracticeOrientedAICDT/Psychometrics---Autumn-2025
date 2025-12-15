import pandas as pd

# read the Excel file
df = pd.read_excel("Pyramids/Pyramids_UserResponses.xlsx")

# export to CSV
df.to_csv("Pyramids/raw.csv", index=False)
