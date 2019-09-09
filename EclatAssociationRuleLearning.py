# Eclat Association Rule Learning

# Import the required libraries and the collected dataset
import pandas as pd
collectedDataset = pd.read_csv('OnlineGroceryStore.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([x for x in collectedDataset.values[i] if str(x) != 'nan'])

# Train the Apriori on the dataset
# With no min confidence and lift, for Eclat
from apyori import apriori
rules = apriori(transactions, min_support = 0.004, min_confidence = 0, min_lift = 0)

# Visualize the results
eclatResults = list(rules)
# Sort by the support in decending order
eclatResults.sort(key=lambda tup: tup[1], reverse=True)
# Set the min length for the results
min_length = 2
eclatResults_list = []
for i in range(0, len(eclatResults)):
   if len(eclatResults[i][0]) >= min_length:
      eclatResults_list.append('RULE:\t' + str(eclatResults[i][0]) + '\nSUPPORT:\t' + str(eclatResults[i][1]))

