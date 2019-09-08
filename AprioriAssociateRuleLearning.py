# Apriori Associate Rule Learning

# Import the required libraries and the collected dataset
import pandas as pd
collectedDataset = pd.read_csv('OnlineGroceryStore.csv', header = None)
onlineTransactions = []
for i in range(0, 7501):
    onlineTransactions.append([str(collectedDataset.values[i,j]) for j in range(0, 20)])

# Train the Apriori algorithm on the dataset
from apyori import apriori
associationRules = apriori(onlineTransactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualize the results
associationRuleResults = list(associationRules)
output =[]
for i in range(0, len(associationRuleResults)):
    output.append(['Rule:\t' + str(associationRuleResults[i][2][0][0]), 'Effect:\t' + str(associationRuleResults[i][2][0][1]),
                       'Support:\t' + str(associationRuleResults[i][1]), 'Confidence:\t' + str(associationRuleResults[i][2][0][2]),
                       'Lift:\t' + str(associationRuleResults[i][2][0][3])])