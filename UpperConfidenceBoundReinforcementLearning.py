# Upper Confidence Bound Reinforcement Learning

# Import the required libraries and the collected dataset
import pandas as pd
import math
import matplotlib.pyplot as plotter
collectedDataset=pd.read_csv('WebsiteAds.csv')

# Implement the UCB algorithm in the collected dataset

NumberOfRounds=10000
NumberOfAds=10
ads_selected=[]
numbers_of_selections=[0]*NumberOfAds
sums_of_rewards= [0]*NumberOfAds
total_reward=0
for n in range(0,NumberOfRounds):
  ad=0
  max_upper_bound=0
  for i in range(0,NumberOfAds):
    if(numbers_of_selections[i]>0):
        average_reward=sums_of_rewards[i]/numbers_of_selections[i]
        delta_i=math.sqrt(3/2*math.log(n+1)/numbers_of_selections[i])
        upper_bound=average_reward+delta_i
    else:
        upper_bound=1e400
    if upper_bound > max_upper_bound:
        max_upper_bound=upper_bound
        ad=i
  ads_selected.append(ad)
  numbers_of_selections[ad]=numbers_of_selections[ad]+1
  reward=collectedDataset.values[n,ad]
  sums_of_rewards[ad]=sums_of_rewards[ad]+reward
  total_reward=total_reward+reward
  
# Visualize the Results
plotter.hist(ads_selected)
plotter.title('Histogram of Advertisement selections')
plotter.xlabel('Advertisements')
plotter.ylabel('Number of times each Advertisement was selected')
plotter.show()











    
    



