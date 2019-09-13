# Thompson Sampling Reinforcement Learning

# Import the required libraries and the collected dataset
import pandas as pd
import random
import matplotlib.pyplot as plotter
collectedDataset=pd.read_csv('WebsiteAds.csv')

# Implement the Thompson Sampling Algorithm on the collected dataset
NumberOfRounds=10000
NumberOfAds=10
ads_selected=[]
numbers_of_rewards_1=[0]*NumberOfAds
numbers_of_rewards_0=[0]*NumberOfAds
total_reward=0
for n in range(0,NumberOfRounds):
  ad=0
  max_random=0
  for i in range(0,NumberOfAds):
    random_beta=random.betavariate(numbers_of_rewards_1[i]+1,numbers_of_rewards_0[i]+1)
    if random_beta > max_random:
        max_random=random_beta
        ad=i
  ads_selected.append(ad)
  reward=collectedDataset.values[n,ad]
  if reward==1:
    numbers_of_rewards_1[ad]=numbers_of_rewards_1[ad]+1
  else:
    numbers_of_rewards_0[ad]=numbers_of_rewards_0[ad]+1
  total_reward=total_reward+reward
  
# Visualize the Results
plotter.hist(ads_selected)
plotter.title('Histogram of Advertisement selections')
plotter.xlabel('Advertisements')
plotter.ylabel('Number of times each Advertisement was selected')
plotter.show()









    
    



