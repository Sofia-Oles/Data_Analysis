import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('COVID_19.xlsx', 'Sheet1')

t = pd.pivot_table(df[(df['Gender'] == 'Male (Чоловік)')],

                   index=['Age'],
                   columns=['Gender'],
                   aggfunc='mean')

list_of_temp, list_of_ages = [], []

for i in t.iterrows():
    list_of_ages.append(i[0])
    list_of_temp.append(float(i[1][1]))

list_of_temp.remove(list_of_temp[-1])
list_of_ages.remove(list_of_ages[-1])

print(list_of_temp, list_of_ages)

plt.bar([x for x in list_of_ages], [d for d in list_of_temp],
        width=0.2, color='blue')
plt.ylim(ymin=37)
plt.show()
