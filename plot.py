import pandas as pd
import matplotlib.pyplot as plt

plt.style.use(['science', 'no-latex'])

# Pie chart
# def autopct_format(values):
#     def my_format(pct):
#         total = sum(values)
#         val = int(round(pct*total/100.0))
#         return '{:.1f}%\n({v:d})'.format(pct, v=val)
#     return my_format

# lgihts = ['Cool Light', 'Warm Light', 'Sun Light']
 
# data = [3720, 2460, 1440]

# fig = plt.figure()
# plt.pie(data, labels = lgihts, autopct=autopct_format(data), startangle=90, colors=['aqua', 'yellow', 'orange'])
# plt.title("ShaDocs Data Distribution", fontsize = 10)

# plt.savefig("pie.eps")
# plt.show()

# Bar chart
test_df = pd.DataFrame({'Dataset': ['Bako','Jung','Kligler','RDSRD','ShaDocs'], 'Text': [81, 87, 300, 540, 3600], 'Picture': [0, 0, 0, 0, 4020]})

p1 = test_df.plot(kind='bar', x='Dataset', rot=0, color=['b', 'orange'], ylim=(0,4200), fontsize=8, width=1)
plt.xlabel("")

for p in p1.containers:
    p1.bar_label(p, label_type='edge', fontsize=5)
plt.title("Quantitative Comparison Across Different Datasets", fontsize = 10)

plt.savefig("bar.eps")

plt.show()