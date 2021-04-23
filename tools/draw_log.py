import matplotlib.pyplot as plt
import pandas as pd

path = "../log/TSM_kinetics_RGB_repvggA0_softshift8_blockres_avg_segment8_e50/log.csv"
data = pd.read_csv(path, sep='\t', names=["lr","time","dta","loss","prec_1","prec_5"])

loss_list = []
prec1_list = []

x_range = 0

for i in range(data.shape[0]):
    row = data.iloc[i].values.tolist()
    if pd.isnull(row[0]):
        continue
    elif "Test" in row[0] or "Best" in row[0]:
        continue
    else:
        loss = float(row[3].split()[1])
        prec1 = float(row[4].split()[1])
        loss_list.append(loss)
        prec1_list.append(prec1)
        x_range+=1





#我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
x1 = range(0, x_range)
x2 = range(0, x_range)
y1 = prec1_list
y2 = loss_list
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Test prec1 vs. epoches')
plt.ylabel('Test prec1')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')
plt.show()
plt.savefig("prec1_loss.jpg")