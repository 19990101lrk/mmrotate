import os
import csv
import matplotlib.pyplot as plt
from matplotlib import rcParams
# 创建两个空列表来存储所有文件的recall和precision数据
all_recalls = []
all_precisions = []
file_names=[]

root = 'E:/lrk/trail/logs/data/PR_CSV/DOTA/'

# 遍历文件夹中的所有CSV文件
for filename in os.listdir(root):
    if filename.endswith('dota_0.5.csv'):
        # 读取CSV文件
        with open(os.path.join(root, filename)) as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # 跳过表头
            recalls = []
            precisions = []
            for row in csv_reader:
                recalls.append(float(row[0]))
                precisions.append(float(row[1]))

            # 将该文件的recall和precision添加到所有文件的列表中
            all_recalls.append(recalls)
            all_precisions.append(precisions)
            file_name = filename.split('.')[0][:-7]
            file_names.append(file_name)

# 绘制PR曲线
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 颜色列表
for i in range(len(all_recalls)):
    # plt.plot(all_recalls[i], all_precisions[i], color=colors[i % len(colors)], label=f'File {i + 1}')
    plt.plot(all_recalls[i], all_precisions[i], color=colors[i % len(colors)], label=file_names[i])
plt.rcParams['font.sans-serif'] = ['Times new Roman']  # 设置全部字体为Euclid
config = {
    "font.family": 'Times new Roman',  # 设置字体类型
    "font.size": 9,
#     "mathtext.fontset":'stix',
}
rcParams.update(config)
plt.ylim(0.825,1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('P_R_Curve')
plt.legend()
plt.savefig(root + 'img/dota_ap50.svg', format='svg', dpi=600, bbox_inches='tight')
plt.show()

