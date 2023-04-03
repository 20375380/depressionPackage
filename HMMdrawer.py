# 绘制高频图在 target/HMM_high, 低频图在 target/HMM_low, 不考虑特殊
# 发现low频度图似乎有数字重叠的现象，扩大画布大小为10*10

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import operator as op
import traceback


#file_path = 'D:\\AAA_study\\抑郁症小鼠\\大三下学期\\第四周\\storage-8\\mouse.txt'



class HMM_generator():
    def __init__(self,file_path):
        self.file_path = file_path

    def generate(self):
        with open(self.file_path) as f:
            x = f.read()
        x = x.replace('\n', '')

        count = [0] * 11
        for digit in x.split():
            if digit.isdigit() and 0 <= int(digit) <= 10:
                count[int(digit)] += 1
        # count储存了每个字符出现的次数

        markov_matrix = np.zeros([len(count), len(count)])
        countj = 0
        for j in x.split():
            if countj > 0:
                for m in range(len(count)):
                    for n in range(len(count)):
                        if int(i) == m and int(j) == n:
                            markov_matrix[m][n] += 1
                i = j
            else:
                i = j
                countj += 1
        for t in range(len(count)):
            if count[t] != 0:
                markov_matrix[t, :] /= count[t]

        high_markov_matrix_value = np.zeros([len(count), len(count)])
        low_markov_matrix_value = np.zeros([len(count), len(count)])
        markov_matrix_value = np.zeros([len(count),len(count)])
        for i in range(len(count)):
            for j in range(len(count)):
                markov_matrix_value[i][j] = high_markov_matrix_value[i][j] = low_markov_matrix_value[i][j] = round(markov_matrix[i][j], 3)
        for i in range(len(count)):
            for j in range(len(count)):
                if i == j:
                    high_markov_matrix_value[i][j] = low_markov_matrix_value[i][j] = 0
                    # 低频图>0.005 高频图<0.3
                elif high_markov_matrix_value[i][j] < 0.3:
                    high_markov_matrix_value[i][j] = 0
                elif low_markov_matrix_value[i][j]>0.005:
                    low_markov_matrix_value[i][j] = 0
        print(markov_matrix)

        #画图
        behavior = ['climb', 'groomhead', 'laydown', 'lickabdomen', 'lickback', 'lickforpaws', 'licktail', 'rear', 'rearpause',
                    'turn', 'walk']
        fig, ax = plt.subplots()
        markov_matrix = np.array(markov_matrix)
        annot = np.around(markov_matrix, decimals=2)  # 保留两位小数
        for i in range(markov_matrix.shape[0]):
            for j in range(markov_matrix.shape[1]):
                text = ax.text(j, i, annot[i, j], ha="center", va="center", color="w")
        im = ax.imshow(markov_matrix, cmap='Purples', interpolation='nearest')
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(behavior)))
        ax.set_yticks(np.arange(len(behavior)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(behavior)
        ax.set_yticklabels(behavior)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
        cbar = ax.figure.colorbar(im, ax=ax)
        plt.show()

'''
# print(markov_marix)
behavior = ['climb', 'groomhead', 'laydown', 'lickabdomen', 'lickback', 'lickforpaws', 'licktail', 'rear', 'rearpause',
            'turn', 'walk']
# 这里是创建一个画布
fig, ax = plt.subplots(figsize=(12,12))
ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
# 低频Blues 高频Reds 特殊bwr
im_h = ax.imshow(high_markov_matrix_value, cmap="Reds")

# 这里是修改标签
# We want to show all ticks...
ax.set_xticks(np.arange(len(behavior)))
ax.set_yticks(np.arange(len(behavior)))
# ... and label them with the respective list entries
ax.set_xticklabels(behavior)
ax.set_yticklabels(behavior)

# 因为x轴的标签太长了，需要旋转一下，更加好看
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

ax.set_title('Behavior of mice ' + name[:-5] + '_high')
# Loop over data dimensions and create text annotations.
for i in range(len(behavior)):
    for j in range(len(behavior)):
        try:
            text = ax.text(j, i, markov_matrix_value[i, j],
                            ha="center", va="center", color="black")
        except Exception as e:
            print("Error at " + src_file)
            traceback.print_exc()
            command = "copy " + src_file.replace('/','\\') + " " + target_folder.replace('/','\\') + "\\ERROR"
            os.system(command)
            print("")
            exit(1)


fig.tight_layout()
plt.colorbar(im_h)
plt.savefig('./HMM_high/HMM_heat_' + '_high.png')


# 整个重新创建一次图片，太不优雅了！
fig_, ax_ = plt.subplots(figsize=(12,12))
ax_.grid(which="minor", color="w", linestyle='-', linewidth=3)
# 低频Blues 高频Reds 特殊bwr
im_l = ax_.imshow(high_markov_matrix_value, cmap="Blues")

# 这里是修改标签
# We want to show all ticks...
ax_.set_xticks(np.arange(len(behavior)))
ax_.set_yticks(np.arange(len(behavior)))
# ... and label them with the respective list entries
ax_.set_xticklabels(behavior)
ax_.set_yticklabels(behavior)

# 因为x轴的标签太长了，需要旋转一下，更加好看
# Rotate the tick labels and set their alignment.
plt.setp(ax_.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

ax_.set_title('Behavior of mice ' + name[:-5] + '_low')
# Loop over data dimensions and create text annotations.

for i in range(len(behavior)):
    for j in range(len(behavior)):
        try:
            text = ax_.text(j, i, markov_matrix_value[i, j],
                           ha="center", va="center", color="black")
        except:
            print("Error at " + src_file)

fig_.tight_layout()
plt.colorbar(im_l)
plt.savefig(target_folder + '/HMM_low/HMM_heat_' + name[:-5] + '_low.png')
'''