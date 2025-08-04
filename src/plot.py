import matplotlib.pyplot as plt
import numpy as np

# 数据点
x = [32, 128, 512, 1024]
y1 = [60.5, 60.75, 59.2, 59.1]  # 第一条线的y值
y2 = [62.8, 63.3, 63.97, 63.9]  # 第二条线的y值

# 计算y轴的合适范围
ymin = min(min(y1), min(y2))
ymax = max(max(y1), max(y2))
y_margin = (ymax - ymin) * 0.05  # 添加5%的边距

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制两条折线
plt.plot(x, y1, marker='o', color='blue', linewidth=2, markersize=8, label='T5-base')
plt.plot(x, y2, marker='s', color='red', linewidth=2, markersize=8, label='T5-large')

# 添加标题
plt.title('Restaurant', fontsize=14, pad=15)

# 设置x轴为对数刻度
plt.xscale('log', base=2)

# 设置标签
plt.xlabel('Output dimensional', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)

# 设置x轴刻度
plt.xticks(x, x)

# 设置y轴范围，确保数据点不会突出
plt.ylim(ymin - y_margin, ymax + y_margin)

# 添加图例
plt.legend(fontsize=10)

# 去除网格
plt.grid(False)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 保存图形（如果需要的话）
# plt.savefig('line_plot.png', dpi=300, bbox_inches='tight')
