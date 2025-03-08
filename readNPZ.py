# read maps_interp.npz

import numpy as np

# 要读取的文件名
filenames = ['smaller_data/data_time_0032_smaller.npz']

for filename in filenames:
    # 使用 np.load 读取 npz 文件
    data = np.load(filename)

    # data.keys() 即为文件中存储的数组名称列表
    print("Arrays contained in the NPZ file:")
    for key in data.keys():
        print(f"  - {key}")

    # 依次查看每个数组的形状 / 类型
    print("\nArray details:")
    for key in data.keys():
        arr = data[key]
        print(f"{key}: shape = {arr.shape}, dtype = {arr.dtype}")

    # 示例：假设里面存储了某个数组叫 'interp_map'
    if 'labels' in data.keys():
        truth = data['labels']
        where_positive = np.where(truth == 1)
        print(f"Positive labels: {len(where_positive[0])}")

    # 如果你想以字典形式存储所有数组
    arrays_dict = {k: data[k] for k in data.keys()}

    # 关闭文件
    data.close()
