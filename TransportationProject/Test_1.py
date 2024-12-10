import pandas as pd

# 示例数据
data = {
    'hrs': [0, 5, 10, 15, 20, 25, 30],
    'min': [0, 30, 0, 15, 45, 0, 0]
}
PeMS_data = pd.DataFrame(data)

# 设置起始日期
start_date = '2018-01-01'

# 将 hrs 和 min 转换为 timedelta，并加到起始日期上
PeMS_data['datetime'] = pd.to_datetime(start_date) + pd.to_timedelta(PeMS_data['hrs'], unit='h') + pd.to_timedelta(PeMS_data['min'], unit='m')

# 查看结果
print(PeMS_data)