# 数据目录

- raw/：原始 MAT 与来源、下载时间和 SHA256 清单。
- processed/：将来的预处理缓存，不覆盖原始数据。

在仓库根目录执行 `python -m sciml download`，即可下载或校验 `raw/cylinder_nektar_wake.mat`。首次下载时会在同目录记录来源与 SHA256。
数据加载、时空抽样与速度加噪由 [data.py](../sciml/data.py) 实现。原始数据和预处理缓存保存在本地。
来源：https://github.com/maziarraissi/PINNs/tree/master/main/Data
