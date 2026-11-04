# 数据目录

- raw/：原始 MAT 与来源、下载时间和 SHA256 清单。
- processed/：将来的预处理缓存，不覆盖原始数据。

原始参考数据已下载至 raw/cylinder_nektar_wake.mat，来源与哈希记录于同目录的 .source.json 文件。
预处理缓存不提交版本控制。程序化加载、采样与加噪接口在 sciml/data.py，留给你实现。
来源：https://github.com/maziarraissi/PINNs/tree/master/main/Data
