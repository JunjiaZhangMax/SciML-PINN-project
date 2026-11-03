# 配置

generated/ 是 config.suite() 导出的 25 份 JSON 和 manifest.json，尚未运行。
顺序为 known -> reference -> sensitivity。字段见 sciml/config.py 和 docs/DESIGN.md。

可运行 `python -m sciml init-configs --output configs/my_protocol` 另导出一份。
正式实验前冻结配置；不能延长单个条件后将结果与不同预算的运行混在一起。
purpose 区分 production/pilot/smoke，后两种不纳入正式统计。
