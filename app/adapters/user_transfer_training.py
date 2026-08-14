"""
论文原始训练代码适配器模板（预留）。

当前软件界面默认调用 core.transfer_learning.train_and_activate_aefst。
当你提供论文中实际使用的训练代码后，可以把原网络、损失函数、数据划分、
少样本策略和模型保存逻辑封装到本文件，再在 main.py 的 start_training 中替换调用。

建议保持类似接口：

def train_and_export(
    sim_undamaged_path,
    sim_damaged_path,
    actual_undamaged_path,
    actual_damaged_path,
    geometry_path,
    output_dir,
    config,
    progress_callback,
):
    # 运行论文原始AE-FST训练代码
    # 返回可由在线监测页面读取的模型包路径和训练报告
    return {
        "bundle_path": ".../aefst_model_bundle.pt",
        "metadata_path": ".../training_report.json",
    }
"""
