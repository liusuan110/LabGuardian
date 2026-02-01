from ultralytics import YOLO
import os

# 1. 路径设置
# 获取当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据集配置文件路径
DATA_YAML = os.path.join(BASE_DIR, 'OneShot_Demo_Dataset', 'data.yaml')

# 2. 加载模型
print("正在加载 YOLOv8-OBB 模型...")
# 使用 nano 版本的 OBB 预训练模型，速度最快
model = YOLO('yolov8n-obb.pt') 

if __name__ == '__main__':
    # 3. 开始训练
    print(f"开始针对 {DATA_YAML} 进行过拟合训练...")
    print("目标：专门优化演示图片的斜向元件识别")

    # 训练参数解释：
    # data: 数据集配置
    # epochs: 100次迭代 (对于单张图片，过拟合需要较多轮次，但因为数据少会很快)
    # imgsz: 640 或 960 (高分辨率有助于细线识别)
    # batch: 1 (数据少)u
    # name: 结果保存的名字 (对应 main.py 里的查找逻辑)
    # device: 0 (GPU) 或 cpu
    results = model.train(
        data=DATA_YAML,
        epochs=150,           # 针对单张图过拟合，多训练几轮
        imgsz=960,            # 这种精细电路图建议用大分辨率
        batch=2,              # 小批量
        name='lab_guardian_oneshot_v1', # 固定名字，方便 main.py 自动读取
        exist_ok=True,        # 允许覆盖旧结果
        device=0,             # 尝试使用 GPU (0)
        patience=0            # 关闭早停，强制跑完
    )

    print("训练完成！")
    print(f"模型已保存至: runs/detect/lab_guardian_oneshot_v1/weights/best.pt")
    print("现在您可以直接运行 main.py，系统将自动加载此专用模型。")
