import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime

import matplotlib

# 设置非交互式后端（无需GUI，直接保存图片）
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 标签映射（与训练保持一致）
label_map = {
    '0': 0,  # w1_down
    '1': 1,  # m1_up
    '2': 2,  # w2_up_or_down
    '3': 3,  # m2_up_or_down
    '4': 4,  # 震荡、正反V、单边，趋势不明
    '5': 5  # 不确定类别
}

# 类别名称映射
class_names = ['w1_down', 'm1_up', 'w2_up_or_down', 'm2_up_or_down', 'trend_unknown', 'uncertain']


class TwoStreamLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, num_classes=6):
        super().__init__()
        self.price_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.diff_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.confidence_threshold = 0.7

    def forward(self, price_seq, diff_seq, lengths, apply_threshold=True):
        # 价格流
        price_packed = nn.utils.rnn.pack_padded_sequence(
            price_seq, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        price_out, _ = self.price_lstm(price_packed)
        price_out, _ = nn.utils.rnn.pad_packed_sequence(price_out, batch_first=True)
        price_last = price_out[torch.arange(price_out.size(0)), lengths - 1]

        # 差分流
        diff_packed = nn.utils.rnn.pack_padded_sequence(
            diff_seq, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        diff_out, _ = self.diff_lstm(diff_packed)
        diff_out, _ = nn.utils.rnn.pad_packed_sequence(diff_out, batch_first=True)
        diff_last = diff_out[torch.arange(diff_out.size(0)), lengths - 1]

        # 特征融合
        combined = torch.cat((price_last, diff_last), dim=1)
        logits = self.fc(combined)

        if apply_threshold:
            probs = torch.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            uncertain_mask = max_probs < self.confidence_threshold
            preds[uncertain_mask] = 5  # 不确定类别
            return preds, probs
        else:
            return logits


class KLinePredictor:
    def __init__(self, model_path):
        # 初始化模型
        self.model = TwoStreamLSTMModel(
            input_size=1,
            hidden_size=128,
            num_layers=2,
            num_classes=6
        ).to(device)

        # 加载训练好的权重
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        print(f"Loaded model from {model_path}")

    def preprocess_sequence(self, price_list):
        """预处理K线序列：归一化价格，计算并标准化差分序列"""
        # 转换为tensor
        price_seq = torch.tensor(price_list, dtype=torch.float32)

        # 归一化价格序列 (min-max)
        min_val = price_seq.min()
        max_val = price_seq.max()
        if max_val - min_val < 1e-6:
            price_normalized = torch.zeros_like(price_seq)
        else:
            price_normalized = (price_seq - min_val) / (max_val - min_val)

        # 计算差分序列
        diff_seq = torch.diff(price_seq, prepend=price_seq[0].unsqueeze(0))

        # 标准化差分序列 (z-score)
        diff_mean = diff_seq.mean()
        diff_std = diff_seq.std()
        if diff_std < 1e-6:
            diff_normalized = torch.zeros_like(diff_seq)
        else:
            diff_normalized = (diff_seq - diff_mean) / diff_std

        return price_normalized, diff_normalized, len(price_list)

    def predict(self, kline_sequence):
        """
        预测K线序列的形态

        参数:
            kline_sequence (list): K线价格序列，例如 [1.0, 1.1, 1.05, ...]

        返回:
            tuple: (预测类别名称, 置信度, 各类别概率)
        """
        # 预处理序列
        price_seq, diff_seq, seq_len = self.preprocess_sequence(kline_sequence)

        # 创建batch (batch_size=1)
        price_batch = pad_sequence([price_seq], batch_first=True).unsqueeze(-1).to(device)
        diff_batch = pad_sequence([diff_seq], batch_first=True).unsqueeze(-1).to(device)
        lengths = torch.tensor([seq_len], dtype=torch.long).to(device)

        # 模型预测
        with torch.no_grad():
            pred, probs = self.model(price_batch, diff_batch, lengths, apply_threshold=True)

        # 获取结果
        class_idx = pred.item()
        confidence = torch.max(probs).item()
        probabilities = probs.cpu().numpy()[0]

        return class_names[class_idx], confidence, probabilities


def check_kline_trend(test_sequence):
    import os
    # 当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 初始化预测器（指定模型路径）
    MODEL_PATH = os.path.join(current_dir, 'best_two_stream_model_6class.pth')
    predictor = KLinePredictor(MODEL_PATH)
    res = predictor.predict(test_sequence)

    # pattern, confidence, probs = res[0], res[1], res[2]
    return res


def plot_simple_line(data: list, save_path: str = "line_chart.png"):
    """
    极简折线图绘制（直接保存为图片，避免GUI报错）

    参数:
        data: 待绘制的列表数据（如[12, 24, 18, 30, 22]）
        save_path: 图片保存路径（默认"line_chart.png"）
    """
    # 绘制折线图（x轴为索引，y轴为数据值）
    plt.plot(data, marker='o')  # 极简参数：自动生成x轴索引，默认蓝色实线

    # 设置基础标签（可选）
    plt.title("Simple Line Chart")
    plt.xlabel("Index")
    plt.ylabel("Value")

    # 保存图片（关键！不依赖GUI）
    plt.savefig(save_path, dpi=300)
    plt.close()  # 关闭画布释放内存


if __name__ == "__main__":
    sample_data = [2779.0825, 2778.15, 2778.085, 2777.63, 2777.255, 2779.2475, 2779.5550000000003, 2779.0475,
                   2775.8949999999995,
                   2769.6175000000003, 2767.0600000000004, 2764.78, 2760.7275, 2758.665, 2756.9350000000004, 2754.3325,
                   2752.8325,
                   2749.7975, 2755.0250000000005, 2759.1225, 2758.3325000000004, 2757.725, 2759.75, 2759.605, 2757.9525,
                   2759.2025,
                   2761.1949999999997, 2759.835, 2757.7400000000002, 2754.2225000000003, 2753.59, 2754.5175, 2755.39,
                   2756.5150000000003, 2758.5099999999998, 2761.41, 2761.2175, 2761.5825, 2762.76, 2763.1549999999997,
                   2763.1725,
                   2764.42, 2765.225, 2764.9175000000005, 2763.4900000000002, 2762.0699999999997, 2759.2825, 2758.6275,
                   2758.6925,
                   2756.2525, 2756.7575, 2756.4325, 2757.7075, 2760.8100000000004, 2762.6124999999997, 2763.535,
                   2763.4875,
                   2763.9025,
                   2766.63, 2768.435, 2768.8199999999997, 2767.54, 2766.5525, 2765.415, 2765.26, 2765.0575000000003,
                   2766.3199999999997, 2768.6775, 2770.2874999999995, 2771.5099999999998, 2771.0975, 2768.9474999999998,
                   2769.445,
                   2770.9599999999996, 2771.5950000000003, 2771.605, 2769.26, 2770.3175, 2771.0375, 2770.5724999999998,
                   2770.3725000000004, 2769.7050000000004, 2769.215, 2770.9425, 2771.9700000000003, 2772.7174999999997,
                   2773.2574999999997, 2774.5150000000003, 2775.235, 2776.27, 2776.4675, 2775.73, 2775.5125,
                   2776.1175000000003,
                   2776.1075, 2775.8725, 2775.225, 2773.1349999999998, 2772.8075, 2773.8675]

    # 模型识别
    res = check_kline_trend(sample_data)
    pattern, confidence, probs = res[0], res[1], res[2]

    # 绘图观察
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # ✅ 正确
    plot_simple_line(sample_data, "uncertain.png")

    # 4. 打印结果
    print("\n" + "=" * 50)
    print(f"输入序列长度: {len(sample_data)}")
    print(f"预测形态: {pattern} (置信度: {confidence:.2%})")
    print("各类别概率:")
    for i, (class_name, prob) in enumerate(zip(class_names, probs)):
        print(f"  {class_name:<20}: {prob:.2%}")
    print("该形态类似具有W2形态特征和M1形态特征，后续走势不明，判定为不明确是合理的")
    print("=" * 50)

    # m1_up
    sample_data2 = [2540.92, 2543.5299999999997, 2544.9574999999995, 2546.8825, 2549.9925000000003, 2548.2650000000003,
                    2546.6675, 2546.9425, 2547.5899999999997, 2546.125, 2542.965, 2541.0625, 2539.75, 2541.3475,
                    2543.7949999999996, 2545.3725, 2546.42, 2546.39, 2546.365, 2545.3675, 2545.255, 2543.4399999999996,
                    2542.7124999999996, 2543.1075, 2542.4049999999997, 2543.215, 2544.64, 2547.0125, 2546.03, 2544.16,
                    2543.1549999999997, 2543.275, 2544.8575, 2545.9175, 2545.0775, 2544.7025000000003, 2542.65,
                    2542.6625,
                    2543.5800000000004, 2544.95, 2545.69, 2544.6425, 2545.785, 2548.08, 2548.5725, 2548.2974999999997,
                    2547.9325, 2550.4975, 2552.04, 2551.8625, 2553.9975, 2554.38, 2554.445, 2552.7400000000002, 2549.06,
                    2546.5575, 2547.77, 2548.91, 2550.17, 2551.065, 2552.6475, 2551.8849999999998, 2551.8275, 2552.42,
                    2551.6800000000003, 2552.3525, 2554.5825, 2557.175, 2557.735, 2555.6175000000003, 2554.0, 2551.5575,
                    2550.1225000000004, 2550.5125, 2551.8775, 2553.4175, 2553.4175, 2552.2675, 2552.705, 2552.2725,
                    2555.1425, 2557.1775, 2560.5099999999998, 2561.3225, 2561.9025, 2561.9, 2560.2475, 2564.125,
                    2565.9575,
                    2563.94, 2561.7375, 2560.6749999999997, 2559.52, 2559.5625, 2559.7125000000005, 2561.005,
                    2560.1674999999996, 2558.8774999999996, 2558.9475, 2558.17]

    plot_simple_line(sample_data2, "m1_up.png")

    res = check_kline_trend(sample_data)
    pattern, confidence, probs = res[0], res[1], res[2]

    print("\n" + "=" * 50)
    print(f"输入序列长度: {len(sample_data)}")
    print(f"预测形态: {pattern} (置信度: {confidence:.2%})")
    print("各类别概率:")
    for i, (class_name, prob) in enumerate(zip(class_names, probs)):
        print(f"  {class_name:<20}: {prob:.2%}")
    print("=" * 50)
