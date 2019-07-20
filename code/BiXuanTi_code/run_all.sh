echo "EDA.py"
python EDA.py
echo "构建train_label.py"
python 构建train_label.py

echo "第一轮特征.py"
python 第一轮特征.py    

echo "高消费地点的消费次数特征提取.py"
python 高消费地点的消费次数特征提取.py

echo "活跃天数特征+高消费地点的消费次数+消费次数除以活跃天数特征提取.py"
python 活跃天数特征+高消费地点的消费次数+消费次数除以活跃天数特征提取.py

echo "大众消费地点的消费次数特征提取.py"
python 大众消费地点的消费次数特征提取.py

echo "学生24小时的每个小时的消费次数除以活跃天数特征.py"
python 学生24小时的每个小时的消费次数除以活跃天数特征.py

echo "学生周末在食堂消费的次数与工作日在食堂消费次数的比值特征.py"
python 学生周末在食堂消费的次数与工作日在食堂消费次数的比值特征.py

echo "寒暑假是否在校特征提取.py"
python 寒暑假是否在校特征提取.py

echo "消费金额在0-10，10-20，20元以上之间的次数除以消费总次数特征提取.py"
python 消费金额在0-10，10-20，20元以上之间的次数除以消费总次数特征提取.py

echo "学生每种消费方式的花费占总消费金额的比例特征提取.py"
python 学生每种消费方式的花费占总消费金额的比例特征提取.py

echo "学生的性别_补卡_班车次数特征提取.py"
python 学生的性别_补卡_班车次数特征提取.py

echo "特征合并.py"
python 特征合并.py

echo "Prob_Stacking.py"
python Prob_Stacking.py