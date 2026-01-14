import kagglehub
import shutil

# 下载 DEAP 数据集（Kaggle 缓存目录）
src = kagglehub.dataset_download("manh123df/deap-dataset")

print("Downloaded to:", src)

# 如果你想拷贝到当前项目目录
dst = "./DEAP"
shutil.copytree(src, dst, dirs_exist_ok=True)
