import pkg_resources
from pathlib import Path


def generate_requirements(output_path='requirements.txt'):
    """生成项目依赖的requirements.txt文件

    Args:
        output_path: 输出文件路径
    """
    try:
        # 获取当前环境中所有已安装的包
        installed_packages = pkg_resources.working_set

        # 生成依赖列表，格式为"包名==版本号"
        requirements = [f"{pkg.key}=={pkg.version}" for pkg in installed_packages]

        # 写入requirements.txt
        with open(output_path, 'w') as f:
            f.write('\n'.join(requirements))

        print(f"已成功生成依赖文件: {output_path}")
        print(f"共导出 {len(requirements)} 个依赖包")

    except Exception as e:
        print(f"生成依赖文件时出错: {e}")


if __name__ == "__main__":
    generate_requirements()