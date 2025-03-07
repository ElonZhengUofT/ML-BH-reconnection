import importlib
import os
import pkgutil

def import_submodules(package_name, package_path):
    print(f"Scanning package path: {package_path}")

    for _, module_name, is_pkg in pkgutil.iter_modules(package_path):
        full_module_name = f"{package_name}.{module_name}"
        print(f"Importing module: {full_module_name}")

        module = importlib.import_module(full_module_name)
        globals()[module_name] = module  # 让模块本身可用

        # **新增：将模块内部的所有成员导入全局**
        for attr in dir(module):
            if not attr.startswith("_"):  # 排除私有属性
                globals()[attr] = getattr(module, attr)

        if is_pkg:
            subpackage_path = os.path.join(package_path[0], module_name)
            import_submodules(full_module_name, [subpackage_path])

# 递归导入 `src/` 目录下的所有模块，并自动加载函数和类
package_name = "src"
package_path = [os.path.abspath(os.path.dirname(__file__))]

import_submodules(package_name, package_path)
print("All modules imported successfully!")