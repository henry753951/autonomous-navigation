import glob
import re
import traceback

from setuptools import find_namespace_packages, setup
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import CUDAExtension


def parse_requirements(fname="requirements.txt", with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.
    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs
    Returns:
        List[str]: list of requirements items
    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists

    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith("-r "):
            # Allow specifying requirements in other files
            target = line.split(" ")[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {"line": line}
            if line.startswith("-e "):
                info["package"] = line.split("#egg=")[1]
            else:
                # Remove versioning from the package
                pat = "(" + "|".join([">=", "==", ">"]) + ")"
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info["package"] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ";" in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(";"))
                        info["platform_deps"] = platform_deps
                    else:
                        version = rest
                    info["version"] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath) as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info["package"]]
                if with_version and "version" in info:
                    parts.extend(info["version"])
                if not sys.version.startswith("3.4"):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get("platform_deps")
                    if platform_deps is not None:
                        parts.append(";" + platform_deps)
                item = "".join(parts)
                yield item

    packages = list(gen_packages_items())
    print("parsed requirements (%s): %r" % (len(packages), packages))
    return packages


install_requires = parse_requirements()


def get_extensions():
    extensions = []

    op_files = glob.glob("src/models/lane/ops/csrc/*.c*")
    extension = CUDAExtension
    ext_name = "src.models.lane.ops.nms_impl"

    ext_ops = extension(
        name=ext_name,
        sources=op_files,
        extra_compile_args={
            "cxx": [],  # 可選的 C++ 編譯參數
            "nvcc": ["-O2"],  # CUDA 編譯參數
        },
    )

    extensions.append(ext_ops)

    return extensions


class CustomBuildExt(build_ext):
    def run(self):
        try:
            super().run()
        except Exception as e:
            print("Build failed with error:")
            traceback.print_exc()  # 打印详细的错误日志
            raise e  # 重新抛出异常


setup(
    name="clrkd",
    version="1.0",
    packages=find_namespace_packages(include=["src.*"]),
    include_package_data=True,
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    install_requires=install_requires,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
)
