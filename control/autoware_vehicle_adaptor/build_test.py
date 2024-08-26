from pathlib import Path
import subprocess
package_path = {}
package_path["path"] = str(Path(__file__).parent)
build_cpp_command = (
    "g++ -Ofast -Wall -shared -std=c++23 -fPIC $(python3 -m pybind11 --includes) "
)
build_cpp_command += "-DBUILD_PATH=\\\"" + str(Path(__file__).parent) + "\\\" "
print(build_cpp_command)
build_cpp_command += "autoware_vehicle_adaptor/scripts/utils.cpp "
build_cpp_command += (
    "-o autoware_vehicle_adaptor/scripts/utils$(python3-config --extension-suffix) "
)
build_cpp_command += "-lrt -I/usr/include/eigen3 -lyaml-cpp"
subprocess.run(build_cpp_command, shell=True)