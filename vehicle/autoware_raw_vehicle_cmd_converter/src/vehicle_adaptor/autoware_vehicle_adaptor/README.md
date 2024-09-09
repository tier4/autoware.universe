# インストール

autoware_vehicle_adaptorディレクトリ上で
```bash
pip3 install .
```
2回目以降は
```bash
pip3 install -U .
```
動かなくなったらひとまずこのコマンドは叩いてみてください。

setup.pyの
```
SKIP_PRE_INSTALL_FLAG = True
```
にするとpip3のインストールが速く終わる(C++ファイルの更新がなければこっちでやれば良い)


それぞれのC++ファイルのビルドを試すには
autoware_vehicle_adaptorディレクトリ上で
```bash
python3 build_test.py
python3 build_actuation_map_2d_test.py
```

# python simulatorのテスト

なんとなく試すにはpython_simulatorディレクトリ上で
```bash
python3 run_vehicle_adaptor.py
```
をやる。
パラメータ変更しつつやるには
```bash
python3 run_auto_parameter_change_sim.py --root=time --param_name=steer_scaling
```

などやればできる。
キャリブレーションのテストを動かすには

```bash
python3 run_accel_brake_map_calibrator.py
```

をやる。

pythonファイルの中でシミュレータを動かすには

```python
from data_collection_utils import ControlType
import python_simulator
```

save_dirを定義して

```python
simulator.drive_sim(control_type=ControlType.pp_eight, max_control_time=1500, save_dir=save_dir, max_lateral_accel=0.5)
```

で8の字走行。

```python
simulator.drive_sim(save_dir=save_dir)
```

でノミナルのMPC走行。
python_simulatorにてコントローラ側加速度入力からアクセル・ブレーキ踏み込み量への変換マップ(キャリブレーションにより作成)をずらすには

```python
map_dir = "../actuation_cmd_maps/accel_brake_maps/low_quality_map"
sim_setting_dict = {}
sim_setting_dict["accel_brake_map_control_path"] = map_dir
simulator.perturbed_sim(sim_setting_dict)
```

とする。
python_simulatorにてアクセル・ブレーキ踏み込み量から入力加速度への変換マップ（車両特性）をずらすには

```python
map_dir = "../actuation_cmd_maps/accel_brake_maps/low_quality_map"
sim_setting_dict = {}
sim_setting_dict["accel_brake_map_sim_path"] = map_dir
simulator.perturbed_sim(sim_setting_dict)
```
とする。

map_dirで別のマップを指定すればそのディレクトリにあるアクセル・ブレーキマップを読む。

# キャリブレーション

```python
from autoware_vehicle_adaptor.calibrator import accel_brake_map_calibrator
calibrator = accel_brake_map_calibrator.CalibratorByNeuralNetwork()
```
でキャリブレータの準備。
走行データのディレクトリcsv_dirがあったとき

```python
calibrator.add_data_from_csv(csv_dir)
```

でキャリブレータにデータを追加できる。

マップを保存したいsave_dirを定義した上で
```python
map_accel = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
map_brake = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
map_vel=[0.0,1.39,2.78,4.17,5.56,6.94,8.33,9.72,11.11,12.5,13.89]
calibrator.calibrate_by_NN()
calibrator.save_accel_brake_map_NN(map_vel,map_accel,map_brake,save_dir)
```
によってキャリブレーション結果を保存できる。
NNではなく多項式回帰を用いる場合はdegreeを指定して
```python
calibrator.calibrate_by_polynomial_regression(degree=degree)
calibrator.save_accel_brake_map_poly(map_vel,map_accel,map_brake,save_dir)
```
とすれば良い。


# CSVデータの読み込み

CSVデータの読み込みについて
準備としてヨー角が連続的に変化するようにする関数。
```python
import numpy as np
import scipy.interpolate
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

def yaw_transform(raw_yaw: np.ndarray) -> np.ndarray:
    """Adjust and transform within a period of 2π so that the yaw angle is continuous."""
    transformed_yaw = np.zeros(raw_yaw.shape)
    transformed_yaw[0] = raw_yaw[0]
    for i in range(raw_yaw.shape[0] - 1):
        rotate_num = (raw_yaw[i + 1] - transformed_yaw[i]) // (2 * np.pi)
        if raw_yaw[i + 1] - transformed_yaw[i] - 2 * rotate_num * np.pi < np.pi:
            transformed_yaw[i + 1] = raw_yaw[i + 1] - 2 * rotate_num * np.pi
        else:
            transformed_yaw[i + 1] = raw_yaw[i + 1] - 2 * (rotate_num + 1) * np.pi
    return transformed_yaw
```
CSVファイルが保存されたディレクトリsave_dirがあったとき

```python
kinematic = np.loadtxt(
    dir_name + "/kinematic_state.csv", delimiter=",", usecols=[0, 1, 4, 5, 7, 8, 9, 10, 47]
)
acc_status = np.loadtxt(dir_name + "/acceleration.csv", delimiter=",", usecols=[0, 1, 3])
steer_status = np.loadtxt(
    dir_name + "/steering_status.csv", delimiter=",", usecols=[0, 1, 2]
)
control_cmd = np.loadtxt(
    dir_name + "/control_cmd_orig.csv", delimiter=",", usecols=[0, 1, 8, 16]
)


pose_position_x = kinematic[:, 2]
pose_position_y = kinematic[:, 3]
vel = kinematic[:, 8]
raw_yaw = Rotation.from_quat(kinematic[:, 4:8]).as_euler("xyz")[:, 2]
yaw = yaw_transform(raw_yaw)
acc = acc_status[:, 2]
steer = steer_status[:, 2]

acc_des = control_cmd[:, 3]
steer_des = control_cmd[:, 2]
```
のようにすれば、実現された状態x,y,vel,yaw,acc,steerとコントローラの入力acc_des, steer_desが得られる。
タイムスタンプは1列目(sec)と2列目(nanosec)に保存されている。
例えばkinematicのタイムスタンプの配列は
```python
kinematic_timestamp = kinematic[:, 0] + 1e-9 * kinematic[:, 1]
```
により取得できる。
