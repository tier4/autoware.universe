# jari_rosbag_replayer

事前にスクリプト内で

- rosbagへのpath
- rosbagを再生するトリガーとなる自車位置ライン
- rosbagの再生開始時刻

を設定して以下コマンドで実行。隣でuniverseのplanning_simulatorが動いていれば動作開始。

```sh
python3 jari_rosbag_replayer.py
```

rosbagに入っている認識情報に対して、simの車両が動く

## 値の決め方

- rosbagを再生するトリガーとなる自車位置ライン
  - テストデータで定常状態（車速一定、加速度0になった場所）の左右の点をセット。（goal poseとかを置いてtopic echoでpos.x,yを確認するのが楽）
- rosbagの再生開始時刻
  - 再生トリガーラインをセットした後にrosbagを流すと、rosbag内部の自己位置がラインを超えたタイミングでターミナルに「ライン超えたよ！」と印字される。そのときに同時に表示されているros timeをrosbagの再生開始時刻にセットする
  - rosbagの再生開始時刻はrosbag保存時のros timeを入力する必要があるため、再生開始時刻を決めるときのみ以下のコマンドで実行する

```sh
python3 jari_rosbag_replayer.py --ros-args -p use_sim_time:=true
```


## TODO

- perceptionのデータがbaselink基準になっているので、rosbagに入っているTFを使ってmap座標系に変換する必要がある。map座標変換後のデータを使ってこのスクリプトを回さないといけない
- perceptionの真値データ（GNSS）からperceptionのDetection結果を生成する（後回しでok)
- perceptionの取得time stampを使ってるけど、headerのtime stampを使うべきかも？（要確認）

※この部分

```py
while reader.has_next():
    (topic, data, stamp) = reader.read_next()
    msg_type = get_message(type_map[topic])
    msg = deserialize_message(data, msg_type)
    if (topic == perception_topic):
        self.rosbag_objects_data.append((stamp, msg))
```
