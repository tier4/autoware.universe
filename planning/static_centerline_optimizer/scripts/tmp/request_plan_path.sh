#!/bin/bash

# curl -H "Content-type: application/json" -X POST -d '{"start_lane_id": 125}' localhost:4010/get_planned_path
# curl -X GET -sS 'localhost:4010/planned_path?start_lane_id=125'
curl -X GET -sS -b cokie.txt "localhost:4010/planned_path?route[]=125&route[]=51&route[]=112&route[]=20&route[]=114&route[]=26&route[]=132"
