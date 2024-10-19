# note call "export QT_QPA_PLATFORM=offscreen" in the terminal at first
bagnames=(
"mocap_difficult"
"mocap_dark"
"gym"
"outdoor_street"
"mocap_dark_fast"
"indoor_floor"
"mocap_medium"
"outdoor_campus"
"mocap_easy"
)  # Use parentheses and quotes for array initialization

for bag in "${bagnames[@]}"; do  # Correct array referencing
    for i in $(seq 1 1); do  # Use seq to generate a sequence (1 to 4 inclusive)
        cmd="python3 run.py configs/RRXIO/thermal.yaml --only_tracking \
          --input_folder=/media/pi/BackupPlus/jhuai/data/rrxio/irs_rtvi_datasets_2021/$bag --scene=$bag$i"
        echo $cmd
        $cmd
    done
done
