#!/bin/bash
working_dir="/home/petalinux/home/phase_transf_folder"
destination_dir="/lib/firmware/xilinx/phase_eulerian"
cd "$working_dir" || { echo "Failed to change directory to $working_dir"; exit 1; }
mv binary_container_1.xclbin binary_container_1.bin
sudo cp pl.dtbo binary_container_1.bin shell.json phase_eulerian event_timer.o main.o xcl2.o "$destination_dir"
sudo xmutil listapps
sudo xmutil unloadapp
sudo xmutil loadapp phase_eulerian
chmod +x "$working_dir/phase_eulerian"
"$working_dir/phase_eulerian" "$working_dir/binary_container_1.bin"
