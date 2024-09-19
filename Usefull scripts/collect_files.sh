#!/bin/bash
destination="ngiannopoulos/kria_tutorial/Kria_KR260/kr260_custom_platform/phase_eulerian_file_transfer/"
files=(
  "ngiannopoulos/kria_tutorial/Kria_KR260/kr260_custom_platform/phase_eulerian_system_hw_link/Hardware/binary_container_1.xclbin"
  "ngiannopoulos/kria_tutorial/Kria_KR260/kr260_custom_platform/phase_eulerian/Hardware/phase_eulerian"
  "ngiannopoulos/kria_tutorial/Kria_KR260/kr260_custom_platform/phase_eulerian/Hardware/src/event_timer.o"
  "ngiannopoulos/kria_tutorial/Kria_KR260/kr260_custom_platform/phase_eulerian/Hardware/src/main.o"
  "ngiannopoulos/kria_tutorial/Kria_KR260/kr260_custom_platform/phase_eulerian/Hardware/src/xcl2.o"
  "ngiannopoulos/kria_tutorial/Kria_KR260/kr260_custom_platform/dtg_output/dtg_output/kria_kr260/psu_cortexa53_0/device_tree_domain/bsp/pl.dtbo"
)
for file in "${files[@]}"; do
  cp "$file" "$destination"
done
cd "$destination" || exit
sshpass -p 'PASSWORD' sudo scp pl.dtbo binary_container_1.xclbin shell.json phase_eulerian event_timer.o main.o xcl2.o petalinux@IP:/home/petalinux/home/phase_transf_folder/

