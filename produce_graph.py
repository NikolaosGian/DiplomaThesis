import matplotlib.pyplot as plt
import re

def parse_sw_execution_times(file_path):
    iterations = []
    times = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = re.search(r"Itteration (\d+) Kernel at SW Execution time: ([\d.]+) seconds.", line)
                if match:
                    iteration = int(match.group(1))
                    time = float(match.group(2))
                    iterations.append(iteration)
                    times.append(time)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred while reading '{file_path}': {str(e)}")
    return iterations, times

def parse_hw_execution_times(file_path):
    iteration_times = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = re.search(r"Itteration (\d+) HW Execution time: ([\d.eE-]+) seconds.", line)
                if match:
                    iteration = int(match.group(1))
                    time = float(match.group(2))
                    if iteration not in iteration_times:
                        iteration_times[iteration] = []
                    iteration_times[iteration].append(time)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred while reading '{file_path}': {str(e)}")

    iterations = sorted(iteration_times.keys())
    average_times = [(10000+sum(times)) / len(times) for times in [iteration_times[i] for i in iterations]]
    return iterations, average_times

i5_iterations, i5_times = parse_sw_execution_times('Profilling_info_SW_EXECUTION_MYLAPTOP.txt')

kr260_sw_iterations, kr260_sw_times = parse_sw_execution_times('Profilling_KR260_SW_info.txt')

kr260_hw_iterations, kr260_hw_times = parse_hw_execution_times('Profilling_info_100_ITTERATIONS.txt')

speedup_kr260_hw_vs_i5 = [i5_times[i] / kr260_hw_times[i] for i in range(len(kr260_hw_iterations))]

speedup_kr260_hw_vs_kr260_sw = [kr260_sw_times[i] / kr260_hw_times[i] for i in range(len(kr260_hw_iterations))]

average_speedup_kr260_hw_vs_i5 = sum(speedup_kr260_hw_vs_i5) / len(speedup_kr260_hw_vs_i5)
average_speedup_kr260_hw_vs_kr260_sw = sum(speedup_kr260_hw_vs_kr260_sw) / len(speedup_kr260_hw_vs_kr260_sw)

print(f"Average Speedup of KR260 HW vs I5 SW: {average_speedup_kr260_hw_vs_i5:.2f}")
print(f"Average Speedup of KR260 HW vs KR260 SW: {average_speedup_kr260_hw_vs_kr260_sw:.2f}")

# Function to save individual curve plots
def save_individual_curve_plot(iterations, times, label, filename, color):
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, times, label=label, marker='o', linestyle='-', color=color)
    plt.xlabel('Iteration')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'{label} Execution Times')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

save_individual_curve_plot(i5_iterations, i5_times, 'I5_SW_EXE', 'I5_SW_EXE.png', 'blue')
save_individual_curve_plot(kr260_sw_iterations, kr260_sw_times, 'KR260_SW_EXE', 'KR260_SW_EXE.png', 'green')
save_individual_curve_plot(kr260_hw_iterations, kr260_hw_times, 'KR260_HW_EXE', 'KR260_HW_EXE.png', 'red')

# Combined Execution Times Plot
fig, ax1 = plt.subplots(figsize=(14, 8))

ax1.plot(i5_iterations, i5_times, label='I5_SW_EXE', marker='o', linestyle='-', color='blue', markersize=5, linewidth=2)
ax1.plot(kr260_sw_iterations, kr260_sw_times, label='KR260_SW_EXE', marker='x', linestyle='-', color='green', markersize=5, linewidth=2)
ax1.plot(kr260_hw_iterations, kr260_hw_times, label='KR260_HW_EXE', marker='s', linestyle='-', color='red', markersize=5, linewidth=2)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Execution Time (seconds)')
ax1.set_title('SW vs HW Execution of Kernel')
ax1.legend(loc='upper left')
ax1.grid(True)

plt.tight_layout()
plt.savefig('SW_vs_HW_Execution_of_Kernel_Combined.png', dpi=300)
plt.close()

# Separate Speedup Plots Combined in One Graph
fig, ax2 = plt.subplots(figsize=(14, 8))

ax2.plot(kr260_hw_iterations, speedup_kr260_hw_vs_i5, label='Speedup: KR260_HW vs I5_SW', marker='^', linestyle='--', color='purple', markersize=5, linewidth=2)
ax2.plot(kr260_hw_iterations, speedup_kr260_hw_vs_kr260_sw, label='Speedup: KR260_HW vs KR260_SW', marker='v', linestyle='--', color='orange', markersize=5, linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Speedup')
ax2.set_title('Speedup Comparisons')
ax2.legend(loc='upper right')
ax2.grid(True)

# Display the average speedups on the plot
ax2.text(0.5, 0.7, f'Speedup KR260_HW vs KR260_SW: {average_speedup_kr260_hw_vs_kr260_sw:.2f}', transform=ax2.transAxes, color='orange', fontsize=12)
ax2.text(0.5, 0.65, f'Speedup KR260_HW vs I5_SW: {average_speedup_kr260_hw_vs_i5:.2f}', transform=ax2.transAxes, color='purple', fontsize=12)

plt.tight_layout()
plt.savefig('Speedup_Comparisons.png', dpi=300)
plt.close()



'''
import matplotlib.pyplot as plt
import re

def parse_sw_execution_times(file_path):
    iterations = []
    times = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = re.search(r"Itteration (\d+) Kernel at SW Execution time: ([\d.]+) seconds.", line)
                if match:
                    iteration = int(match.group(1))
                    time = float(match.group(2))
                    iterations.append(iteration)
                    times.append(time)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred while reading '{file_path}': {str(e)}")
    return iterations, times

def parse_hw_execution_times(file_path):
    iteration_times = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = re.search(r"Itteration (\d+) HW Execution time: ([\d.eE-]+) seconds.", line)
                if match:
                    iteration = int(match.group(1))
                    time = float(match.group(2))
                    if iteration not in iteration_times:
                        iteration_times[iteration] = []
                    iteration_times[iteration].append(time)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred while reading '{file_path}': {str(e)}")

    iterations = sorted(iteration_times.keys())
    average_times = [(10000+sum(times)) / len(times) for times in [iteration_times[i] for i in iterations]]
    return iterations, average_times

i5_iterations, i5_times = parse_sw_execution_times('Profilling_info_SW_EXECUTION_MYLAPTOP.txt')

kr260_sw_iterations, kr260_sw_times = parse_sw_execution_times('Profilling_KR260_SW_info.txt')

kr260_hw_iterations, kr260_hw_times = parse_hw_execution_times('Profilling_info_100_ITTERATIONS.txt')

speedup_kr260_hw_vs_i5 = [i5_times[i] / kr260_hw_times[i] for i in range(len(kr260_hw_iterations))]

speedup_kr260_hw_vs_kr260_sw = [kr260_sw_times[i] / kr260_hw_times[i] for i in range(len(kr260_hw_iterations))]

average_speedup_kr260_hw_vs_i5 = sum(speedup_kr260_hw_vs_i5) / len(speedup_kr260_hw_vs_i5)
average_speedup_kr260_hw_vs_kr260_sw = sum(speedup_kr260_hw_vs_kr260_sw) / len(speedup_kr260_hw_vs_kr260_sw)

print(f"Average Speedup of KR260 HW vs I5 SW: {average_speedup_kr260_hw_vs_i5:.2f}")
print(f"Average Speedup of KR260 HW vs KR260 SW: {average_speedup_kr260_hw_vs_kr260_sw:.2f}")

# Function to save individual curve plots
def save_individual_curve_plot(iterations, times, label, filename, color):
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, times, label=label, marker='o', linestyle='-', color=color)
    plt.xlabel('Iteration')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'{label} Execution Times')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

save_individual_curve_plot(i5_iterations, i5_times, 'I5_SW_EXE', 'I5_SW_EXE.png', 'blue')
save_individual_curve_plot(kr260_sw_iterations, kr260_sw_times, 'KR260_SW_EXE', 'KR260_SW_EXE.png', 'green')
save_individual_curve_plot(kr260_hw_iterations, kr260_hw_times, 'KR260_HW_EXE', 'KR260_HW_EXE.png', 'red')

fig, ax1 = plt.subplots(figsize=(14, 8))

ax1.plot(i5_iterations, i5_times, label='I5_SW_EXE', marker='o', linestyle='-', color='blue', markersize=5, linewidth=2)
ax1.plot(kr260_sw_iterations, kr260_sw_times, label='KR260_SW_EXE', marker='x', linestyle='-', color='green', markersize=5, linewidth=2)
ax1.plot(kr260_hw_iterations, kr260_hw_times, label='KR260_HW_EXE', marker='s', linestyle='-', color='red', markersize=5, linewidth=2)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Execution Time (seconds)')
ax1.set_title('SW vs HW Execution of Kernel')
ax1.legend(loc='upper left')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(kr260_hw_iterations, speedup_kr260_hw_vs_i5, label='Speedup: KR260_HW vs I5_SW', marker='^', linestyle='--', color='purple', markersize=5, linewidth=2)
ax2.plot(kr260_hw_iterations, speedup_kr260_hw_vs_kr260_sw, label='Speedup: KR260_HW vs KR260_SW', marker='v', linestyle='--', color='orange', markersize=5, linewidth=2)
ax2.set_ylabel('Speedup')
ax2.legend(loc='upper right')

ax2.text(0.5, 0.8, f'Speedup KR260_HW vs I5_SW: {average_speedup_kr260_hw_vs_i5:.2f}', transform=ax2.transAxes, color='purple', fontsize=12)
ax2.text(0.5, 0.75, f'Speedup KR260_HW vs KR260_SW: {average_speedup_kr260_hw_vs_kr260_sw:.2f}', transform=ax2.transAxes, color='orange', fontsize=12)

plt.tight_layout()
plt.savefig('SW_vs_HW_Execution_of_Kernel_Combined.png', dpi=300)
plt.show()

'''