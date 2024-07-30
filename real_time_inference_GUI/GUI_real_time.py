import os
import asyncio
import numpy as np
import sys
import torch
from bleak import BleakClient, BleakScanner
import argparse
import threading
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import sys
import time
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import signal
import pandas as pd
import time
import pickle as pkl

recognition_record, time_record = [], []
# Choice labels
choice = ['sitting', 'fall', 'sit_down', 'stand up', 'walking', 'walking_stairs']
stop_flag = False

start_time = None
relative_times = []

# Tkinter GUI setup
win = tk.Tk()
win.title("即時動作預測")
win.geometry("800x700")
win.minsize(width = 1200, height = 750)
win.attributes("-topmost", 1)

# 滑動調界面
text_area = ScrolledText(win, wrap=tk.WORD, width=50, height=5)
text_area.pack(padx=10, pady=10)

text_area.tag_configure("message", font=("Helvetica", 10))


#====ploting on GUI =====
fig = Figure(figsize=(8, 4), dpi=100)
ax = fig.add_subplot(111)
lines = [ax.plot([], [], lw=2, label=label)[0] for label in ['acc-x', 'acc-y', 'acc-z']] 
ax.set_xlim(0, 100)
ax.set_ylim(-1, 1)
ax.grid()
ax.legend()

canvas = FigureCanvasTkAgg(fig, master=win)
canvas.draw()
canvas.get_tk_widget().pack(fill=tk.BOTH, padx=5, pady=5)



def update_plot(data, x_dim = 0, continue_running = True):
    if continue_running:
        data = np.squeeze(data, axis = 0)

        for i in range(3):  # 假設有三個通道
            lines[i].set_ydata(data[:, i])
            lines[i].set_xdata(np.arange(len(data)))
        ax.set_xlim(0, len(data))
        ax.set_ylim(np.min(data), np.max(data))
        canvas.draw()
    else:
        for i in range(3):  # 假設有三個通道
            lines[i].set_ydata(data[:, i])
            lines[i].set_xdata(x_dim)
        ax.set_xlim(np.min(x_dim), np.max(x_dim))
        ax.set_ylim(np.min(data), np.max(data))
        ax.set_xlabel('Seconds')  # 设置横轴标签为秒
        ax.set_ylabel('acc')
        ax.grid()
        canvas.draw()
        with open('./data.pkl', 'wb') as f:
            pkl.dump(data, f)

# 在GUI上顯示
def update_gui(message):
    text_area.insert(tk.END, message + "\n", "message")
    text_area.see(tk.END)

def store_state(recognition_record, time_record):
    
    # 字典定义
    dictionary = {
        'time': time_record,
        'actions': recognition_record
    }

    # 将字典转换为DataFrame
    df = pd.DataFrame(dictionary)
    df['time'] = pd.to_datetime(df['time'])

    # 存储到Excel
    df.to_excel('output.xlsx', index=False)

def draw_figure():
    global collect_for_drawing, relative_times, time_record , recognition_record
    # x = np.arange(int(collect_for_drawing.shape[0]))
    x = np.array(relative_times)
    print('================================================================')
    print(x.shape)
    print(collect_for_drawing.shape)
    print('================================================================')
    # 绘制每个通道的数据
    plt.plot(x, collect_for_drawing[:, 0], label='axis-x')
    plt.plot(x, collect_for_drawing[:, 1], label='axis-y')
    plt.plot(x, collect_for_drawing[:, 2], label='axis-z')

    # 添加标题和标签
    plt.title('History waveform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Waveform')
    # 添加图例
    plt.legend()

    # win.after(1000, win.destroy)
    # 显示图表
    store_state(recognition_record, time_record)
    plt.savefig('record.png')
    plt.close()
    update_plot(collect_for_drawing, x_dim = x, continue_running = False)
    
    

def stop_program():
    global stop_flag
    stop_flag = True
    draw_figure()


# 抓圖片
pic_path = "./activity/"
icons_on = []
icons_off = []
height = 75
width = 75

for i in range(1, 7):
    icon_on = ImageTk.PhotoImage(Image.open(pic_path + f"icon{i}_on.png").resize((width, height)))
    icon_off = ImageTk.PhotoImage(Image.open(pic_path + f"icon{i}_off.png").resize((width, height)))
    icons_on.append(icon_on)
    icons_off.append(icon_off)

# 顯示圖示
frame = tk.Frame(win)
frame.pack(padx=10, pady=10)

# 建立一個frame放入六個label
labels = []
icon_frame = tk.Frame(win)
icon_frame.place(relx=0.85, rely=0.05, anchor = 'ne')  # 調整這裡的位置參數以適應您的需求
for i in range(len(choice)):
    row = i // 3
    col = i % 3
    label = tk.Label(frame, image=icons_on[i])
    label.grid(row=row, column=col, padx=5, pady=5)
    labels.append(label)

argsparser = argparse.ArgumentParser()
argsparser.add_argument('--xsens_num', type=int, default=1)
args = argsparser.parse_args()

address = None
short_payload_characteristic_uuid = None

if args.xsens_num == 1:
    address = 'D4:22:CD:00:38:5A'
    short_payload_characteristic_uuid = '15172003-4947-11e9-8646-d663bd873d93'
elif args.xsens_num == 2:
    address = 'D4:22:CD:00:38:5B'
    short_payload_characteristic_uuid = '15172003-4947-11e9-8646-d663bd873d93'
else:
    address = 'D4:22:CD:00:38:55'
    short_payload_characteristic_uuid = '15172003-4947-11e9-8646-d663bd873d93'

measurement_characteristic_uuid = '15172001-4947-11e9-8646-d663bd873d93'

# Real-time settings
channel = 3
slide_length = 96
batch_size = 8
step_ratio = 0.2
step = int(slide_length * step_ratio)
first = True

torch.manual_seed(0)
get_data = np.empty((0, channel))
all_data = np.empty((0, slide_length, channel))
collect_for_drawing = get_data

# Load model
model = torch.load('best_model.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


payload_modes = {
    "High Fidelity (with mag)": [1, b'\x01'],
    "Extended (Quaternion)": [2, b'\x02'],
    "Complete (Quaternion)": [3, b'\x03'],
    "Orientation (Euler)": [4, b'\x04'], #ok
    "Orientation (Quaternion)": [5, b'\x05'], 
    "Free acceleration": [6, b'\x06'], #OK
    "Extended (Euler)": [7, b'\x07'],
    "Complete (Euler)": [16, b'\x10'],
    "High Fidelity": [17, b'\x11'],
    "Delta quantities (with mag)": [18, b'\x12'],
    "Delta quantities": [19, b'\x13'],
    "Rate quantities (with mag)": [20, b'\x14'],
    "Rate quantities": [21, b'\x15'],
    "Custom mode 1": [22, b'\x16'],
    "Custom mode 2": [23, b'\x17'],
    "Custom mode 3": [24, b'\x18'],
    "Custom mode 4": [25, b'\x19'],
    "Custom mode 5": [26, b'\x1A'],
}

def prediction(output_T, output_F, alpha):
    final_pred = alpha * output_T + (1 - alpha) * output_F
    _, final_pred = torch.max(final_pred, dim=1)
    return final_pred

def real_time(data):
    global recognition_record, time_record
    # update_gui(f'{data.shape}')
    
    with torch.no_grad():
        f_input = np.abs(np.fft.fft(data, axis=1))
        t_input = torch.tensor(data).to(device)
        f_input = torch.tensor(f_input).to(device)
        
        outputs_T, outputs_F, alpha = model(t_input.float(), f_input.float())
        outputs = prediction(outputs_T, outputs_F, alpha)
        
        for idx, predict in enumerate(outputs):
            # print(idx) 
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) #current time

            update_gui(f"Current time: {current_time} action prediction is : {choice[predict]}")
            recognition_record.append(choice[predict]) # 紀錄預測結果
            time_record.append(current_time)

            update_icons(predict) # 對icon

        if data.shape[0] > 0:
            update_plot(data)  # 假設要顯示第一個通道的數據

# 圖示
def update_icons(predict):
    # i是索引
    for i, label in enumerate(labels):
        if i == predict:
            label.config(image=icons_on[i])
        else:
            label.config(image=icons_off[i])

def first_process():
    global get_data, all_data, first

    all_data = np.expand_dims(get_data, axis=0)
    real_time(all_data)
    all_data = all_data[:, step:, :]
    first = False
    get_data = np.empty([0, channel])

def not_first_process():
    global get_data, all_data, first
    changed_shape_GetData = np.expand_dims(get_data, axis=0)
    all_data = np.concatenate([all_data, changed_shape_GetData], axis=1)
    real_time(all_data)
    get_data = np.empty([0, channel])
    all_data = all_data[:, step:, :]

def encode_free_accel_bytes_to_string(bytes_):
    data_segments = np.dtype([
        ('timestamp', np.uint32),
        ('Euler_X', np.float32),
        ('Euler_Y', np.float32),
        ('Euler_Z', np.float32),
        ('FreeAcc_X', np.float32),
        ('FreeAcc_Y', np.float32),
        ('FreeAcc_Z', np.float32),
        ('Ang_X', np.float32),
        ('Ang_Y', np.float32),
        ('Ang_Z', np.float32)])
    formatted_data = np.frombuffer(bytes_, dtype=data_segments)
    return formatted_data

def handle_short_payload_notification(sender, data):
    global get_data, all_data, channel, slide_length, step_ratio, first, collect_for_drawing,stop_flag, timestamps, start_time, relative_times
    record = []
    formatted_data = encode_free_accel_bytes_to_string(data)
    for field_name in formatted_data.dtype.names:
        if channel == 3:
            if field_name != 'timestamp' and ('Ang' not in field_name) and ('Euler' not in field_name):
                record.append(formatted_data[field_name])
        elif channel == 6:
            if field_name != 'timestamp' and ('Ang' not in field_name):
                record.append(formatted_data[field_name])
        elif channel == 9:
            if field_name != 'timestamp':
                record.append(formatted_data[field_name])
    record = np.asarray(record).transpose(1, 0)
    get_data = np.concatenate([record, get_data], axis=0)
    if not stop_flag:
        collect_for_drawing = np.concatenate([collect_for_drawing, record], axis=0)
        if start_time is None:
            start_time = time.time()
        current_time = time.time()
        relative_times.extend([current_time - start_time] * record.shape[0])  # 记录相时间
    
    if first:
        if len(get_data) == slide_length:
            first_process()
    else:
        if len(get_data) == step:
            not_first_process()

# 跑這個程式
async def main(ble_address):
    global stop_flag
    update_gui(f'Looking for Bluetooth LE device at address `{ble_address}`...')
    device = await BleakScanner.find_device_by_address(ble_address, timeout=20.0)
    if device is None:
        update_gui(f'A Bluetooth LE device with the address `{ble_address}` was not found.')
    else:
        update_gui(f'Client found at address: {ble_address}')
        update_gui(f'Connecting...')

        async with BleakClient(device) as client:
            if client.is_connected:
                btn_con.config(bg="green")


            update_gui(f'Client connection = {client.is_connected}')
            update_gui(f'Turning on Short Payload notification at `{short_payload_characteristic_uuid}`...')
            await client.start_notify(short_payload_characteristic_uuid, handle_short_payload_notification)
            update_gui('Notifications turned on.')

            payload_mode_values = payload_modes["Custom mode 1"]
            payload_mode = payload_mode_values[1]
            measurement_default = b'\x01'
            start_measurement = b'\x01'
            full_turnon_payload = measurement_default + start_measurement + payload_mode
            update_gui(f'Setting payload with binary: {full_turnon_payload}')
            await client.write_gatt_char(measurement_characteristic_uuid, full_turnon_payload, True)
            update_gui(f'Streaming turned on.')

            while not stop_flag:
                await asyncio.sleep(0.01)  # Small delay to avoid busy-waiting

            await client.stop_notify(short_payload_characteristic_uuid)
            await client.disconnect()
            update_gui(f'Streaming turned off.')

            # await asyncio.sleep(100.0)
            # update_gui(f'Streaming turned off.')

        btn_con.config(bg=win.cget('bg'))
        update_gui(f'Disconnected from `{ble_address}`')

def run_ble_program():
    asyncio.run(main(address))


btn_stop = tk.Button(win, text="Stop", command=stop_program)
btn_stop.config(width=4, height=3)
btn_stop.pack(side=tk.LEFT, padx=10, pady=20)

btn_con = tk.Button(win, text="Connect", command=lambda: threading.Thread(target=run_ble_program).start())
btn_con.config(width=4, height=3)
btn_con.pack(padx=10, pady=20)

win.mainloop()

