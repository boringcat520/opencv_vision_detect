import serial
import time
import struct

def send_data(data):#串口发送数据
    if data is not None:
        #data=numbers_process(data)
        ser = serial.Serial("/dev/ttyAMA0", 115200)#set up serial
        ser.write(data)#write data

def receive_data():  # 串口接收数据
    ser = serial.Serial("/dev/ttyAMA0", 115200)  # set up serial

    while True:
        # 获得接收缓冲区字符
        count = ser.inWaiting()  # 获取接收缓冲区字符的字节长度，返回值为整型，可用于判断是否接收到数据
        if count != 0:
            # 读取内容并回显
            recv = ser.read(count)

            return recv
            # 清空接收缓冲区
        ser.flushInput()
        # 必要的软件延时
        time.sleep(0.1)


while True:
    time.sleep(0.7)
    send_data(b'Hello world!')
    data_packet = struct.pack('<2s6f2s', b'\x0D\x0A',0.0, 0.0, 0.0, 3.14, 42, 0.0, b'\x0A\x0D')
    #send_data(data_packet)