import serial
import time
import struct

def send_data(data):#串口发送数据
    if data is not None:
        #data=numbers_process(data)
        ser = serial.Serial("/dev/ttyAMA0", 115200)#set up serial
        ser.write(data)#write data

while True:
    time.sleep(0.1)
    send_data(b'Hello world!')
    data_packet = struct.pack('<2s6f2s', b'\x0D\x0A',0.0, 0.0, 0.0, 3.14, 42, 0.0, b'\x0A\x0D')
    #send_data(data_packet)