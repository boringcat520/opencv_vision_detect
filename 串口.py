import serial



def send_data(data):#串口发送数据
    if data is not None:
        #data=numbers_process(data)
        ser = serial.Serial("/dev/ttyAMA0", 115200)#set up serial
        ser.write(data.encode('utf-8'))#write data

send_data('1')