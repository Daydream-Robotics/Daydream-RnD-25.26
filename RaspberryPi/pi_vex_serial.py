import serial
import time
import realtimeDetectCentroidHeatmap as realtimeDetect
import numpy as np
import sys
import os

show_preview = True
if (len(sys.argv) > 1):
    show_preview = False

#VEX brain port '/dev/ttyACM1'
SERIAL_PORT = '/dev/ttyACM1'
BAUD_RATE = 115200

try:
    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUD_RATE,
        timeout=None
    )
    #clear input stream
    # ser.flushInput()
    #print success message to console
    # ser.write('\n'.encode('utf-8'))
    print(f"Serial port {SERIAL_PORT} opened SUCCESSFULLY.")
except serial.SerialException as e:
    #print error exception to console
    print(f"Error opening serial port: {e}")
    exit()

#Assume this method runs your CNN and returns the results
def run_cnn_and_get_output():   
    object_found = [False] * 3
    data_string = ''
    objects = realtimeDetect.step(show_preview=show_preview)

    if objects is None or len(objects) == 0:
        return 'N'

    for i, object in enumerate(objects):
        class_id = int(object['class_id'])
        if class_id != 4:
            if object_found[class_id]:
                continue

        if i > 0:
            data_string += "|"
        data_string += f"{class_id},{object['x']:.2f},{object['y']:.2f},{object['conf']:.2f}"
        object_found[class_id] = True

        print(f"Object: {class_id},{object['x']:.2f},{object['y']:.2f},{object['conf']:.2f}")

    return data_string

def main():
    REQ_CHECK_TIME = 0.5

    while True:
        data_string = run_cnn_and_get_output()        

        pre_req_time = time.time()
        # check for serial messages until REQ_CHEC_TIME has passed
        while not (time.time() - pre_req_time > REQ_CHECK_TIME):
            if ser.in_waiting > 0:
                #read and store '\n' terminated line from input stream
                input_string = ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"VEX: <{input_string}>")

                # checks for request character
                if 'soutA' in input_string:
                    # send data to VEX brain if a request was recieved
                    try:
                        payload = (data_string + '\n').encode('utf-8')
                        ser.write(payload)
                        print(f"{time.time()} Wrote: <{data_string}>")
                    except serial.SerialTimeoutException as e:
                        print(f"Error writing to serial port: {e}")
                    # break from loop
                    break
        
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting program.")
        ser.close()
