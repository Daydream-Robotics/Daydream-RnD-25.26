import serial
import time
import realtimeDetectCentroidHeatmap as realtimeDetect
import numpy as np

#VEX brain port '/dev/ttyACM1'
SERIAL_PORT = '/dev/ttyACM1'
BAUD_RATE = 115200

try:
    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUD_RATE,
        timeout=1
    )
    #clear input stream
    ser.flushInput()
    #print success message to console
    print(f"Serial port {SERIAL_PORT} opened SUCCESSFULLY.")
except serial.SerialException as e:
    #print error exception to console
    print(f"Error opening serial port: {e}")
    exit()

#Assume this method runs your CNN and returns the results
def run_cnn_and_get_output():   
    object_found = [False] * 3
    data_string = ''
    objects = realtimeDetect.step(show_preview=True)

    if objects is None or len(objects) == 0:
        return 'N'

    for i, object in enumerate(objects):
        class_id = int(object['class_id'])
        if class_id != 4:
            if object_found[class_id]:
                continue

        if i > 0:
            data_string += "|"
        data_string += f"{class_id},{int(object['x'])},{int(object['y'])},{object['conf']:.2f}"

    return objects

def main():
    REQ_CHECK_TIME = 0.5

    while True:
        data_string = run_cnn_and_get_output()        

        pre_req_time = time.time()
        # check for serial messages until REQ_CHEC_TIME has passed
        while not (time.time() - pre_req_time > REQ_CHECK_TIME):
            if ser.in_waiting > 0:
                #read and store '\n' terminated line from input stream
                input_string = ser.readline().decode('utf-8').strip()
                print(input_string)

                # checks for request character
                if input_string == 'A':
                    # send data to VEX brain if a request was recieved
                    try:
                        payload = (data_string + '\n').encode('utf-8')
                        ser.write(payload)
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
