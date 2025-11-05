import serial
import time
import realtimeDetect
import numpy as np

# VEX brain port '/dev/ttyACM1'
SERIAL_PORT = '/dev/ttyACM1'
BAUD_RATE = 115200

try:
    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUD_RATE,
        timeout=1 # Port timeout is still used for the initial readline attempts
    )
    # clear input stream
    ser.flushInput()
    # print success message to console
    print(f"Serial port {SERIAL_PORT} opened SUCCESSFULLY.")
except serial.SerialException as e:
    # print error exception to console
    print(f"Error opening serial port: {e}")
    exit()

def run_cnn_and_get_output():
    
    #returns 2d np array that contains detected object
    #class_id, condfidence, x_center, y_center
    np_list = realtimeDetect.step(0.25)
    
    return np_list

def main_loop():
    while True:
        #if input stream contains more than 0 bytes
        if ser.in_waiting > 0:
            
            #read and store '\n' terminated line from input buffer
            input_string = ser.readline().decode('utf-8').strip()
            
            if "REQUEST_OBJECT_DATA" in input_string:
                print(f"<{input_string}> RECEIVED")
                
                #detected object data
                np_list = run_cnn_and_get_output()
                if np_list is not None:
                    #send object count
                    data_string = f"{len(np_list)}"
                    payload = (data_string + '\n').encode('utf-8')
                    #output to vex
                    try:
                        ser.write(payload)
                        print(f"Sent: {data_string}")
                    except serial.SerialTimeoutException as e:
                        print(f"Error writing count to serial port: {e}")
                        exit()

                    for object in np_list:
                        conf = object[1]
                        x = object[2]
                        y = object[3]
                        class_id = object[0]
                        
                        data_string = f"{conf:.2f},{x},{y},{int(class_id)}"
                        
                        #append the newline terminator and encode to bytes
                        payload = (data_string + '\n').encode('utf-8')
                        
                        #output to vex
                        try:
                            ser.write(payload)
                            print(f"Sent: {data_string}")
                        except serial.SerialTimeoutException as e:
                            print(f"Error writing object data to serial port: {e}")
                            exit()


                    #prep input buffer for vex debugging output 
                    
                    #print all debugging output 
                    #duration must be shorter than VEX's 200ms polling delay
                    print("Waiting 50ms for VEX ACKs to buffer...")
                    time.sleep(0.05) 
                    
                    #read input buffer
                    if ser.in_waiting > 0:
                        try:
                            #read all available bytes and decode, ignoring errors
                            vex_output_bytes = ser.read(ser.in_waiting)
                            vex_output = vex_output_bytes.decode('utf-8', errors='ignore')
                            
                            #print the drained data for debugging
                            if vex_output.strip():
                                #remove leading whtespace
                                print(f"VEX ACKs Drained:\n{vex_output.strip()}")

                        except Exception as e:
                            print(f"Error draining buffer: {e}")

                    print("--- Cycle Synchronization complete ---")
                else:
                    #send object count
                    data_string = f"{0}"
                    payload = (data_string + '\n').encode('utf-8')
                    #output to vex
                    try:
                        ser.write(payload)
                        print(f"Sent: {data_string}")
                    except serial.SerialTimeoutException as e:
                        print(f"Error writing count to serial port: {e}")
                        exit()

                    #Send None object details
                    conf = 0.0
                    x = 0.0
                    y = 0.0
                    class_id = -1
                    
                    data_string = f"{conf:.2f},{x},{y},{class_id}"
                    
                    #append the newline terminator and encode to bytes
                    payload = (data_string + '\n').encode('utf-8')
                    
                    #output to vex
                    try:
                        ser.write(payload)
                        print(f"Sent: {data_string}")
                    except serial.SerialTimeoutException as e:
                        print(f"Error writing object data to serial port: {e}")
                        exit()


                    #prep input buffer for vex debugging output 
                    
                    #print all debugging output 
                    #duration must be shorter than VEX's 200ms polling delay
                    print("Waiting 50ms for VEX ACKs to buffer...")
                    time.sleep(0.05) 
                    
                    #read input buffer
                    if ser.in_waiting > 0:
                        try:
                            #read all available bytes and decode, ignoring errors
                            vex_output_bytes = ser.read(ser.in_waiting)
                            vex_output = vex_output_bytes.decode('utf-8', errors='ignore')
                            
                            #print the drained data for debugging
                            if vex_output.strip():
                                #remove leading whtespace
                                print(f"VEX ACKs Drained:\n{vex_output.strip()}")

                        except Exception as e:
                            print(f"Error draining buffer: {e}")

                    print("--- Cycle Synchronization complete ---")
            else:
                #handle any other output from vex
                print(f"Input does not contain REQUEST_OBJECT_DATA it is: {input_string}")

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nExiting program.")
        ser.close()
