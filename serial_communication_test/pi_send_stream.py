import serial
import time
import realtimeDetect
import numpy as np

#VEX brain port '/dev/ttyACM1'
SERIAL_PORT = '/dev/ttyACM1'
BAUD_RATE = 115200

DEBUG = True

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
    
    
    #retunrs 2d np array that contains detected object
    #class_id, condfidence, x_center, y_center
    objects = realtimeDetect.step(0.25)
    
    
    return objects

def main_loop():
    while True:
        #if input stream contains more than 0 bytes (non-blocking)
        if DEBUG:
            if ser.in_waiting > 0:
                #read and store '\n' terminated line from input stream
                input_string = ser.readline().decode('utf-8').strip()
                print(f"VEX: {input_string}")

        objects = run_cnn_and_get_output()        

        if objects is None:
            continue

        if len(objects) == 0:
            continue

        data_string = ""
        for i, object in enumerate(objects):
            if i > 0:
                data_string += "|"
            data_string += f"{int(object['class_id'])} {int(object['x'])} {int(object['y'])} {object['conf']:.2f}"

        # output to vex
        try:
            payload = (data_string + '\n').encode('utf-8')
            ser.write(payload)
            print(f"Sent: {data_string}")
        except serial.SerialTimeoutException as e:
            #In case write timeout is exceeded
            print(f"Error writing to serial port: {e}")
            #exit if ser.write(payload) fails
            exit()

        #if input stream contains more than 0 bytes (non-blocking)
        # if ser.in_waiting > 0:
            
        #     #read and store '\n' terminated line from input stream
        #     input_string = ser.readline().decode('utf-8').strip()
            

            
        #     if "REQUEST_OBJECT_DATA" in input_string:
        #         print(f"<{input_string}> RECEIVED")
                
        #         #detected object data
        #         np_list = run_cnn_and_get_output()
        #         if np_list is not None:
        #             data_string = f"{len(np_list)}"
        #             payload = (data_string + '\n').encode('utf-8')
        #             #output to vex
        #             try:
        #                 ser.write(payload)
        #                 print(f"Sent: {data_string}")
        #             except serial.SerialTimeoutException as e:
        #                 #In case write timeout is exceeded
        #                 print(f"Error writing to serial port: {e}")
        #                 #exit if ser.write(payload) fails
        #                 exit()
        #             for object in np_list:
        #                 conf = object[1]
        #                 x = object[2]
        #                 y = object[3]
        #                 class_id = object[0]
                        
        #                 data_string = f"{conf:.2f},{x},{y},{int(class_id)}"
                        
        #                 #Append the newline terminator and encode to bytes
        #                 payload = (data_string + '\n').encode('utf-8')
                        
        #                 #output to vex
        #                 try:
        #                     ser.write(payload)
        #                     print(f"Sent: {data_string}")
        #                 except serial.SerialTimeoutException as e:
        #                     #In case write timeout is exceeded
        #                     print(f"Error writing to serial port: {e}")
        #                     #exit if ser.write(payload) fails
        #                     exit()
        #             # 3. CRITICAL SYNCHRONIZATION FIX: CONSUME VEX VERIFICATION OUTPUT
        #             # This block drains the buffer, looking for the VEX's final acknowledgment
        #             # (like "DATA_STREAM_COMPLETE") to ensure the next line read is the next request.
        #             timeout_start = time.time()
        #             print("Waiting for VEX verification output (Draining Buffer)...")
                    
        #             # Wait for up to 0.5 seconds, or until the buffer is empty
        #             while (time.time() - timeout_start < 0.5): 
        #                 if ser.in_waiting > 0:
        #                     try:
        #                         # Read the line sent by the VEX (its acknowledgment/status)
        #                         vex_output = ser.readline().decode('utf-8').strip()
        #                         if vex_output:
        #                             print(f"VEX ACK: {vex_output}")
        #                             if "DATA_STREAM_COMPLETE" in vex_output or "DMA_FAILED" in vex_output:
        #                                 # Exit early if the final signal is received
        #                                 break 
        #                     except serial.SerialTimeoutException:
        #                         # This should not happen with ser.in_waiting check, but good practice
        #                         break 
        #                 else:
        #                     # Sleep briefly if no data is currently available
        #                     time.sleep(0.01)

        #             print("--- Cycle Synchronization complete ---")
        #     else:
        #         # Handle any other output from the V5 (like debug prints)
        #         print(f"Input does not contain REQUEST_OBJECT_DATA it is: {input_string}")

        #small delay may also be needed to keep the loop from hogging CPU resources
        #time.sleep(0.01)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nExiting program.")
        ser.close()
