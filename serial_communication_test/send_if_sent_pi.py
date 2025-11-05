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

# Assume this method runs your CNN and returns the results
def run_cnn_and_get_output():
    
    # returns 2d np array that contains detected object
    # class_id, condfidence, x_center, y_center
    # NOTE: Using a placeholder value for step time
    np_list = realtimeDetect.step(0.25)
    
    return np_list

def main_loop():
    while True:
        # if input stream contains more than 0 bytes (non-blocking)
        if ser.in_waiting > 0:
            
            # read and store '\n' terminated line from input stream
            # This is the expected 'REQUEST_OBJECT_DATA' trigger
            input_string = ser.readline().decode('utf-8').strip()
            
            if "REQUEST_OBJECT_DATA" in input_string:
                print(f"<{input_string}> RECEIVED")
                
                # detected object data
                np_list = run_cnn_and_get_output()
                if np_list is not None:
                    # 1. Send object count
                    data_string = f"{len(np_list)}"
                    payload = (data_string + '\n').encode('utf-8')
                    # output to vex
                    try:
                        ser.write(payload)
                        print(f"Sent: {data_string}")
                    except serial.SerialTimeoutException as e:
                        print(f"Error writing count to serial port: {e}")
                        exit()

                    # 2. Send object details
                    for object in np_list:
                        conf = object[1]
                        x = object[2]
                        y = object[3]
                        class_id = object[0]
                        
                        data_string = f"{conf:.2f},{x},{y},{int(class_id)}"
                        
                        # Append the newline terminator and encode to bytes
                        payload = (data_string + '\n').encode('utf-8')
                        
                        # output to vex
                        try:
                            ser.write(payload)
                            print(f"Sent: {data_string}")
                        except serial.SerialTimeoutException as e:
                            print(f"Error writing object data to serial port: {e}")
                            exit()

                    # 3. CRITICAL SYNCHRONIZATION FIX (CORRECTLY INDENTED)
                    # This block runs ONCE after all data is sent.
                    
                    # Give the VEX a small moment (50ms) to print all its status/ACK messages. 
                    # This MUST be shorter than the VEX's 200ms polling delay.
                    print("Waiting 50ms for VEX ACKs to buffer...")
                    time.sleep(0.05) 
                    
                    # Now, read ALL bytes that have accumulated in the buffer (NON-BLOCKING DRAIN).
                    if ser.in_waiting > 0:
                        try:
                            # Read all available bytes and decode, ignoring errors
                            vex_output_bytes = ser.read(ser.in_waiting)
                            vex_output = vex_output_bytes.decode('utf-8', errors='ignore')
                            
                            # Print the drained data for debugging
                            if vex_output.strip():
                                # The strip removes any leading/trailing whitespace including newlines
                                print(f"VEX ACKs Drained:\n{vex_output.strip()}")

                        except Exception as e:
                            print(f"Error draining buffer: {e}")

                    print("--- Cycle Synchronization complete ---")
                    
            else:
                # Handle any other output from the V5 (like debug prints)
                print(f"Input does not contain REQUEST_OBJECT_DATA it is: {input_string}")

        # small delay may also be needed to keep the loop from hogging CPU resources
        # time.sleep(0.01)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nExiting program.")
        ser.close()