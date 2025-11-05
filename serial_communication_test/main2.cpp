#include "main.h"

#include <cstdio>//standard C i/o functions AUTO
#include <cstdlib>//dynamic memory allocation
#include <cstring>//string manipulation AUTO

#include <string>
#include <iostream>

/**
 * A callback function for LLEMU's center button.
 *
 * When this callback is fired, it will toggle line 2 of the LCD text between
 * "I was pressed!" and nothing.
 */
void on_center_button() {
	static bool pressed = false;
	pressed = !pressed;
	if (pressed) {
		pros::lcd::set_text(2, "I was pressed!");
	} else {
		pros::lcd::clear_line(2);
	}
}

/**
 * Runs initialization code. This occurs as soon as the program is started.
 *
 * All other competition modes are blocked by initialize; it is recommended
 * to keep execution time for this mode under a few seconds.
 */
void initialize() {
	pros::lcd::initialize();
	pros::lcd::set_text(1, "COMP_COMM_TEST_1");

	pros::lcd::register_btn1_cb(on_center_button);
}

/**
 * Runs while the robot is in the disabled state of Field Management System or
 * the VEX Competition Switch, following either autonomous or opcontrol. When
 * the robot is enabled, this task will exit.
 */
void disabled() {}

/**
 * Runs after initialize(), and before autonomous when connected to the Field
 * Management System or the VEX Competition Switch. This is intended for
 * competition-specific initialization routines, such as an autonomous selector
 * on the LCD.
 *
 * This task will exit when the robot is enabled and autonomous or opcontrol
 * starts.
 */
void competition_initialize() {}

/**
 * Runs the user autonomous code. This function will be started in its own task
 * with the default priority and stack size whenever the robot is enabled via
 * the Field Management System or the VEX Competition Switch in the autonomous
 * mode. Alternatively, this function may be called in initialize or opcontrol
 * for non-competition testing purposes.
 *
 * If the robot is disabled or communications is lost, the autonomous task
 * will be stopped. Re-enabling the robot will restart the task, not re-start it
 * from where it left off.
 */

int OBJECT_COUNT=0;
#define OBJECT_DATA_FIELDS_COUNT 4

//struct to be initialized by pi output
typedef struct{
	float confidence_value;
	float x_center;
	float y_center;
	int class_id;
}ObjectData;

ObjectData* initialize_objectdata(){
    std::string input;
	std::cin >> input;

	std::cout << input << endl;

	//char array buffer to temporarly store pi output until '\n'
	//buffer array size may be tuned for efficiency
	// char input_stream_buffer[128];

	// //recieved in the pi input stream, used to request detected object data
	// // printf("REQUEST_OBJECT_DATA\n");
	// //immediately output
	// // fflush(stdout);

	// if (fgets(input_stream_buffer, sizeof(buffer), stdin) != NULL) {
    //     // Print the most recently read line
    //     std::cout << "Most recent line from terminal: " << input_stream_buffer;
    // } else {
    //     std::cerr << "Error reading from stdin." << std::endl;
    // }

	// //enter only if fgets succeeds in reading input from the pi and storing it into the input_stream_buffer
	// if(fgets(input_stream_buffer,sizeof(input_stream_buffer),stdin) != NULL){
	// 	//read and store obj_count
	// 	int obj_count_parse = sscanf(input_stream_buffer,"%d",&OBJECT_COUNT);
	// 	if(obj_count_parse == 1){
	// 		if (OBJECT_COUNT <= 0) {
    //         	printf("OBJECT_COUNT_IS_0\n");
    //         	fflush(stdout);
	// 			//if no objects were detected a 0 count was sent by pi
    //         	return NULL;
    //     	}
	// 		printf("OBJECT_COUNT_PARSE_SUCCESS\n");
	// 		fflush(stdout);
	// 	}else{
	// 		printf("OBJECT_COUNT_PARSE_FAILED\n");
	// 		fflush(stdout);
	// 	}

		
		
	// 	//dynamic memory allocation for ObjectData array
	// 	ObjectData* detected_objects=(ObjectData*)calloc(OBJECT_COUNT,sizeof(ObjectData));//NOT FREE
		
	// 	//if dma was successful
	// 	if(detected_objects != NULL){
	// 		//read and store a single detected object confidence_value,x,y,class_id
	// 		for(int i=0;i<OBJECT_COUNT;i++){
	// 			if(fgets(input_stream_buffer,sizeof(input_stream_buffer),stdin) != NULL){
	// 				int csv_data_in = sscanf(input_stream_buffer, "%f,%f,%f,%d",
	// 									&detected_objects[i].confidence_value,
	// 									&detected_objects[i].x_center,
	// 									&detected_objects[i].y_center,
	// 									&detected_objects[i].class_id);
	// 				//success or fail parse acknowledge signal to pi
	// 				if(csv_data_in == 4){
	// 					printf("OBJECT_DATA_PARSE_SUCCESS\n");
	// 					fflush(stdout);
	// 				}else{
	// 					printf("OBJECT_DATA_PARSE_FAILED\n");
	// 					fflush(stdout);
	// 					//unreliable information stored
	// 					free(detected_objects);
	// 					return NULL;
	// 				}
	// 			}else{
	// 				printf("NP_ARRAY_ELEMENT_CSV_FAILED_INPUTSTREAM");
	// 				fflush(stdout);
	// 			}
	// 		}
	// 	}else{
	// 		printf("DMA_FAILED\n");
	// 		fflush(stdout);
	// 		return NULL;
	// 	}
	// 	return detected_objects;
	// }else{
	// 	printf("FAILED_INPUTSTREAM_READ\n");
	// 	fflush(stdout);
	// 	return NULL;
	}
}

void autonomous() {
	ObjectData* detected_obj_arr = initialize_objectdata();
	// Continuous loop to poll the vision system until the robot is disabled or communication fails
    while (true) {
        ObjectData* detected_obj_arr = initialize_objectdata();
        
        // Output results to the VEX terminal (optional, but useful for VEX side debugging)
        // if (detected_obj_arr != NULL) {
        //     printf("\n--- VEX CYCLE START: Found %d Objects ---\n", OBJECT_COUNT);
        //     for (int i = 0; i < OBJECT_COUNT; i++) {
        //         printf("Obj %d: ID=%d, Conf=%.2f, X=%.0f, Y=%.0f\n", 
        //                i, 
        //                detected_obj_arr[i].class_id, 
        //                detected_obj_arr[i].confidence_value, 
        //                detected_obj_arr[i].x_center, 
        //                detected_obj_arr[i].y_center);
        //     }

        //     // CRITICAL STEP: FREE MEMORY AFTER USE
        //     free(detected_obj_arr);
        //     printf("--- VEX CYCLE END: Memory Freed ---\n");
        // } else {
        //     // Log when a cycle yields no data or an error occurred
        //     printf("--- VEX CYCLE END: No Data or Communication Error ---\n");
        // }

        // // Delay to prevent CPU hogging and control the polling rate (e.g., 5 times per second)
        // pros::delay(200); // Wait 200 milliseconds (5 Hz polling rate)
    }
}

/**
 * Runs the operator control code. This function will be started in its own task
 * with the default priority and stack size whenever the robot is enabled via
 * the Field Management System or the VEX Competition Switch in the operator
 * control mode.
 *
 * If no competition control is connected, this function will run immediately
 * following initialize().
 *
 * If the robot is disabled or communications is lost, the
 * operator control task will be stopped. Re-enabling the robot will restart the
 * task, not resume it from where it left off.
 */



void opcontrol() {
	pros::Controller master(pros::E_CONTROLLER_MASTER);
	pros::MotorGroup left_mg({1, -2, 3});    // Creates a motor group with forwards ports 1 & 3 and reversed port 2
	pros::MotorGroup right_mg({-4, 5, -6});  // Creates a motor group with forwards port 5 and reversed ports 4 & 6


	while (true) {
		pros::lcd::print(0, "%d %d %d", (pros::lcd::read_buttons() & LCD_BTN_LEFT) >> 2,
		                 (pros::lcd::read_buttons() & LCD_BTN_CENTER) >> 1,
		                 (pros::lcd::read_buttons() & LCD_BTN_RIGHT) >> 0);  // Prints status of the emulated screen LCDs

		// Arcade control scheme
		int dir = master.get_analog(ANALOG_LEFT_Y);    // Gets amount forward/backward from left joystick
		int turn = master.get_analog(ANALOG_RIGHT_X);  // Gets the turn left/right from right joystick
		left_mg.move(dir - turn);                      // Sets left motor voltage
		right_mg.move(dir + turn);                     // Sets right motor voltage
		pros::delay(20);                               // Run for 20 ms then update
	}
}