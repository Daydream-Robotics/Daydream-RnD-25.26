#include "main.h"
#include <cstdlib>//dynamic memory allocation
#include <cstring>//string manipulation AUTO

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
	pros::lcd::set_text(1, "struct_overwrite_TEST_3");

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

//global struct array that will be used to store ObjectData structs
//this represents all detected objects in a single frame
//this array of structs will be overwritten each time a new frame of dected objects is sent



typedef struct ObjectData{
	float confidencevalue;
	int xcenter;
	int ycenter;
	int classid;
	struct ObjectData* next;
}ObjectData;

ObjectData* insert_node(ObjectData* root,ObjectData* node){
	if(root != NULL)
		root->next = node;
	return node;
}//end of insert_node() function

ObjectData* objects_in_frame(){
	//input buffer read to from stdin
	int size = 1200;
	char inputbuffer[size];
	ObjectData* head = NULL;

	char* latestFrame;

	pros::lcd::print(3, "frame captured");
	int counter = 0;
	while (fgets(inputbuffer,sizeof(inputbuffer),stdin) != NULL){
		// std::memcpy(latestFrame, inputbuffer, size * sizeof(char));
		latestFrame = inputbuffer;
		pros::lcd::print(4, std::to_string(counter++).c_str());
		pros::delay(10)
	}
	pros::lcd::print(1, "frame captured");
	
	//initialize root
	ObjectData* root = NULL;
	//initalize node
	ObjectData* node = (ObjectData*)malloc(sizeof(ObjectData));
	if(node != NULL)
		node->next = NULL;
	else
		return NULL;
	//initialize first node mandatory for strtok()

	//tokenize inputbuffer
	char* token = strtok(latestFrame,",|\n");
	//initialize class_id field
	sscanf(token,("%d"),&node->classid);

	token = strtok(NULL,",|\n");
	//initialize xcenter field
	sscanf(token,("%d"),&node->xcenter);

	token = strtok(NULL,",|\n");
	//initialize ycenter field
	sscanf(token,("%d"),&node->ycenter);

	token = strtok(NULL,",|\n");
	//initialize confidencevalue field
	sscanf(token,("%f"),&node->confidencevalue);

	//insert node to linkedlist
	root = insert_node(root,node);

	//store linkedlist head
	head = root;

	while(token != NULL){
		//next node
		token = strtok(NULL,",|\n");

		if(token == NULL)
			break;

		//initalize node
		ObjectData* node = (ObjectData*)malloc(sizeof(ObjectData));
		if(node != NULL)
			node->next = NULL;
		else
			return NULL;

		//initialize class_id field
		sscanf(token,("%d"),&node->classid);

		token = strtok(NULL,",|\n");
		//initialize xcenter field
		sscanf(token,("%d"),&node->xcenter);

		token = strtok(NULL,",|\n");
		//initialize ycenter field
		sscanf(token,("%d"),&node->ycenter);

		token = strtok(NULL,",|\n");
		//initialize confidencevalue field
		sscanf(token,("%f"),&node->confidencevalue);

		//insert node to linkedlist
		root = insert_node(root,node);
	}

	return head;
} //end of objects_in_frame() function

void autonomous() {
	while(1){
		ObjectData* root = objects_in_frame();
		ObjectData* temproot = root;
        int objcount=1;
		pros::lcd::print(2, "Obects Returned");
		//constantly output overwritten linkedlist to pi terminal
		while(root != NULL){
			printf("\n___%d___\n\nClass_id:\t%d\nX Center:\t%d\nY Center:\t%d\nConfidence:\t%0.2f\n"
            ,objcount,root->classid,root->xcenter,root->ycenter,root->confidencevalue);
            objcount++;
			root = root->next;
		}

		//free linked list
		while (temproot != NULL){
			ObjectData* freeroot = temproot;
			temproot = temproot->next;
			free(freeroot);
		}
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
