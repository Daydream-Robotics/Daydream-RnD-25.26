#include <cstdlib>//dynamic memory allocation
#include <cstring>//string manipulation AUTO
#include <objectHandler.h>
#include <iostream>

GamePieceData* objects_in_frame(){
	//input buffer read to from stdin
	char inputbuffer[1200];
	GamePieceData* head = NULL;

	if(fgets(inputbuffer,sizeof(inputbuffer),stdin) != NULL){
		//initialize root
		GamePieceData* root = NULL;
		//initalize node
		GamePieceData* node = (GamePieceData*)malloc(sizeof(GamePieceData));
		if(node != NULL)
			node->next = NULL;
		else
			return NULL;
		//initialize first node mandatory for strtok()

		//tokenize inputbuffer
		char* token = strtok(inputbuffer,",|\n");
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
            GamePieceData* node = (GamePieceData*)malloc(sizeof(GamePieceData));
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
	}else{
		//no detected objects due to failed fgets()
		printf("No Objects Detected");
		return NULL;
	}

	return head;
} //end of objects_in_frame() function

void autonomous() {
	while(1){
		GamePieceData* root = objects_in_frame();
		GamePieceData* temproot = root;
        int objcount=1;
		//constantly output overwritten linkedlist to pi terminal
		while(root != NULL){
			printf("\n___%d___\n\nClass_id:\t%d\nX Center:\t%d\nY Center:\t%d\nConfidence:\t%0.2f\n"
            ,objcount,root->classid,root->xcenter,root->ycenter,root->confidencevalue);
            objcount++;
			root = root->next;
		}

		//free linked list
		while (temproot != NULL){
			GamePieceData* freeroot = temproot;
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
