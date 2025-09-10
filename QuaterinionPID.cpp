/* 
Filename: QuaternionPID.cpp
Uses the IMU's Qauternion and PIDs to turn to or by specified degrees. Accurate to within 0.025*
Author: Alexander Nardi
Created: 2/24/25 -> Last update: 9/10/25. Cleaned code and added relative turn
Requires: PROS VEX library
*/
#include "main.h"
#include "liblvgl/llemu.hpp"
#include "okapi/impl/device/controllerUtil.hpp"
#include "okapi/impl/util/timeUtilFactory.hpp"
#include "pros/llemu.hpp"
#include "pros/rtos.hpp"
#include <cmath>

pros::Imu imu(1); //REPLACE 1 HERE WITH PORT OF YOUR IMU

//Gets the Yaw from the IMU's Quaternion
double getYawQuaternion() {
    //Fetch quaternion
	pros::quaternion_s_t qt = imu.get_quaternion();

	//Error fetching quaternion, retry
	if (qt.w == PROS_ERR_F) {
		qt = imu.get_quaternion();
		if (qt.w == PROS_ERR_F) {
			pros::lcd::set_text(1, "ERROR: IMU Quaternion Fetch Failed"); //error remains, abort and return 360
			return 360.0;
		}
	}

	//Extract the yaw from the quaternion
	double yaw = atan2(2 * ((qt.w * qt.z) + (qt.x * qt.y)), 1 - (2 * ((qt.y * qt.y) + (qt.z * qt.z))));

	//Returns yaw in degrees
	return yaw * (180 / M_PI);
}





/*
Calculates the angle needed to turn to ABSOLUTE heading specified.
Absolute heading: Heading relative to starting orientation, NOT current orientation. Starting orientation's 0 degrees is straight forward.
Usage: Known location of objects / Expected location of objects
*/
double calcAbsAngle(double targetAbsoluteAngle) {
    //Get current heading
    double currentYaw = getYawQuaternion();
    if (currentYaw == 360.0) {
		pros::delay(2000);
		pros::lcd::set_text(2, "ERROR: Quaternion Cannot be Fetched");
		return;
	}
    pros::lcd::print(3, "Current Yaw: %lf", currentYaw); //print heading for testing


    // Compute turn amount
    double turnAmount = targetAbsoluteAngle - currentYaw;

    // Calculate shortest turn by normalizing from -180 -> 180
    // EX: if you needed to turn 190*, this normalizes that to -170*
    if (turnAmount > 180) turnAmount -= 360;
    if (turnAmount < -180) turnAmount += 360;

    pros::lcd::print(4, "Target Yaw: %lf | Turn Amt: %lf", targetAbsoluteAngle, turnAmount); //prints target and turn amount for testing
    
    return turnAmount;
}



/*
Uses quaternion to turn to the specified angle.
Use case:
 - Relative Turn: call turnAngleQuat(angle); if angle is 45* will turn BY 45*
 - Absolute Turn: call turnAngleQuat(calcAbsAngle(angle)); if angle is 45* will turn TO 45*
 "drive" variable is name of Chassis object
*/
void turnAngleQuat(double angle) {
    //Get current heading
	double initialYaw = getYawQuaternion();
	if (initialYaw == 360.0) {
		pros::delay(2000);
		pros::lcd::set_text(2, "ERROR: Quaternion Cannot be Fetched");
		return;
	}
	double targetYaw = initialYaw + angle;

	//normalize angles to -180 - 180
	if (targetYaw > 180) targetYaw -= 360;
	if (targetYaw < -180) targetYaw += 360;

	//Custom PID
	//PID constants
    double kP = 0.8;   // Proportional gain (affects how aggressively it turns)
    double kI = 0.001;  // Integral gain (helps correct small errors)
    double kD = 0.15;   // Derivative gain (reduces overshoot)

    //PID components init
    double integral = 0, derivative = 0, prevDiff = 0, turnSpeed = 0;
	double currentYaw = initialYaw; //current bearing
	double diff = targetYaw - currentYaw; // Difference between target bearing and current bearing
	int count = 0;

    // Loop until within 0.1° of target
	while (count <= 10) {
        //get current bearing
        currentYaw = getYawQuaternion();

        // Calculate shortest turn direction and amount; normalize -180 to 180
        diff = targetYaw - currentYaw;
        if (diff > 180) diff -= 360; 
        if (diff < -180) diff += 360;

        //PID equations
		integral += diff * 0.001;
        derivative = (diff - prevDiff) / 0.3;
        turnSpeed = (kP * diff) + (kI * integral) + (kD * derivative);

        // Limite motor speed depending on difference between current bearing and target bearing (further -> more speed, closer -> less speed); Limit: [0.009, 0.19]
        double limSpeed = fmax(0.009, fmin(0.19, fabs(diff) / 225.0));

        //Calculate ideal turnspeed and turn
		turnSpeed = (kP * diff) + (kI * integral) + (kD * derivative);
		turnSpeed = fmax(fmin(turnSpeed, limSpeed), -limSpeed);
		drive->getModel()->left(turnSpeed);
		drive->getModel()->right(-turnSpeed);

        prevDiff = diff; // Store previous error

        //If within threshold for 50 milliseconds break
		if (fabs(diff) < 0.1) {
			count++;
		} else {
			count = 0;
		}

		
        pros::delay(5); // Small delay for PID loop stability
    }

    // Stop motors when target is reached
	pros::lcd::set_text(1, "TURN COMPLETE!");
    drive->getModel()->stop();
}