/*
Upload this code to Arduino Mega affixed to 9-axis motion shield.
This will help publish motion shield twist and its orientation to ROS
*/

#include <ros.h>
#include <ros/time.h>
#include "NineAxesMotion.h"
#include <Wire.h>

// import ros messages
//#include <geometry_msgs/QuaternionStamped.h>
//#include <geometry_msgs/TwistStamped.h>

#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Vector3.h>

#include <std_msgs/UInt8.h>


NineAxesMotion mySensor;  //Object that for the sensor

// ros instantiation
ros::NodeHandle nh;
//
//geometry_msgs::QuaternionStamped quat_msg;
//geometry_msgs::TwistStamped twist_msg;


geometry_msgs::Quaternion quat_msg;
geometry_msgs::Vector3 twist_msg;
//geometry_msgs::Vector3 calibration_msg;
std_msgs::UInt8 calibration_msg;

ros::Publisher quat("quat_motion_shield", &quat_msg);
ros::Publisher twist("twist_motion_shield", &twist_msg);
ros::Publisher calibration("calibration_motion_shield", &calibration_msg);

void setup() {
  
  nh.getHardware()->setBaud(115200);
  Serial.begin(115200);           //Initialize the Serial Port to view information on the Serial Monitor

  I2C.begin();                    //Initialize I2C communication to the let the library communicate with the sensor.

  //Sensor Initialization
  mySensor.initSensor();          //The I2C Address can be changed here inside this function in the library

  
  mySensor.setOperationMode(OPERATION_MODE_IMUPLUS);

//  mySensor.setOperationMode(OPERATION_MODE_NDOF);

  nh.initNode();
  nh.advertise(twist);
  nh.advertise(quat);
  nh.advertise(calibration);


}

void loop() {


    quat_msg.x = mySensor.readQuaternion(X_QUAT);
    quat_msg.y = mySensor.readQuaternion(Y_QUAT);
    quat_msg.z = mySensor.readQuaternion(Z_QUAT);
    quat_msg.w = mySensor.readQuaternion(W_QUAT);

    twist_msg.x = mySensor.readGyro(X_AXIS);
    twist_msg.y = mySensor.readGyro(Y_AXIS);
    twist_msg.z = mySensor.readGyro(Z_AXIS);

    calibration_msg.data = mySensor.readAccelCalibStatus();


//    calibration_msg.x = mySensor.readAccelCalibStatus();
//    calibration_msg.y = mySensor.readGyroCalibStatus();
//    calibration_msg.z = mySensor.readMagCalibStatus();


    quat.publish(&quat_msg);
    twist.publish(&twist_msg);
    calibration.publish(&calibration_msg);

    nh.spinOnce();
//    delay(30);
}

