#ifndef NAVIGATE_H
#define NAVIGATE_H

#include <webots/types.h>

#include <webots/distance_sensor.h>
#include <webots/motor.h>
#include <webots/robot.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "geometry.h"

//-----PRE-SET
#define NAVI_MAX_METHOD 1
#define NAVI_NORM_METHOD 0

//-------------USEFUL-FUNCTION-----------

#define FIELD_VECTOR_DEBUG_MODE 0

#define WHEEL_RADIUS 0.063                     // [m]
#define DISTANCE_WHEEL_TO_ROBOT_CENTRE 0.227  // [m]
#define MAX_SPEED 10*4                            // [m/s]
#define DEMO_SPEED 5                           // [m/s]
#define OBSTACLE_THRESHOLD 0.20                // [m]

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

double Vmax = WHEEL_RADIUS * MAX_SPEED; // Radius of wheel * rad / second
double Wmax = (double)WHEEL_RADIUS * MAX_SPEED / DISTANCE_WHEEL_TO_ROBOT_CENTRE;

WbDeviceTag l_wheels[4];

//----------------PLAYER_VARIABLE
#define NUM_COMPO_VEC 3 

struct Omni_Vector{
  double vx = 0, vy = 0, vw = 0;

  void assign_xy(Point value){
    vx = value.first;
    vy = value.second;
  }

  void reset(){
    vx = 0; vy = 0; vw = 0;
  }

  void ippai(double X){
    vx *= X; vy *= X;
  }
} component_vector[NUM_COMPO_VEC];
//
double actualSpeed[3] = {0.0, 0.0, 0.0};
double targetSpeed[3] = {0.0, 0.0, 0.0};
double maxAcceleration[3] = {8.0, 8.0, 20.0};


// ENABLE TAGGING
bool keyboard_control = 0;

double check_speed(double speed, double _MAX_) 
{ 
    speed = MIN(speed, _MAX_);
    speed = MAX(speed, -_MAX_);
    return speed;
    // *speed /= TUNE_SECOND;
}

void base_set_speeds(double vx, double vy, double omega) {
  component_vector[0] = {check_speed(vx, Vmax),check_speed(vy, Vmax),check_speed(omega, Wmax)};
  keyboard_control = 1;
}

void base_reset() {
  // cout << "I reset \n";
  component_vector[0] = {0, 0, 0};
  component_vector[1] = {0, 0, 0};
  component_vector[2] = {0, 0, 0};
}

void base_forwards() {
  base_set_speeds(DEMO_SPEED / 2.0, 0, 0);
}

void base_backwards() {
  base_set_speeds(-DEMO_SPEED / 2.0, 0, 0);
}

void base_turn_left() {
  base_set_speeds(0, 0, 3.0 * DEMO_SPEED / 2.0);
}

void base_turn_right() {
  base_set_speeds(0, 0, -3.0 * DEMO_SPEED / 2.0);
}

void base_strafe_left() {
  base_set_speeds(0, DEMO_SPEED / 2.0, 0);
}

void base_strafe_right() {
  base_set_speeds(0, -DEMO_SPEED / 2.0, 0);
}

//-----------------MOTION_PLAN--------------
double delta_speed = 0.000001, small_speed = 0.4;
// NOTE: DIS could be F(dist) => 2 radius and gradient
double CHANGE_DIR_RANGE = 0.2;
double K_p1 = 0.2, K_p2 = 0.1;
double K_d1 = 10, K_d2 = 8;  
double K_ball = 0.4;  
double K_velo = 1, K_angle = 3.5;
// REQUIRED V_w_current
double robot_angle_velo = 0;  
double current_robot_dir = -1;

//-------------------

double loss_angle_velo(double a)  
// {return 1 - tanh(fabs(a));} 
// {return 0.5 + cos(a) / 2;}
// {return 2*(1+-1/(1+exp(-fabs(a))) );} 
// {return cos(a);}
//{return fabs(a);} 
// {return MAX(0, (fabs(a) < 1.005 ? cos(a) : 1 - tanh(fabs(a))) );} 
{return MAX(0, (fabs(a) < 0.65 ? cos(a) : 1 - tanh(fabs(a))) );} 

double loss_angle_rotate()  
// { return MIN(K_angle, K_d2 * robot_angle_velo); }  
// {return MAX(0.3,  1 - K_d1 * robot_angle_velo);}  
{return 1;}
double F_angle(double a)  
// {return (a) * (K_angle - loss_angle_rotate());}  
{return (a) * (K_angle*loss_angle_rotate()) + cos(a)*sin(a) * K_angle; }  
// {return (a)* K_angle*loss_angle_rotate();} 
// {return (a)*K_angle;}  
// { return (a);} 

double old_vx = 0, old_vy = 0;
double v_smoothnees = 1;

void base_apply_speeds(double vx, double vy, double omega) {
  // vx = vx * K_velo * loss_angle_velo( omega ) / WHEEL_RADIUS;
  // vy = vy * K_velo * loss_angle_velo( omega ) / WHEEL_RADIUS;
  // omega = F_angle( omega ) * DISTANCE_WHEEL_TO_ROBOT_CENTRE / WHEEL_RADIUS;

  vx /= WHEEL_RADIUS;
  vy /= WHEEL_RADIUS;
  omega = F_angle(omega) * DISTANCE_WHEEL_TO_ROBOT_CENTRE / WHEEL_RADIUS;
  // omega = omega * DISTANCE_WHEEL_TO_ROBOT_CENTRE * K_angle/ WHEEL_RADIUS;

  cout << "               APPLIED VELO " << vx << ' ' << vy << ' ' << omega << '\n';

  if (fabs(vx - old_vx) > fabs(0.3*old_vx)) vx =  old_vx + (vx - old_vx) * v_smoothnees;
  old_vx = vx;
  if (fabs(vy - old_vy) > fabs(0.3*old_vy)) vy =  old_vy + (vy - old_vy) * v_smoothnees;
  old_vy = vy;

  check_speed(vx, Vmax); check_speed(vy, Vmax); check_speed(omega, Wmax);

  // cos(angle between wheel and movement axis)
  // wb_motor_set_velocity(l_wheels[0], -1/sqrt(2) * vx + 1/sqrt(2) * vy + omega);
                                        // -sin(45) + cos(45)
  // wb_motor_set_velocity(l_wheels[1], -1/sqrt(2) * vx - 1/sqrt(2) * vy + omega);
                                        // -sin(135) + cos(135)
  // wb_motor_set_velocity(l_wheels[2],  1/sqrt(2) * vx - 1/sqrt(2) * vy + omega);
  // wb_motor_set_velocity(l_wheels[3],  1/sqrt(2) * vx + 1/sqrt(2) * vy + omega);


  // wb_motor_set_velocity(l_wheels[0], vx * -sin(current_robot_dir + degToRad(45))  + vy * cos(current_robot_dir +degToRad(45)) + omega+current_robot_dir/WHEEL_RADIUS);
  // wb_motor_set_velocity(l_wheels[1], vx * -sin(current_robot_dir + degToRad(135)) + vy * cos(current_robot_dir+degToRad(135)) + omega+current_robot_dir/WHEEL_RADIUS);
  // wb_motor_set_velocity(l_wheels[2], vx *  sin(current_robot_dir + degToRad(45))  - vy * cos(current_robot_dir +degToRad(45)) + omega+current_robot_dir/WHEEL_RADIUS);
  // wb_motor_set_velocity(l_wheels[3], vx *  sin(current_robot_dir + degToRad(135)) - vy * cos(current_robot_dir+degToRad(135)) + omega+current_robot_dir/WHEEL_RADIUS);


  wb_motor_set_velocity(l_wheels[0], vx * -sin(current_robot_dir + degToRad(45))  + vy * cos(current_robot_dir +degToRad(45)) + omega);
  wb_motor_set_velocity(l_wheels[1], vx * -sin(current_robot_dir + degToRad(135)) + vy * cos(current_robot_dir+degToRad(135)) + omega);
  wb_motor_set_velocity(l_wheels[2], vx *  sin(current_robot_dir + degToRad(45))  - vy * cos(current_robot_dir +degToRad(45)) + omega);
  wb_motor_set_velocity(l_wheels[3], vx *  sin(current_robot_dir + degToRad(135)) - vy * cos(current_robot_dir+degToRad(135)) + omega);
}

void base_accelerate() {
  const double time_step = wb_robot_get_basic_time_step();
  double maxSteps = 0;
  for (int i = 3; i--;) {
    double stepsNeeded = fabs(targetSpeed[i] - actualSpeed[i]);
    stepsNeeded /= maxAcceleration[i] * (time_step / 1000.0);
    if (stepsNeeded > maxSteps)
      maxSteps = stepsNeeded;
  }
  if (maxSteps < 1)
    maxSteps = 1;
  for (int i = 3; i--;) {
    actualSpeed[i] += (targetSpeed[i] - actualSpeed[i]) / maxSteps;
  }
  base_apply_speeds(actualSpeed[0], actualSpeed[1], actualSpeed[2]);
}

void link_wheel(WbDeviceTag w0, WbDeviceTag w1, WbDeviceTag w2, WbDeviceTag w3){
  l_wheels[0] = w0;
  l_wheels[1] = w1;
  l_wheels[2] = w2;
  l_wheels[3] = w3;
} 

double activate_function(double value) { return 0.001+fabs(value)+0.5*value*value+fabs(value)*fabs(value)*fabs(value)/6; }

void normalize_2D(){ // input positive
  double denomi = 0;
  for (int i = 0; i < NUM_COMPO_VEC; i++)
    denomi = denomi + activate_function(length_vector(component_vector[i].vx, component_vector[i].vy));
  if (fabs(denomi) < 0.001) return;
  for (int i = 0; i < NUM_COMPO_VEC; i++){

    double old_len = length_vector(component_vector[i].vx, component_vector[i].vy);
    if (fabs(old_len) < 0.001) continue;
    double new_len = Vmax*activate_function(old_len)/denomi;
    // cout << "FUNCTION G at" << i << " " << old_len << " " << new_len << " orig x y " << component_vector[i].vx << " " << component_vector[i].vy << " with max " << Vmax << '\n';
    component_vector[i].vx *= new_len/old_len;
    component_vector[i].vy *= new_len/old_len;

      // cout << component_vector[i].vx  << ' ' << component_vector[i].vy << ' ' << i << " compo vec \n";
  }
}

void normalize_W() // input mix
{
  double denomi = 0;
  for (int i = 0; i < NUM_COMPO_VEC; i++)
    denomi = denomi + activate_function( component_vector[i].vw);
  if (fabs(denomi) < 0.001) return;
  auto sgn = [] (float x) { return (x > 0) ? 1 : ((x < 0) ? -1 : 0); };
  for (int i = 0; i < NUM_COMPO_VEC; i++){
    if (fabs(component_vector[i].vw) < 0.001) continue;
    component_vector[i].vw = sgn(component_vector[i].vw)*Wmax*activate_function(component_vector[i].vw)/denomi;
            // cout << component_vector[i].vw  << ' ' << i << " compo vec \n";
  }
}

void finalize_speed(){
  // if (FIELD_VECTOR_DEBUG_MODE){

  // }
    // for (int i = 0; i < NUM_COMPO_VEC; i++){
    //   cout << "COMPONENT VEC " << i << " speed xyw " << component_vector[i].vx << " " << component_vector[i].vy << " " << component_vector[i].vw << '\n';
    // }
    
  // double temp_sum_w = 0; 
  auto temp_sum_2D = make_pair(0, 0);
  targetSpeed[2] = 0;
  for (int i = 0; i < NUM_COMPO_VEC; i++){
    temp_sum_2D.first  += component_vector[i].vx;
    temp_sum_2D.second += component_vector[i].vy;
    targetSpeed[2]     += component_vector[i].vw;
  }

  if (NAVI_NORM_METHOD){
    if (fabs(targetSpeed[2]) >= Wmax)        normalize_W();
    if (length_vector(temp_sum_2D.first, temp_sum_2D.second) >= Vmax)         normalize_2D();

    targetSpeed[0] = 0; targetSpeed[1] = 0; targetSpeed[2] = 0;
    for (int i = 0; i < NUM_COMPO_VEC; i++){
      targetSpeed[0]  += component_vector[i].vx;
      targetSpeed[1]  += component_vector[i].vy;
      targetSpeed[2]  += component_vector[i].vw;
    }
    cout << "GOUKEI " << targetSpeed[0] << ' ' << targetSpeed[1] << ' ' << targetSpeed[2] << '\n';

  }
  else if (NAVI_MAX_METHOD){
     targetSpeed[2] = bound_x(targetSpeed[2], -Wmax, Wmax);

    if (length_vector(temp_sum_2D.first, temp_sum_2D.second) >= Vmax)
    {
      double temp_len = length_vector(temp_sum_2D.first, temp_sum_2D.second);
      targetSpeed[0] = Vmax * temp_sum_2D.first / temp_len;
      targetSpeed[1] = Vmax * temp_sum_2D.second / temp_len;
    }
    else
      targetSpeed[0] = temp_sum_2D.first,
      targetSpeed[1] = temp_sum_2D.second;
  }


  // targetSpeed[0] = 0; targetSpeed[1] = 0; targetSpeed[2] = 0;
  // for (int i = 0; i < NUM_COMPO_VEC; i++){
  //   targetSpeed[0]  += component_vector[i].vx;
  //   targetSpeed[1]  += component_vector[i].vy;
  //   targetSpeed[2]  += component_vector[i].vw;
  // }
  // double t_x = targetSpeed[0], t_y = targetSpeed[1], t_a = atan2(t_y, t_x);
  // targetSpeed[0] = 2 * cos(t_a);
  // targetSpeed[1] = 2 * sin(t_a);

  // cout << " ROBOT " << targetSpeed[0] << ' ' << targetSpeed[1] << ' ' << targetSpeed[2] << '\n';
//     for (int i = 0; i < NUM_COMPO_VEC; i++)
//       cout << "AFTER NORMALIZE VEC " << i << " speed xyw " << component_vector[i].vx << " " << component_vector[i].vy << " " << component_vector[i].vw << '\n';
// cout << "GOUKEI " << targetSpeed[0] << ' ' << targetSpeed[1] << ' ' << targetSpeed[2] << '\n';
  // if (FIELD_VECTOR_DEBUG_MODE){
  // }

  // if (!keyboard_control)
    for (int i = 0; i < NUM_COMPO_VEC; i++) component_vector[i] = {0, 0, 0};

}

#endif
