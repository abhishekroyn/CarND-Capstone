import rospy
from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
        accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # TODO: Implement
        #rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        
#Throttle PID
        kp_throttle = 0.3
        ki_throttle = 0.2
        kd_throttle = 0.
        mn_throttle = 0. # minimum throttle values
        mx_throttle = 0.2 #maximum throttle values
        self.throttle_controller = PID(kp_throttle,ki_throttle,kd_throttle,mn_throttle,mx_throttle)

#Steering PID
        #.45,.3,.01 is good at least 1 laps with camera
        kp_steer = .7
        kd_steer = .4
        ki_steer = 0.01
        self.steer_controller = PID(kp_steer,ki_steer,kd_steer,-max_steer_angle,max_steer_angle)
        self.steersum = 0;

#Low Pass Filters
        tau=0.5
        ts = 0.02
        self.vel_lpf = LowPassFilter(tau,ts)
        self.steer_lpf = LowPassFilter(20,5)
        
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        
        self.last_time = rospy.get_time()

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # Return throttle, brake, steer
        
        if not dbw_enabled:
            self.throttle_controller.reset()
            self.steer_controller.reset()
            return 0., 0., 0.
        
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        
        self.last_time = current_time
        steering = self.yaw_controller.get_steering(linear_vel,angular_vel,current_vel)
        
        current_vel = self.vel_lpf.filt(current_vel)
        vel_error = linear_vel - current_vel
        self.last_vel = current_vel
        
        #Apply Throttle PID
        throttle = self.throttle_controller.step(vel_error, sample_time)
        
        #Apply Steering PID
        steering = self.steer_controller.step(steering, sample_time)

        #Apply Steering LPF and sum steering for tuning
        steering = self.steer_lpf.filt(steering)
        self.steersum = steering*steering + self.steersum
        #rospy.logwarn("steersum: {0}".format(self.steersum))
        brake = 0
        
        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            brake = 400
        
        elif throttle < .1 and vel_error <0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius
            
        return throttle, brake, steering
