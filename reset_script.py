from CarDataService import CarSocketService, CarData


if __name__ == '__main__':
    # Initialize the car service with a system delay (adjust for performance)
    # system_delay: It is recommended to set 0.1 seconds
    car_service = CarSocketService(system_delay=0.1)
    # Start the car service with the RL process
    car_service.start_with_nothing()

    car_service.wait_for_new_data()
    car_service.send_control(0, 0, 1)
    car_service.wait_for_new_data()
    car_service.send_control(0, 0, 0)
    car_service.wait_for_new_data()
    car_service.send_control(0, 0, 0)
