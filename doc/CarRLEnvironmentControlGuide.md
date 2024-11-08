# Car RL Environment Control Guide

## Action Space

In this RL environment, the car’s actions are controlled by a continuous action space, consisting of:

| Action         | Range                | Description                                                                                                                                                                     |
|----------------|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Steering Angle | -1 to 1              | Controls the direction of the car; -1 for full left, 1 for full right.                                                                                                          |
| Throttle       | -1 to 1              | Controls speed; positive values for forward acceleration, 0 to maintain speed, negative to brake or reverse.                                                                    |
| Reset Trigger  | 0, 1 or upper than 1 | If set to 0, no position reset will be performed. If set to 1, the car will be reset to its initial position. If set to 1 or higher, reset the car to the specified checkpoint. |

---

## Methods to Control the Car

`CarDataService.CarSocketService` provides two primary methods for controlling the car in this RL environment:

1. **`start_with_nothing`** - For manual or Gym-style step-by-step control.
2. **`start_with_RLProcess`** - For automated control via a callback function.

These methods can be used in different scenarios to suit various levels of control over the car's actions.

### 1. `start_with_nothing` - For Manual Control in Gym-like Environments

The `start_with_nothing` method in `CarSocketService` initializes the car’s service and waits for a client connection
without running any continuous decision-making loop. This method provides a customizable setup, allowing you to manage
the car’s control logic manually on a step-by-step basis

#### How It Works:

- `start_with_nothing` initiates the car’s connection service but leaves control fully open for customization. The car’s
  state can be updated as needed by calling step(action) or equivalent functions in your control logic.
- In this approach, the controlling class or script class (
  ex: [here](https://github.com/Bacon9629/PyAutoDriveRL-Env/blob/main/CarRLEnvironment.py#L87)) is responsible for
  managing the car’s actions and states. The environment directly calls `send_control` with each new action, providing
  full control at each time step.
- **`CarSocketService.wait_for_new_data()` is necessary for each step!!!!!**

#### Example Usage in `CarRLEnvironment` (from `CarRLEnvironment.py`):

1. **Initialize the Environment**: The `CarRLEnvironment` initializes the car service and calls `start_with_nothing`.
2. **Control Flow in Custom Steps**: Each action (`[steering_angle, throttle]`) is directly passed to `send_control`
   during each step.

```python
import numpy as np
from CarDataService import CarSocketService

# Only start the car service
car_service = CarSocketService(system_delay=0.1)
car_service.start_with_nothing()

# Control Flow in Custom Steps
while True:
    steering_angle, throttle = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
    reset_trigger = 0  # If set to 0, no position reset will be performed. If set to 1, the car will be reset to its initial position. If set to 1 or higher, reset the car to the specified checkpoint.
    car_service.send_control(steering_angle, throttle, reset_trigger)  # sending data to unity env
    car_service.wait_for_new_data()  # necessary!!!
```

### 2.` start_with_RLProcess` - For Automated Control via Callback

The `start_with_RLProcess` method in `CarSocketService` enables automated control using a callback mechanism.
It allows the RL agent to control the car continuously based on a callback function (`RL_Process`), which the car
service calls in each decision loop iteration.

#### How It Works:

- `start_with_RLProcess` accepts an RL_Process function as a parameter. This function is repeatedly called by the car
  service, which automatically passes the latest car state (`CarData`) to it.
- `RL_Process` takes the car’s observations and returns the **Steering Angle**, **Throttle**, and **Reset status**
  directly. This tuple of values is automatically applied as control inputs to the car.

#### Example Usage in `inference_template.py` and `train_my.py`:

1. **Define `RL_Process`**: This function takes car_data as input, processes observations, computes actions (steering
   and throttle), and optionally triggers a reset. The function then returns a tuple (steering_angle, throttle,
   reset_trigger).
2. **Start the Service**: `start_with_RLProcess` is called with RL_Process as a parameter, allowing the car to be
   continuously controlled by the RL model in an automated loop.

```python
import numpy as np
from CarDataService import CarSocketService, CarData


# Define RL_Process function
def RL_Process(car_data: CarData):
    # obs = extract_observations(car_data)  # Process observations
    steering_angle, throttle = np.random.uniform(-1, 1), np.random.uniform(-1, 1)  # Get actions from model
    reset_trigger = 0  # If set to 0, no position reset will be performed. If set to 1, the car will be reset to its initial position. If set to 1 or higher, reset the car to the specified checkpoint.
    return steering_angle, throttle, reset_trigger  # Return control values


# Start the car service with RL_Process callback
car_service = CarSocketService(system_delay=0.1)
car_service.start_with_RLProcess(RL_Process)
```


