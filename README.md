# Automata

## Installation
1. Download the appropriate build for CARLA 0.9.11 (Windows zip or Ubuntu tar.gz) from https://github.com/carla-simulator/carla/releases.  
2. After downloading, navigate to the PythonAPI subdirectory and clone the following repo: https://cseegit.essex.ac.uk/2020_ce903/ce903_team06.git 
3. Prepare the Python dependencies by creating a conda virtual env for Python 3.7 
4. Install the appropriate libraries by running: `pip install -r requirements.txt`

## How to run

1. Run the CARLA server by executing CarlaUE4.sh or CarlaEU4.exe (depending on the OS build), located at the topmost level of the WindowsNoEditor folder. 
2. On a separate terminal, navigate to `WindowsNoEditor/PythonAPI/ce903_team06/automata` 
3. Run the Main.py script with: `python Main.py`
    - -model (-m): To select model number (1-6)
    - -spawn (-s): To select spawn point (0-255)
    
## To generate more training data

1. Run the CARLA server by executing CarlaUE4.sh or CarlaEU4.exe (depending on the OS build), located at the topmost level of the WindowsNoEditor folder. 
2. On a separate terminal, navigate to `WindowsNoEditor/PythonAPI/ce903_team06/automata`
3. Run the Autopilot.py script with: `python Autopilot.py`

## Branches

- `traffic_sign_detection:` Contains experiments for object detection
- `experimento_multivariable_car_control:` Contains more experiments for model development for triple branch car control
- `model_driving:` Contains more experiments for model development for single branch car control
- `object_detection:` Contains experiments for object masking using YOLO
- `traffic_sign:` Contains detection experiments for traffic lights and signs