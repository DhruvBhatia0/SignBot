import torch
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.policies.pi0fast import PI0FASTPolicy
from lerobot.scripts.control_robot import busy_wait
import time

# Load the policy
print("Loading pi0fast_base...")
policy = PI0FASTPolicy.from_pretrained("lerobot/pi0fast_base")
policy.eval()
policy.to("cuda" if torch.cuda.is_available() else "cpu")

# Define robot config (update your USB ports if needed)
robot_config = So100RobotConfig(
    cameras={
        "webcam": {
            "type": "opencv",
            "camera_index": 0,
            "fps": 30,
            "width": 640,
            "height": 480,
        }
    }
)

# Connect the robot
print("Connecting SO100 robot...")
robot = ManipulatorRobot(robot_config)
robot.connect()

# Run inference loop
print("Running inference with prompt: 'Pick up the red pen'")
prompt = "Pick up the red pen"
fps = 30
duration_s = 10

for _ in range(int(duration_s * fps)):
    start_time = time.perf_counter()

    obs = robot.capture_observation()

    # Preprocess images to [0, 1] float32 and make batch dim
    for key in obs:
        if "image" in key:
            img = obs[key].type(torch.float32) / 255
            obs[key] = img.permute(2, 0, 1).unsqueeze(0)
        else:
            obs[key] = obs[key].unsqueeze(0)

        obs[key] = obs[key].to(policy.device)

    obs["task"] = [prompt]  # must be a list of strings

    with torch.no_grad():
        action = policy.select_action(obs).squeeze(0).cpu()

    robot.send_action(action)

    # Wait to keep fps steady
    dt = time.perf_counter() - start_time
    busy_wait(1 / fps - dt)

robot.disconnect()
print("Done.")
