import numpy as np
from omni.isaac.kit import SimulationApp
from omni.isaac.core import World, SimulationContext
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.physics_context import PhysicsContext
from sklearn.metrics import r2_score  # Need to pip install scikit-learn (if not installed)

# Configure the Isaac Sim application
simulation_app = SimulationApp({"headless": False})  # Set to True to run in headless mode

def perturbation_robustness(n_runs=10, gravity_range=(0.7, 1.3), damping_range=(0.05, 0.5), friction_range=(0.1, 0.5), mass_scale_range=(0.8, 1.2)):
    success_count = 0
    for run in range(n_runs):
        # Create the world
        world = World(stage_units_in_meters=1.0)
        simulation_context = SimulationContext()
        
        # Set random gravity (one of the multi-dimensional perturbations)
        g_factor = np.random.uniform(*gravity_range)
        physics_context = PhysicsContext()
        physics_context.set_gravity(value=np.array([0, 0, -9.81 * g_factor]))
        
        # Create a simple pendulum prim (or replace it with your URDF path)
        pendulum_prim_path = "/World/Pendulum"
        create_prim(pendulum_prim_path, "Xform")
        # Add joints and links (simplified: a revolute joint)
        link1 = create_prim(pendulum_prim_path + "/Link1", "Capsule", attributes={"radius": 0.05, "height": 1.0})
        joint = create_prim(pendulum_prim_path + "/Joint", "RevoluteJoint")
        # Set the initial position and direction
        world.scene.add(link1)
        world.scene.add(joint)
        
        # Get the Articulation object
        articulation = Articulation(pendulum_prim_path)
        world.scene.add(articulation)
        
        # Randomizing joint damping and friction (multi-dimensional perturbation)
        damping = np.random.uniform(*damping_range)
        friction = np.random.uniform(*friction_range)
        articulation.set_joint_damping(np.array([damping]))  # Assume a single joint
        articulation.set_joint_friction(np.array([friction]))
        
        # Randomized Quality Scaling (Multi-dimensional Perturbation)
        mass_scale = np.random.uniform(*mass_scale_range)
        for link in articulation.get_links():
            link.set_mass(link.get_mass() * mass_scale)
        
        # Reset the world and run the simulation (simulate 100 steps)
        world.reset()
        joint_positions = []
        joint_torques = []
        for step in range(100):
            world.step(render=True)
            joint_positions.append(articulation.get_joint_positions())
            joint_torques.append(articulation.get_applied_joint_efforts())  # Torque as an example indicator
        
        # Calculate R² as a robustness indicator (example of linear regression between fitted position vs. torque)
        pos = np.array(joint_positions).flatten()
        torq = np.array(joint_torques).flatten()
        if len(pos) > 1 and len(torq) > 1:
            r2 = r2_score(pos, torq)
            if r2 >= 0.9:
                success_count += 1
        
        # Clean up the world
        world.clear()
    
    robustness = success_count / n_runs
    print(f"Robustness under multi-dimensional perturbations: {robustness * 100:.2f}%")
    return robustness >= 0.8

# Run the test
result = perturbation_robustness()
print(f"Test passed: {result}")

# Close the app
simulation_app.close()
