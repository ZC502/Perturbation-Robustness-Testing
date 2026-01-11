import numpy as np
from omni.isaac.kit import SimulationApp
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.prims import define_prim
from pxr import UsdPhysics, PhysxSchema, Gf

# ---------------------------------------------
# Launch Isaac Sim
# ---------------------------------------------
simulation_app = SimulationApp({"headless": False})


# ---------------------------------------------
# Create a pendulum Articulation
# ---------------------------------------------
def create_pendulum(prim_path="/World/Pendulum"):
    stage = get_current_stage()

    # Articulation Root
    root_prim = define_prim(prim_path, "Xform")
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)

    # -------- Base（Fixed Base）--------
    base_path = prim_path + "/Base"
    base_prim = define_prim(base_path, "Cube")
    base_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(0.1, 0.1, 0.1))

    UsdPhysics.RigidBodyAPI.Apply(base_prim)
    UsdPhysics.CollisionAPI.Apply(base_prim)

    # Key: Fixed base
    PhysxSchema.PhysxRigidBodyAPI.Apply(base_prim).CreateKinematicEnabledAttr().Set(True)

    # -------- Link（pendulum rod）--------
    link_path = prim_path + "/Link1"
    link_prim = define_prim(link_path, "Capsule")
    link_prim.GetAttribute("radius").Set(0.05)
    link_prim.GetAttribute("height").Set(1.0)
    link_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(0, 0, -0.5))

    UsdPhysics.RigidBodyAPI.Apply(link_prim)
    UsdPhysics.CollisionAPI.Apply(link_prim)

    mass_api = UsdPhysics.MassAPI.Apply(link_prim)
    mass_api.CreateMassAttr().Set(1.0)

    # -------- Revolute Joint --------
    joint_path = prim_path + "/Joint"
    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateAxisAttr("X")
    joint.CreateBody0Rel().SetTargets([base_path])
    joint.CreateBody1Rel().SetTargets([link_path])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0.5))

    # Limit (prevent numerical divergence)
    joint.CreateLowerLimitAttr(-np.pi)
    joint.CreateUpperLimitAttr(np.pi)

    # -------- Drive（damping）--------
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateStiffnessAttr().Set(0.0)
    drive.CreateDampingAttr().Set(0.0)

    # -------- Joint Friction --------
    physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
    physx_joint.CreateJointFrictionAttr().Set(0.0)

    return link_prim, drive, physx_joint


# ---------------------------------------------
# Single simulation
# ---------------------------------------------
def run_simulation(world, articulation, n_steps=100, initial_pos=0.0):
    articulation.set_joint_positions(np.array([initial_pos]))

    positions = []
    for _ in range(n_steps):
        world.step(render=True)
        positions.append(articulation.get_joint_positions()[0])
    return np.array(positions)


# ---------------------------------------------
# Robustness testing
# ---------------------------------------------
def perturbation_robustness(
    n_runs=10,
    gravity_range=(0.7, 1.3),
    damping_range=(0.05, 0.5),
    friction_range=(0.1, 0.5),
    mass_scale_range=(0.8, 1.2),
    rms_threshold=0.1,
    bound_threshold=10.0,
):
    world = World(stage_units_in_meters=1.0)

    link_prim, drive, physx_joint = create_pendulum()

    articulation = Articulation("/World/Pendulum")
    world.scene.add(articulation)

    # ---------- Reference trajectory ----------
    world.reset()
    world.physics_context.set_gravity(np.array([0, 0, -9.81]))

    drive.GetDampingAttr().Set(0.0)
    physx_joint.GetJointFrictionAttr().Set(0.0)
    UsdPhysics.MassAPI.Get(link_prim.GetStage(), link_prim.GetPath()).GetMassAttr().Set(1.0)

    ref_positions = run_simulation(world, articulation, initial_pos=np.deg2rad(45))

    success = 0

    # ---------- Perturbation test ----------
    for _ in range(n_runs):
        world.reset()
        world.step()  # Synchronization PhysX

        world.physics_context.set_gravity(
            np.array([0, 0, -9.81 * np.random.uniform(*gravity_range)])
        )

        drive.GetDampingAttr().Set(np.random.uniform(*damping_range))
        physx_joint.GetJointFrictionAttr().Set(np.random.uniform(*friction_range))

        mass_scale = np.random.uniform(*mass_scale_range)
        UsdPhysics.MassAPI.Get(link_prim.GetStage(), link_prim.GetPath()).GetMassAttr().Set(
            1.0 * mass_scale
        )

        positions = run_simulation(world, articulation, initial_pos=np.deg2rad(45))

        T = min(len(ref_positions), len(positions))
        rms = np.sqrt(np.mean((positions[:T] - ref_positions[:T]) ** 2))
        bounded = np.all(np.abs(positions) < bound_threshold) and not np.any(np.isnan(positions))

        if rms <= rms_threshold and bounded:
            success += 1

    robustness = success / n_runs
    print(f"[Robustness] Success rate: {robustness * 100:.2f}%")
    return robustness >= 0.8


# ---------------------------------------------
# Run
# ---------------------------------------------
result = perturbation_robustness()
print("Perturbation test passed:", result)

simulation_app.close()
