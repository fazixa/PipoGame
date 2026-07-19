# Preamble for exporting a clip from the model master (v35.Model):
# - appends the action from the clip's own blend file
#   (env: CLIP_FILE, CLIP_ACTION, CLIP_FPS), renames it "ClipExport",
#   sets the scene fps to the clip's rate
# - forces the ARM chains to FK (ik_fk_switch = 1.0): the Mixamo remap
#   actions animate the FK arm controls and never key the switch, but the
#   master's arms resolve to IK — leaving the hands pinned to static IK
#   targets (the "no arm animation in the app" bug). Legs stay IK: the
#   remaps key the foot IK controllers directly.
# - makes the body's face winding consistent IN MEMORY (the master still
#   carries the mirrored side inside-out on disk; RealityKit backface-culls)
# Run before blender_export.py in the same headless session.
import bpy
import os

path = os.environ["CLIP_FILE"]
action_name = os.environ["CLIP_ACTION"]
with bpy.data.libraries.load(path) as (src, dst):
    assert action_name in src.actions, f"{action_name!r} not in {path}: {src.actions}"
    dst.actions = [action_name]
act = dst.actions[0]
act.name = "ClipExport"
bpy.context.scene.render.fps = int(os.environ["CLIP_FPS"])

rig = bpy.data.objects["rig"]
for side in (".l", ".r"):
    rig.pose.bones["c_hand_ik" + side]["ik_fk_switch"] = 1.0

if bpy.context.mode != "OBJECT":
    bpy.ops.object.mode_set(mode="OBJECT")
body = bpy.data.objects["Cube.006"]
body.hide_set(False)
bpy.context.view_layer.objects.active = body
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode="OBJECT")

print(f"appended {action_name!r} from {os.path.basename(path)} as 'ClipExport' "
      f"@ {bpy.context.scene.render.fps}fps; arms -> FK; normals fixed in memory")
