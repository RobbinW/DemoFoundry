"""Blender entry point for a calibrated view of a simulated episode."""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector


def pose_matrix(pose):
    matrix = Quaternion((pose[6], *pose[3:6])).to_matrix().to_4x4()
    matrix.translation = Vector(pose[:3])
    return matrix


def robot_visuals(workcell, link_names):
    robot = ET.parse(workcell / "robot/marvin.urdf")
    visuals = []
    for link in robot.findall("link"):
        name = link.get("name")
        for i, visual in enumerate(link.findall("visual")):
            obj = bpy.data.objects[name if i == 0 else f"{name}_visual_{i}"]
            offset = Matrix.Identity(4)
            origin = visual.find("origin")
            if origin is not None:
                offset = (
                    Euler(tuple(map(float, origin.get("rpy", "0 0 0").split())))
                    .to_matrix()
                    .to_4x4()
                )
                offset.translation = Vector(tuple(map(float, origin.get("xyz", "0 0 0").split())))
            visuals.append((obj, link_names.index(name), offset))
    return visuals


def configure_cycles(scene):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 16
    scene.render.use_persistent_data = True
    scene.cycles.device = "CPU"
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if "OPTIX" in {item[0] for item in prefs.get_device_types(bpy.context)}:
        prefs.compute_device_type = "OPTIX"
        prefs.refresh_devices()
        devices = [device for device in prefs.devices if device.type == "OPTIX"]
        if devices:
            for device in prefs.devices:
                device.use = device == devices[0]
            scene.cycles.device = "GPU"
            scene.cycles.denoiser = "OPTIX"
            scene.cycles.denoising_use_gpu = True


def render_view(workcell, run, view, destination):
    scene = bpy.context.scene
    configure_cycles(scene)
    layout = json.loads((run / "scene.json").read_text())
    config = json.loads((workcell / "config/cameras.json").read_text())[view]
    with np.load(run / "states.npz") as states:
        links = states["link_poses"]
        objects = states["object_poses"]
        names = states["link_names"].tolist()
        visuals = robot_visuals(workcell, names)
        cubes = [bpy.data.objects[str(name)] for name in states["object_names"]]

    table = bpy.data.objects["table"]
    for obj, body in [
        (table, "table"),
        (bpy.data.objects["tray_base"], "tray"),
        (bpy.data.objects["tray_rim"], "tray"),
    ]:
        obj.location = layout[body]["center"]
        obj.rotation_euler = (0, 0, layout[body]["yaw"])
    table.dimensions = layout["table"]["size"]
    cube_scales = []
    for obj in cubes:
        edge = layout["objects"][obj.name]["edge_m"]
        extent = np.ptp(np.array(obj.bound_box), axis=0)
        cube_scales.append(Matrix.Diagonal((*map(float, edge / extent), 1)))

    camera = bpy.data.objects[view]
    scene.camera = camera
    w, h = config["width"], config["height"]
    k = np.reshape(config["K"], (3, 3))
    aspect = k[0, 0] / k[1, 1]
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = 1, aspect
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36
    camera.data.lens = k[0, 0] * 36 / w
    camera.data.shift_x = (w / 2 - k[0, 2]) / w
    camera.data.shift_y = (k[1, 2] - h / 2) * aspect / w
    camera.data.clip_start = 0.005
    mount = Matrix(config["T_parent_camera"]) @ Matrix.Diagonal((1, -1, -1, 1))
    parent = names.index(config["parent"])

    def set_pose(current_scene, depsgraph=None):
        index = current_scene.frame_current * 2
        for obj, joint, offset in visuals:
            obj.matrix_world = pose_matrix(links[index, joint]) @ offset
        for obj, pose, scale in zip(cubes, objects[index], cube_scales):
            obj.matrix_world = pose_matrix(pose) @ scale
        camera.matrix_world = pose_matrix(links[index, parent]) @ mount

    scene.frame_start, scene.frame_end = 0, (len(links) - 1) // 2
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.compression = 15
    scene.render.filepath = str(destination / "#####")
    bpy.app.handlers.frame_change_pre.append(set_pose)
    try:
        scene.frame_set(0)
        bpy.ops.render.render(animation=True)
    finally:
        bpy.app.handlers.frame_change_pre.remove(set_pose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workcell", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("view")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    render_view(args.workcell, args.run, args.view, args.destination)
