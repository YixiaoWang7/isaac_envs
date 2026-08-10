"""Render downloaded 3x3x3 candidate assets in Isaac Sim."""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); parser.add_argument("--output",default="outputs/asset_multiview/meal_task_assets.png"); parser.add_argument("--size",type=int,default=384); AppLauncher.add_app_launcher_args(parser); args=parser.parse_args(); args.enable_cameras=True; args.headless=True
launcher=AppLauncher(args); simulation_app=launcher.app
from pathlib import Path
import gymnasium as gym
import torch
from PIL import Image,ImageDraw,ImageFont
from isaaclab_tasks.utils import parse_env_cfg
import cg_isaac_envs
from cg_isaac_envs.tasks.household.env_cfg import ASSET_DIR,dynamic_usd
ASSETS=(
 ("Bowl / YCB 024","bowl",ASSET_DIR/"024_bowl"/"bowl.usd",(0.75,)*3),
 ("Plate / YCB 029","plate",ASSET_DIR/"029_plate"/"plate.usd",(0.55,)*3),
 ("Serving pan / YCB 027","mug",ASSET_DIR/"027_skillet"/"serving_pan.usd",(0.45,)*3),
 ("Apple / YCB 013","candy",ASSET_DIR/"013_apple"/"apple.usd",(0.80,)*3),
 ("Banana / YCB 011","tea_bag",ASSET_DIR/"011_banana"/"banana.usd",(0.70,)*3),
 ("Snack package / YCB 008","spoon",ASSET_DIR/"008_pudding_box"/"candy_package.usd",(0.55,)*3),
)
VIEWS=(("front",(0.20,0.,0.08)),("top",(0.001,0.,0.24)),("isometric",(0.15,-0.15,0.13)))
def main():
 count=len(ASSETS)*len(VIEWS); cfg=parse_env_cfg("Isaac-CG-Household-Franka-IK-Rel-Vision-v0",device=args.device,num_envs=count); cfg.scene.front_camera.width=args.size; cfg.scene.front_camera.height=args.size; cfg.scene.front_camera.spawn.focal_length=45.
 for _,slot,path,scale in ASSETS: getattr(cfg.scene,slot).spawn=dynamic_usd(str(path),scale)
 env=gym.make("Isaac-CG-Household-Franka-IK-Rel-Vision-v0",cfg=cfg).unwrapped; env.reset(seed=0); eyes=[]; targets=[]
 for row,(_,slot,_,_) in enumerate(ASSETS):
  for col,(_,offset) in enumerate(VIEWS):
   idx=row*len(VIEWS)+col; target=env.scene[slot].data.root_pos_w[idx].clone(); target[2]+=0.025; eyes.append(target+torch.tensor(offset,device=env.device)); targets.append(target)
 camera=env.scene["front_camera"]
 camera.set_world_poses_from_view(torch.stack(eyes),torch.stack(targets))
 for _ in range(4):
  env.sim.render()
  camera.update(0.0, force_recompute=True)
 frames=camera.data.output["rgb"][...,:3].clamp(0,255).byte().cpu().numpy()
 tile=args.size; head=52; canvas=Image.new("RGB",(tile*3,(tile+head)*6),(28,28,28)); draw=ImageDraw.Draw(canvas); font=ImageFont.load_default(size=18)
 for row,(label,_,_,_) in enumerate(ASSETS):
  y=row*(tile+head); draw.text((10,y+8),label,fill="white",font=font)
  for col,(view,_) in enumerate(VIEWS):
   img=Image.fromarray(frames[row*3+col]); canvas.paste(img,(col*tile,y+head)); draw.rectangle((col*tile,y+head,col*tile+105,y+head+28),fill=(0,0,0)); draw.text((col*tile+7,y+head+4),view,fill="white",font=font)
 out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); canvas.save(out); print(out.resolve()); env.close()
if __name__=="__main__":
 try: main()
 finally: simulation_app.close()
