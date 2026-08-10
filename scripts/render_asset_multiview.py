"""Render actual candy and tea-package assets from six close-up views."""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); parser.add_argument("--output",default="outputs/asset_multiview/candy_tea_multiview.png"); parser.add_argument("--size",type=int,default=512); AppLauncher.add_app_launcher_args(parser); args=parser.parse_args(); args.enable_cameras=True; args.headless=True
launcher=AppLauncher(args); simulation_app=launcher.app
from pathlib import Path
import gymnasium as gym
import torch
from PIL import Image,ImageDraw,ImageFont
from isaaclab_tasks.utils import parse_env_cfg
import cg_isaac_envs
NAMES=("candy","tea_bag"); LABELS=("Candy proxy (YCB pudding package)","Tea proxy (YCB gelatin package)")
VIEWS=(("front",(0.15,0.,0.055)),("right",(0.,-0.15,0.055)),("back",(-0.15,0.,0.055)),("left",(0.,0.15,0.055)),("top",(0.001,0.,0.18)),("isometric",(0.11,-0.11,0.11)))
def main():
 count=len(NAMES)*len(VIEWS); cfg=parse_env_cfg("Isaac-CG-Household-Franka-IK-Rel-Vision-v0",device=args.device,num_envs=count); cfg.scene.front_camera.width=args.size; cfg.scene.front_camera.height=args.size; cfg.scene.front_camera.spawn.focal_length=58.; env=gym.make("Isaac-CG-Household-Franka-IK-Rel-Vision-v0",cfg=cfg).unwrapped; env.reset(seed=0)
 eyes=[]; targets=[]
 for row,name in enumerate(NAMES):
  for col,(_,offset) in enumerate(VIEWS):
   idx=row*len(VIEWS)+col; target=env.scene[name].data.root_pos_w[idx].clone(); target[2]+=0.012; eyes.append(target+torch.tensor(offset,device=env.device)); targets.append(target)
 camera=env.scene["front_camera"]; camera.set_world_poses_from_view(torch.stack(eyes),torch.stack(targets)); actions=torch.zeros(count,7,device=env.device); actions[:,6]=1.; observations,*_=env.step(actions); frames=observations["policy"]["front_rgb"][...,:3].clamp(0,255).byte().cpu().numpy()
 tile=args.size; header=56; canvas=Image.new("RGB",(tile*len(VIEWS),(tile+header)*len(NAMES)),(32,32,32)); draw=ImageDraw.Draw(canvas); font=ImageFont.load_default(size=18)
 for row,label in enumerate(LABELS):
  y=row*(tile+header); draw.text((12,y+8),label,fill="white",font=font)
  for col,(view_name,_) in enumerate(VIEWS):
   idx=row*len(VIEWS)+col; canvas.paste(Image.fromarray(frames[idx]),(col*tile,y+header)); draw.rectangle((col*tile,y+header,col*tile+110,y+header+28),fill=(0,0,0)); draw.text((col*tile+7,y+header+4),view_name,fill="white",font=font)
 output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True); canvas.save(output); print(output.resolve()); env.close()
if __name__=="__main__":
 try: main()
 finally: simulation_app.close()
