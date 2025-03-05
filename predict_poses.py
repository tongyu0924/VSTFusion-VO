from torchvision import transforms
from datasets.kitti import KITTI
import pickle
import torch
from timesformer.models.vit import VisionTransformer
from functools import partial
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
from timesformer.models.swin_vit import SwinTransformer3D


checkpoint_path = "/home/leohsu/tongyu/TSformer-VO/checkpoints/Exp_video_swin_transformer"
checkpoint_name = "checkpoint_e80"
sequences = ["01", "03", "04", "05", "06", "07", "10"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# read hyperparameters and configuration
with open(os.path.join(checkpoint_path, "args.pkl"), 'rb') as f:
    args = pickle.load(f)
f.close()
model_params = args["model_params"]
args["checkpoint_path"] = checkpoint_path
print(args)

# preprocessing operation
preprocess = transforms.Compose([
    transforms.Resize((model_params["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.34721234, 0.36705238, 0.36066107],
        std=[0.30737526, 0.31515116, 0.32020183]),
])

# build and load model
model = SwinTransformer3D(
        img_size=model_params["image_size"],
        num_classes=model_params["num_classes"],
        depth=model_params["depth"],
        num_frames=model_params["num_frames"],
        # num_heads=model_params["heads"],  # Ensure this is a list/tuple
        window_size=(2, 7, 7),
        embed_dim=model_params["dim"],
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=model_params["attn_dropout"],
        drop_path_rate=model_params["ff_dropout"],
        norm_layer=nn.LayerNorm,
    )

checkpoint = torch.load(os.path.join(args["checkpoint_path"], "{}.pth".format(checkpoint_name)),
                        map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'])
if torch.cuda.is_available():
    model.cuda()


for sequence in sequences:
    # test dataloader
    dataset = KITTI(transform=preprocess, sequences=[sequence],
                    window_size=args["window_size"], overlap=args["overlap"])
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              shuffle=False,
                                             )

    with tqdm(test_loader, unit="batch") as batchs:
      pred_poses = torch.zeros((1, args["window_size"] - 1, 6), device=device)
      batchs.set_description(f"Sequence {sequence}")
      for images, depth, flow, gt  in batchs:
        if torch.cuda.is_available():
          images, depth, flow, gt = images.cuda(), depth.cuda(), flow.cuda(), gt.cuda()

          with torch.no_grad():
            model.eval()
            model.training = False

            # predict pose
            pred_pose = model(images.float(), depth.float(), flow.float())
            pred_pose = torch.reshape(pred_pose, (args["window_size"] - 1, 6)).to(device)
            pred_pose = pred_pose.unsqueeze(dim=0)
            pred_poses = torch.concat((pred_poses, pred_pose), dim=0)
    
    # save as numpy array
    pred_poses = pred_poses[1:, :, :].cpu().detach().numpy()

    save_dir = os.path.join(args["checkpoint_path"], checkpoint_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, "pred_poses_{}.npy".format(sequence)), pred_poses)
