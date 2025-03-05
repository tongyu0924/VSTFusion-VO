import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
from datasets.utils import rotation_to_euler
import torch
import matplotlib.pyplot as plt
from torchvision import transforms


class KITTI(torch.utils.data.Dataset):
    def __init__(self,
                 data_path=r"data/sequences_jpg",
                 gt_path=r"data/poses",
                 camera_id="0",
                 sequences=["00", "02", "08", "09"],
                 flow_path=r"/home/leohsu/tongyu/TSformer-VO/data/sequences_jpg",
                 window_size=3,
                 overlap=1,
                 read_poses=True,
                 transform=None):
        self.data_path = data_path
        self.gt_path = gt_path
        self.camera_id = camera_id
        self.read_poses = read_poses
        self.window_size = window_size
        self.overlap = overlap
        self.transform = transform
        self.flow_path = flow_path

        # KITTI normalization
        self.mean_angles = np.array([1.7061e-5, 9.5582e-4, -5.5258e-5])
        self.std_angles = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
        self.mean_t = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
        self.std_t = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])

        # Define sequences for training, test, and validation
        self.sequences = sequences

        # Read frames list and ground truths
        frames, seqs = self.read_frames()
        gt = self.read_gt()

        # Create dataframe with frames and ground truths
        data = pd.DataFrame({"gt": gt})
        data = data["gt"].apply(pd.Series)
        data["frames"] = frames
        data["sequence"] = seqs
        self.data = data
        self.windowed_data = self.create_windowed_dataframe(data)
        self.depth_transform = transforms.Compose([
            transforms.Resize((192, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 單通道
        ])
        
        self.flow_transform = transforms.Compose([
            transforms.Resize((192, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 單通道
        ])

    def __len__(self):
        return len(self.windowed_data["w_idx"].unique())

    def __getitem__(self, idx):
        """
        Returns:
            imgs_rgb {ndarray}: RGB特徵 [C, T, H, W]
            imgs_depth {ndarray}: 深度特徵 [C, T, H, W]
            y {list}: ground truth pose
        """
        # Get data of corresponding window index
        data = self.windowed_data.loc[self.windowed_data["w_idx"] == idx, :]

        # Read frames as RGB and Depth separately
        frames = data["frames"].values
        imgs_rgb, imgs_depth, imgs_flow = [], [], []
        
        for i, fname in enumerate(frames):
            # Load RGB
            rgb_path = fname
            rgb = Image.open(rgb_path).convert('RGB')
            
            # print(rgb_path)
            import re
            sequence_numbers = re.search(r"sequences_jpg/(\d+)/image_0", rgb_path).group(1)

            # Load Flow
            flow_file = f"flow_{i:04}_to_{i+1:04}.png"  # Construct flow file name
            flow_path = os.path.join(self.flow_path, "flow", sequence_numbers, flow_file)
            if os.path.exists(flow_path):
                flow = Image.open(flow_path).convert('L')  # Convert flow to grayscale (L)
                flow = np.array(flow).astype(np.float32) / 255.0  # Normalize flow
                flow = self.flow_transform(Image.fromarray((flow * 255).astype(np.uint8)))
            else:
                flow = torch.zeros((1, 192, 640))  # Placeholder tensor if flow does not exist

            # Load Depth
            depth_path = fname.replace(".jpg", "_disp.jpeg")
            if not os.path.exists(depth_path):
                raise FileNotFoundError(f"Depth map not found: {depth_path}")
            depth = Image.open(depth_path).convert('L')  # Load as grayscale

            # Normalize Depth and RGB
            depth = np.array(depth).astype(np.float32) / 255.0  # Normalize depth
            rgb = np.array(rgb).astype(np.float32) / 255.0  # Normalize RGB

            # Apply transform if provided
            if self.transform:
                rgb = self.transform(Image.fromarray((rgb * 255).astype(np.uint8)))
                depth = self.depth_transform(Image.fromarray((depth * 255).astype(np.uint8)))

            imgs_rgb.append(rgb.unsqueeze(0))
            imgs_depth.append(depth.unsqueeze(0))
            imgs_flow.append(flow.unsqueeze(0))


        # Concatenate frames for RGB and Depth
        imgs_rgb = np.concatenate(imgs_rgb, axis=0)
        imgs_depth = np.concatenate(imgs_depth, axis=0)
        imgs_flow = np.concatenate(imgs_flow, axis=0)
        imgs_rgb = np.asarray(imgs_rgb)
        imgs_depth = np.asarray(imgs_depth)
        imgs_flow = np.asarray(imgs_flow)

        # T C H W -> C T H W
        imgs_rgb = imgs_rgb.transpose(1, 0, 2, 3)
        imgs_depth = imgs_depth.transpose(1, 0, 2, 3)
        imgs_flow = imgs_flow.transpose(1, 0, 2, 3)

        # Read ground truth [window_size-1 x 6]
        gt_poses = data.loc[:, [i for i in range(12)]].values
        y = []
        for gt_idx, gt in enumerate(gt_poses):
            # Homogeneous pose matrix [4 x 4]
            pose = np.vstack([np.reshape(gt, (3, 4)), [[0., 0., 0., 1.]]])

            # Compute relative pose from frame1 to frame2
            if gt_idx > 0:
                pose_wrt_prev = np.dot(np.linalg.inv(pose_prev), pose)
                R = pose_wrt_prev[:3, :3]
                t = pose_wrt_prev[:3, 3]

                # Euler parameterization (rotations as Euler angles)
                angles = rotation_to_euler(R, seq='zyx')

                # Normalization
                angles = (np.asarray(angles) - self.mean_angles) / self.std_angles
                t = (np.asarray(t) - self.mean_t) / self.std_t

                # Concatenate angles and translation
                y.append(list(angles) + list(t))

            pose_prev = pose

        y = np.asarray(y)  # Discard first value
        y = y.flatten()

        return imgs_rgb, imgs_depth, imgs_flow, y


    def read_frames(self):
        # Get frames list
        frames = []
        seqs = []
        for sequence in self.sequences:
            frames_dir = os.path.join(self.data_path, sequence, "image_{}".format(self.camera_id), "*.jpg")
            frames_seq = sorted(glob.glob(frames_dir))
            frames = frames + frames_seq
            seqs = seqs + [sequence] * len(frames_seq)
        return frames, seqs

    def read_gt(self):
        # Read ground truth
        if self.read_poses:
            gt = []
            for sequence in self.sequences:
                with open(os.path.join(self.gt_path, sequence + ".txt")) as f:
                    lines = f.readlines()

                # Convert poses to float
                for line_idx, line in enumerate(lines):
                    line = line.strip().split()
                    line = [float(x) for x in line]
                    gt.append(line)

        else:  # Test data (sequences 11-21)
            gt = None

        return gt

    def create_windowed_dataframe(self, df):
        window_size = self.window_size
        overlap = self.overlap
        windowed_df = pd.DataFrame()
        w_idx = 0

        for sequence in df["sequence"].unique():
            seq_df = df.loc[df["sequence"] == sequence, :].reset_index(drop=True)
            row_idx = 0
            while row_idx + window_size <= len(seq_df):
                rows = seq_df.iloc[row_idx:(row_idx + window_size)].copy()
                rows["w_idx"] = len(rows) * [w_idx]  # Add window index column
                row_idx = row_idx + window_size - overlap
                w_idx = w_idx + 1
                windowed_df = pd.concat([windowed_df, rows], ignore_index=True)
        windowed_df.reset_index(drop=True)
        return windowed_df


if __name__ == "__main__":

    # Create dataloader
    preprocess = transforms.Compose([
        transforms.Resize((192, 640)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.34721234, 0.36705238, 0.36066107],
            std=[0.30737526, 0.31515116, 0.32020183]), ])

    data = KITTI(transform=preprocess, sequences=["04"], window_size=3, overlap=2)
    test_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

    timings = np.zeros((len(test_loader), 1))
    for i in range(len(test_loader)):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        imgs, gt = data[i]
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[i] = curr_time

    mean_syn = np.sum(timings) / len(test_loader)
    std_syn = np.std(timings)
    print("Mean pre-proc time: ", mean_syn)
    print("Std time: ", std_syn)

# idx = 500
# imgs, gt = data[idx]
# print("imgs.shape: ", imgs.shape)
# print("gt.shape: ", gt.shape)

# img = np.moveaxis(imgs[0, :, :, :], 0, -1)

# # post processing
# channelwise_mean = [0.34721234, 0.36705238, 0.36066107]
# channelwise_std = [0.30737526, 0.31515116, 0.32020183]
# img[:, :, 0] = img[:, :, 0] * channelwise_std[0] + channelwise_mean[0]
# img[:, :, 1] = img[:, :, 1] * channelwise_std[1] + channelwise_mean[1]
# img[:, :, 2] = img[:, :, 2] * channelwise_std[2] + channelwise_mean[2]

# plt.figure()
# plt.imshow((img * 255).astype(int));
