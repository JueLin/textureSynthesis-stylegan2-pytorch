import argparse
import os
import numpy as np

from PIL import Image

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Thresholded Invariant Pixel Percentage, the lower the better")
	parser.add_argument("--root", type=str, required=True)
	parser.add_argument("--output", type=str, default="TIPP")
	parser.add_argument("--thresholds", type=str, default="0.01,0.02,0.05,0.1,0.2,0.5")
	parser.add_argument("--save_std_map", action="store_false")
	parser.add_argument("--verbose", action="store_true")
	args = parser.parse_args()
	out_folder_path = args.output
	if not os.path.exists(out_folder_path):
		os.makedirs(out_folder_path, exist_ok=True)
	thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
	folders = os.listdir(args.root)
	average_TIPPs = []
	for folder in folders:
		folder_path = os.path.join(args.root, folder)
		imgs_name = os.listdir(folder_path)
		imgs = []
		for img_name in imgs_name:
			img_path = os.path.join(folder_path, img_name)
			img = np.array(Image.open(img_path).convert("L"), dtype=np.float32)/255.0
			imgs.append(img)
		imgs = np.array(imgs)
		std_map = np.std(imgs, axis=0)
		tipps =  [np.sum(std_map <= threshold)/std_map.size for threshold in thresholds]
		average_TIPPs.append(tipps)
		if args.verbose:
			for threshold, tipp in zip(thresholds, tipps):
				print("Folder name = %s, threshold = %.5f, TIPP = %.5f"%(folder, threshold, tipp))
		if args.save_std_map:		
			out_img_name = os.path.join(out_folder_path, "%s.png"%folder)
			std_map = 255*np.clip(std_map, a_min=0, a_max=1.0)
			std_map = Image.fromarray(std_map.astype(np.uint8)).convert("L")
			std_map.save(out_img_name)

	average_TIPPs = np.array(average_TIPPs).mean(axis=0)
	out_tipp_file = os.path.join(out_folder_path, "tipp.npy")	
	with open(out_tipp_file, 'wb') as f:
		np.save(f, average_TIPPs)

	for threshold, a_tipp in zip(thresholds, average_TIPPs):
		print("The average TIPP of %s = %.5f, with threshold = %.5f"%(args.root, a_tipp, threshold))

