import argparse

parser = argparse.ArgumentParser(description="Input arguments")
##### General Settings ##############################################################
parser.add_argument("--input", type=str, default="input", help="path to input data")
parser.add_argument("--output", type=str, default="output", help="path to save output")
parser.add_argument("--device", type=str, default="cuda", help="device type")
parser.add_argument('--image_size', action='store', type=int, nargs="+", default=[256, 512, 1024], help="the first is for training, the rest is for inference")
parser.add_argument("--latent_dim", type=int, default=512, help="dimension of latent vector")
parser.add_argument("--load_ckpt", type=str, default=None, help="path to the checkpoints to load")
parser.add_argument("--model_name", type=str, default="texture", choices=["texture", "stylegan2"], help="use our method or original stylegan2")
parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")
parser.add_argument("--n_mlp", type=int, default=8, help="size of multi-layer perceptron")
parser.add_argument("--max_texton_size", type=int, default=64, help="Apply textons to this resolution at max")
parser.add_argument("--n_textons", type=int, default=16, help="Number of textons per module")
parser.add_argument("--seed", type=int, default=0, help="RNG seed")
parser.add_argument("--n_worker", type=int, default=8,help="number of workers")
parser.add_argument("--save_img_every", type=int, default=500, help="save images every N iterations (for training or GAN inversion)")
parser.add_argument("--lr", type=float, default=0.002, help="learning rate (for training or GAN inversion)")
parser.add_argument("--iter", type=int, default=300000, help="total training iterations (for training or GAN inversion)")
parser.add_argument("--batch_size", type=int, default=16, help="mini-batch size (for training)")

##### Training Generator Settings ###################################################
parser.add_argument("--gp", type=float, default=0.01, help="Weight of gradient penalty loss for wasserstein distance")
parser.add_argument("--n_critic", type=int, default=2, help="Number of critic training iterations")
parser.add_argument("--noise_dx", action="store_false", help="add noise to input of discriminator")
parser.add_argument("--noise_std", type=float, default=0.01, help="standard deviation of noise added to Dx input")
parser.add_argument("--random_90_rotate", action="store_true", help="random 90*k degree rotation")
parser.add_argument("--random_flip", action="store_true", help="random horizontal and vertical flip")
parser.add_argument("--random_phase_noise", action="store_false", help="random phase noise for texton broadcast")
parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
parser.add_argument("--save_ckpt_every", type=int, default=500, help="save model ckpt every N iterations")
parser.add_argument("--crops_per_texture", type=int, default=2, help="Number of crops per texture")
parser.add_argument("--textures_per_batch", type=int, default=8, help="Number of textures in a batch")
parser.add_argument("--d_reg_every", type=int, default=16, help="Not actually used in this project (just for consistent adam params init). Interval of the applying r1 regularization")
parser.add_argument("--g_reg_every", type=int, default=4, help="Not actually used in this project (just for consistent adam params init). Interval of the applying path length regularization")
parser.add_argument("--wandb", action="store_true", help="use weights and biases logging")
parser.add_argument("--mixing", type=float, default=0, help="probability of latent code mixing")
parser.add_argument("--start_iter", type=int, default=0, help="Index of starting iteration (will be automatically set to non-zero based on args.load_ckpt)")

##### Inference Settings ############################################################
parser.add_argument("--n_textures", type=int, default=1000, help="total number of generated textures")
parser.add_argument("--samples_per_texture", type=int, default=20, help="Self-similar samples per texture")

##### GAN Inversion #################################################################
# parser.add_argument("--lr_opt_ganinvert", type=float, default=1e-3, help="learning rate")
parser.add_argument("--content_weight", type=float, default=0, help="weight for content-loss, default is 0")
parser.add_argument("--pix_weight", type=float, default=0, help="pixel-wise loss for reconstruction")
parser.add_argument("--style_weight", type=float, default=1e0, help="weight for style-loss, default is 0")
parser.add_argument('-w', '--wspace', help='Is it in W or Z space', action='store_true')
parser.add_argument('-e', '--extended', help='Is it in extended W or Z space', action='store_true')
parser.add_argument("--num_layers", type=int, default=50, help="Number of layers of encoder")


##### Latent Space Interpolation ####################################################
parser.add_argument("--n_pair", type=int, default=100, help="number of pairs of endpoint latent vectors to be generated ") 
parser.add_argument("--n_interpolate", type=int, default=8, help="number of the interpolated latent vectors in a pair, including two endpoints") 

##### Thresholded Invariant Pixel Percentage ########################################
parser.add_argument("--thresholds", action='store', type=float, nargs="+", default=[0.01,0.02,0.05,0.1,0.2,0.5])
parser.add_argument("--save_std_map", action="store_false")

args = parser.parse_args()