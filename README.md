# **Diffusion Models for Crowd Counting**

The implementation of Denoising Diffusion Probabilistic Models presented in the project is based on 
1. [openai/improved-diffusion](https://github.com/openai/improved-diffusion).
2. [JuliaWolleb/Diffusion-based-Segmentation](https://github.com/JuliaWolleb/Diffusion-based-Segmentation)

## **Project Abstract**

TODO


## **Data**

### **ShanghaiTech**
We evaluated our method on the [ShanghaiTech dataset](https://www.kaggle.com/datasets/tthien/shanghaitech-with-people-density-map).
For our dataloader, which can be found in the file *guided_diffusion/CrowdDataset_shangheiTech.py*, the dataset need to be stored in the following structure:

```
data
└───ShanghaiTech
│   └───part_A
│       │   └───train_data
│       │   │      └───ground-truth
│       │   │      └───ground-truth-h5
│       │   │      └───images
│       │   └───test_data
│       │          └───...
│   └───part_B
│       │   └───train_data
│       │   │      └───...
│       │   └───test_data
│       │          └───...
│

```

If you want to apply our code to another dataset, make sure the loaded image has attached the ground truth segmentation as the last channel.

## **Getting Start**

```bash
git clone https://github.com/jimmylin0979/diffusionCrowd.git
cd diffusionCrowd

# Recommend to create a new conda environment
pip install -r requirements.txt
```

## **Usage**

We set the flags as follows: (GPU: 2080 Ti 11GB)

Open another terminal and type the command to start a visdom server session  
The visdom server will display the port it is using, please make sure that the port in command `viz = Visdom(port=XXXX)` is the same with the port that server is using (use port 8097 as default)

```bash
python -m visdom.server
```

Then open another terminal to start training

```bash
MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"

# batch_size can set up to 5, if use 3 2080 Ti with multi-gpu method DP 
# batch_size can set up to 3, if use 1 2080 Ti only 
TRAIN_FLAGS="--lr 1e-4 --batch_size 5"
```
To train the segmentation model, run

```bash
python3 scripts/segmentation_train.py --data_dir ./data/ShanghaiTech/part_A/train_data $TRAIN_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS
```
The model will be saved in the *results* folder.
For sampling an ensemble of 5 segmentation masks with the DDPM approach, run:

```bash
python scripts/segmentation_sample.py  --data_dir ./data/ShanghaiTech/part_A/test_data  --model_path ./results/savedmodel.pt --num_ensemble=5 $MODEL_FLAGS $DIFFUSION_FLAGS
```
The generated segmentation masks will be stored in the *results* folder. A visualization of the sampling process is done using [Visdom](https://github.com/fossasia/visdom).

## **Roadmap**

- [x] Multi-GPU Training (Faster training speed/ Allow larger model)
- [ ] Use DDP as Multi-GPU Training Method
- [x] Evaluation checking
- [ ] Other dataset, like JHU-Crowd++


## **Citation**
If you use this code, please cite

```
@misc{wolleb2021diffusion,
      title={Diffusion Models for Implicit Image Segmentation Ensembles}, 
      author={Julia Wolleb and Robin Sandkühler and Florentin Bieder and Philippe Valmaggia and Philippe C. Cattin},
      year={2021},
      eprint={2112.03145},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
