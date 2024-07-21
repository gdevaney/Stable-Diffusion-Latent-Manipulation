# Stable Diffusion Latent Manipulation
Giving individuals the ability to retain key features of image subjects while generating new, user-prompted backgrounds using Stable Diffusion.

## Introduction/Background
Diffusion models offer a unique approach to high-resolution image generation by leveraging text prompts and existing photos. However, existing architectures often struggle to maintain strong ties between the generated images and the input references. Recent research advancements have been able to produce scene changes in Stable Diffusion produced images using user prompts. However, this novel approach allows users to input their OWN images while still using the Stable Diffusion architechture for scene change tasks. This drastically expands the use cases of Stable Diffusion, as users can now generate new backgrounds to photo subjects that are important to them. For instance, given a photo of an individual's dog playing in a park, my manipulated Stable diffusion model allows the user to prompt a scene change while retaining the original dog in its active state. 

Given that you have the necessary spaCy libraries downloaded, this model works out-of-the-box with the jupyter notebook file "stable_diffusion_latent_manipulation.ipynb". 

## Methods
