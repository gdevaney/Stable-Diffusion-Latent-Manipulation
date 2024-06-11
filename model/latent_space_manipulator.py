import torch
import matplotlib.pyplot as plt

def generate_mask(cumulative_attention_maps, zz, suppress = True):
    # Threshold the cumulative attention maps
    # print("max of cumulative_attention_maps: ", torch.max(torch.log(cumulative_attention_maps)))
    # print("min of cumulative_attention_maps: ", torch.min(torch.log(cumulative_attention_maps)))
    # print("mean of cumulative_attention_maps: ", torch.mean(torch.log(cumulative_attention_maps)))
    # print("median of cumulative_attention_maps: ", torch.median(torch.log(cumulative_attention_maps)))
    mean1 = torch.exp(torch.mean(torch.log(cumulative_attention_maps)))
    mean2 = torch.exp(torch.mean(torch.log(cumulative_attention_maps))+0.2)
    # TODO: devise a strategy to threshold the cumulative_attention_maps
    if suppress:
        mask = (cumulative_attention_maps > mean1).float()
    else:
        mask = (cumulative_attention_maps > mean2).float()
    # plt.imshow(mask.squeeze(), cmap='hot')
    # plt.show()
    return mask

def latent_space_manipulation(latents, noised_latent_t, cumulative_attention_maps, zz, focus):
    for subject, idx in cumulative_attention_maps.items():
        # Generate mask
        if subject != focus:
            mask = generate_mask(cumulative_attention_maps[subject], zz, False)
            plt.imshow(mask.squeeze())
            plt.show()
        # Find indices where mask is 0
            zero_indices = (mask == 0).nonzero()
            one_indices = (mask!=0).nonzero()
        # Replace values in latents with corresponding values from noised_latent_t
            plt.imshow(noised_latent_t[20].squeeze().cpu().numpy().transpose(1,2,0))   
            plt.show()
            plt.imshow(noised_latent_t[780].squeeze().cpu().numpy().transpose(1,2,0))   
            plt.show()
            if zz == 1: 
                for idx in one_indices:
                    noise = noised_latent_t[240]
                    latents[0, :, idx[2], idx[3]] = noise[0, :, idx[2], idx[3]]
            elif zz == 2: 
                for idx in one_indices:
                    noise = noised_latent_t[220]
                    latents[0, :, idx[2], idx[3]] = noise[0, :, idx[2], idx[3]]
            else:
                for idx in one_indices:
                    noise = noised_latent_t[200]
                    latents[0, :, idx[2], idx[3]] = noise[0, :, idx[2], idx[3]]
            # for idx in one_indices:
            #     noise = noised_latent_t[999]
            #     latents[0, :, idx[2], idx[3]] = noise[0, :, idx[2], idx[3]]
                
        # elif zz == 1:
        #     mask = generate_mask(cumulative_attention_maps[subject], zz, False)

        # # Find indices where mask is 0
        #     zero_indices = (mask == 0).nonzero()
        #     one_indices = (mask != 0).nonzero()
        # # Replace values in latents with corresponding values from noised_latent_t
        #     # plt.imshow(noised_latent_t[780].squeeze().cpu().numpy().transpose(1,2,0))
        #     # plt.show()
        #     # for idx in zero_indices:
        #     #     noise2 = noised_latent_t[999]
        #     #     latents[0, :, idx[2], idx[3]] = noise2[0, :, idx[2], idx[3]]
        #     for idx in one_indices:
        #         noise = noised_latent_t[240]
        #         latents[0, :, idx[2], idx[3]] = noise[0, :, idx[2], idx[3]]
        # elif zz == 2:
        #     mask = generate_mask(cumulative_attention_maps[subject], zz, False)

        # # Find indices where mask is 0
        #     zero_indices = (mask == 0).nonzero()
        #     one_indices = (mask != 0).nonzero()
        # # Replace values in latents with corresponding values from noised_latent_t
        #     # plt.imshow(noised_latent_t[780].squeeze().cpu().numpy().transpose(1,2,0))
        #     # plt.show()
        #     # for idx in zero_indices:
        #     #     noise2 = noised_latent_t[999]
        #     #     latents[0, :, idx[2], idx[3]] = noise2[0, :, idx[2], idx[3]]
        #     for idx in one_indices:
        #         noise = noised_latent_t[220]
        #         latents[0, :, idx[2], idx[3]] = noise[0, :, idx[2], idx[3]]

        # else:
        #     mask = generate_mask(cumulative_attention_maps[subject], zz, False)

        # # Find indices where mask is 0
        #     zero_indices = (mask == 0).nonzero()
        #     one_indices = (mask != 0).nonzero()
        # # Replace values in latents with corresponding values from noised_latent_t
        #     # plt.imshow(noised_latent_t[780].squeeze().cpu().numpy().transpose(1,2,0))
        #     # plt.show()
        #     # for idx in zero_indices:
        #     #     noise2 = noised_latent_t[999]
        #     #     latents[0, :, idx[2], idx[3]] = noise2[0, :, idx[2], idx[3]]
        #     for idx in one_indices:
        #         noise = noised_latent_t[200]
        #         latents[0, :, idx[2], idx[3]] = noise[0, :, idx[2], idx[3]]

    return latents

def timestamps_to_manipulate(sampler):
    # control which other noised latents are needed for particular timesteps
    # TODO: devise a strategy to select timesteps for manipulating latent space
    preserve_timesteps = [sampler.timesteps[38], sampler.timesteps[28], sampler.timesteps[29], sampler.timesteps[27], sampler.timesteps[0]]
    edit_timesteps = [sampler.timesteps[2], sampler.timesteps[3], sampler.timesteps[4]]
    return edit_timesteps, preserve_timesteps