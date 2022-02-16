import sys

import torch
from os import path as osp

currentdir = osp.dirname((osp.realpath(__file__)))
parentdir = osp.dirname(currentdir)
sys.path.append(parentdir)
import models.networks.outer_segmenter
import util
from models import BaseModel
import models.networks as networks
import models.networks.loss as loss


class SwappingAutoencoderModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        BaseModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--spatial_code_ch", default=8, type=int)
        parser.add_argument("--global_code_ch", default=2048, type=int)
        parser.add_argument("--lambda_R1", default=10.0, type=float)
        parser.add_argument("--lambda_patch_R1", default=1.0, type=float)
        parser.add_argument("--lambda_L1", default=1.0, type=float)
        parser.add_argument("--lambda_GAN", default=1.0, type=float)
        parser.add_argument("--lambda_PatchGAN", default=1.0, type=float)
        parser.add_argument("--patch_min_scale", default=1 / 8, type=float)
        parser.add_argument("--patch_max_scale", default=1 / 4, type=float)
        parser.add_argument("--patch_num_crops", default=8, type=int)
        parser.add_argument("--patch_use_aggregation",
                            type=util.str2bool, default=True)
        
        parser.add_argument("--lambda_inner_sem", default=1.0, type=float)
        parser.add_argument("--lambda_outer_sem", default=1.0, type=float)
        return parser

    def initialize(self):
        if self.opt.lambda_inner_sem > 0.0:
            self.E = networks.create_network(self.opt, self.opt.netE, "encoderinsemantic")
            self.crossentropy_loss = torch.nn.CrossEntropyLoss()
        else:
            self.E = networks.create_network(self.opt, self.opt.netE, "encoder")

        self.outer_segmenter = models.networks.outer_segmenter.create_outer_segmenter()
        if self.opt.lambda_outer_sem > 0.0:
            self.outer_loss = torch.nn.CrossEntropyLoss()   # SOLT !!
        self.G = networks.create_network(self.opt, self.opt.netG, "generator")
            
        if self.opt.lambda_GAN > 0.0:
            self.D = networks.create_network(
                self.opt, self.opt.netD, "discriminator")
        if self.opt.lambda_PatchGAN > 0.0:
            self.Dpatch = networks.create_network(
                self.opt, self.opt.netPatchD, "patch_discriminator"
            )

        # Count the iteration count of the discriminator
        # Used for lazy R1 regularization (c.f. Appendix B of StyleGAN2)
        self.register_buffer(
            "num_discriminator_iters", torch.zeros(1, dtype=torch.long)
        )
        self.l1_loss = torch.nn.L1Loss()

        if (not self.opt.isTrain) or self.opt.continue_train:
            self.load()

        if self.opt.num_gpus > 0:
            self.to("cuda:0")

    def per_gpu_initialize(self):
        pass

    # SOLT: nincs rá szükség
    # def swap(self, x):
    #     """ Swaps (or mixes) the ordering of the minibatch """
    #     shape = x.shape
    #     assert shape[0] % 2 == 0, "Minibatch size must be a multiple of 2"
    #     new_shape = [shape[0] // 2, 2] + list(shape[1:])
    #     x = x.view(*new_shape)
    #     x = torch.flip(x, [1])
    #     return x.view(*shape)

    def compute_image_discriminator_losses(self, real, rec, mix):
        if self.opt.lambda_GAN == 0.0:
            return {}

        pred_real = self.D(real)
        pred_rec = self.D(rec)
        pred_mix = self.D(mix)

        losses = {}
        losses["D_real"] = loss.gan_loss(
            pred_real, should_be_classified_as_real=True
        ) * self.opt.lambda_GAN

        losses["D_rec"] = loss.gan_loss(
            pred_rec, should_be_classified_as_real=False
        ) * (0.5 * self.opt.lambda_GAN)
        losses["D_mix"] = loss.gan_loss(
            pred_mix, should_be_classified_as_real=False
        ) * (1.0 * self.opt.lambda_GAN)     # SOLT : 0.5->1.0, because there are only half as much mixed imgs

        return losses

    def get_random_crops(self, x, crop_window=None):
        """ Make random crops.
            Corresponds to the yellow and blue random crops of Figure 2.
        """
        crops = util.apply_random_crop(
            x, self.opt.patch_size,
            (self.opt.patch_min_scale, self.opt.patch_max_scale),
            num_crops=self.opt.patch_num_crops
        )
        return crops

    def compute_patch_discriminator_losses(self, real, mix):
        losses = {}
        real_feat = self.Dpatch.extract_features(
            self.get_random_crops(real),
            aggregate=self.opt.patch_use_aggregation
        )
        target_feat = self.Dpatch.extract_features(self.get_random_crops(real))
        mix_feat = self.Dpatch.extract_features(self.get_random_crops(mix))

        losses["PatchD_real"] = loss.gan_loss(
            self.Dpatch.discriminate_features(real_feat, target_feat),
            should_be_classified_as_real=True,
        ) * self.opt.lambda_PatchGAN * 2.0  # SOLT : 1.0-> 2.0

        losses["PatchD_mix"] = loss.gan_loss(
            self.Dpatch.discriminate_features(real_feat, mix_feat),
            should_be_classified_as_real=False,
        ) * self.opt.lambda_PatchGAN * 2.0   # SOLT : 1.0->2.0

        return losses

    def compute_discriminator_losses(self, real):
        self.num_discriminator_iters.add_(1)

        # sp, gl = self.E(real)
        # B = real.size(0)

        realA = real[0] # SOLT
        realB = real[1]
        real = torch.cat((realA, realB), 0)
        
        spA, glA = self.E(realA)    # SOLT
        spB, glB = self.E(realB)
        B = realA.size(0)
        assert B % 2 == 0, "Batch size must be even on each GPU."
        

        # To save memory, compute the GAN loss on only
        # half of the reconstructed images
        # rec = self.G(sp[:B // 2], gl[:B // 2])    # SOLT
        # mix = self.G(self.swap(sp), gl)
        recA = self.G(spA[:B // 2], glA[:B // 2])
        recB = self.G(spB[:B // 2], glB[:B // 2])
        rec = torch.cat((recA, recB), 0)
        mix = self.G(spA, glB)

        losses = self.compute_image_discriminator_losses(real, rec, mix)

        if self.opt.lambda_PatchGAN > 0.0:
            # patch_losses = self.compute_patch_discriminator_losses(real, mix) # SOLT : real-> realB
            patch_losses = self.compute_patch_discriminator_losses(realB, mix)
            losses.update(patch_losses)

        metrics = {}  # no metrics to report for the Discriminator iteration

        # return losses, metrics, sp.detach(), gl.detach()  # SOLT : unnecessary values: sp, gl
        return losses, metrics

    # def compute_R1_loss(self, real):  # SOLT
    def compute_R1_loss(self, images):
        realA = images[0]  # SOLT
        realB = images[1]
        real = torch.cat((realA, realB), 0)
        losses = {}
        if self.opt.lambda_R1 > 0.0:
            real.requires_grad_()
            pred_real = self.D(real).sum()
            grad_real, = torch.autograd.grad(
                outputs=pred_real,
                inputs=[real],
                create_graph=True,
                retain_graph=True,
            )
            grad_real2 = grad_real.pow(2)
            dims = list(range(1, grad_real2.ndim))
            grad_penalty = grad_real2.sum(dims) * (self.opt.lambda_R1 * 0.5)
        else:
            grad_penalty = 0.0

        if self.opt.lambda_patch_R1 > 0.0:
            real_crop = self.get_random_crops(realB).detach()    # SOLT : real-> realB
            real_crop.requires_grad_()
            target_crop = self.get_random_crops(realB).detach()
            target_crop.requires_grad_()

            real_feat = self.Dpatch.extract_features(
                real_crop,
                aggregate=self.opt.patch_use_aggregation)
            target_feat = self.Dpatch.extract_features(target_crop)
            pred_real_patch = self.Dpatch.discriminate_features(
                real_feat, target_feat
            ).sum()

            grad_real, grad_target = torch.autograd.grad(
                outputs=pred_real_patch,
                inputs=[real_crop, target_crop],
                create_graph=True,
                retain_graph=True,
            )

            dims = list(range(1, grad_real.ndim))
            grad_crop_penalty = grad_real.pow(2).sum(dims) + \
                grad_target.pow(2).sum(dims)
            grad_crop_penalty *= (0.5 * self.opt.lambda_patch_R1 * 0.5)
            grad_crop_penalty *= 2.0 # SOLT added this line because #realB = 1/2 #real
        else:
            grad_crop_penalty = 0.0

        # losses["D_R1"] = grad_penalty + grad_crop_penalty # SOLT : concat
        losses["D_R1"] = torch.cat((grad_penalty, grad_crop_penalty), 0)    # size: 16 + 8 = 24.
        

        return losses

    def compute_generator_losses(self, data_i):
        losses, metrics = {}, {}
        # B = real.size(0)
        realA = data_i["real_A"]
        realB = data_i["real_B"]
        
        semA_small = data_i["sem_A_small"]
        semB_small = data_i["sem_B_small"]
        
        semA = data_i["sem_A"]
        semB = data_i["sem_B"]
        
        if self.opt.lambda_inner_sem > 0.0:
            spA, glA, sempredA = self.E(realA, extract_inner_sem=True)
            spB, glB, sempredB = self.E(realB, extract_inner_sem=True)
        else:
            spA, glA = self.E(realA)    # SOLT
            spB, glB = self.E(realB)

        B = realA.size(0)   # SOLT
        
        
        # rec = self.G(sp[:B // 2], gl[:B // 2])  # only on B//2 to save memory

        recA = self.G(spA[:B // 2], glA[:B // 2])  # SOLT
        recB = self.G(spB[:B // 2], glB[:B // 2])
        rec = torch.cat((recA, recB), 0)

        # mix = self.G(sp_mix, gl)
        mix = self.G(spA, glB)  # SOLT

        # record the error of the reconstructed images for monitoring purposes
        # metrics["L1_dist"] = self.l1_loss(rec, real[:B // 2]) # SOLT
        metrics["L1_dist_A"] = self.l1_loss(recA, realA[:B // 2])
        metrics["L1_dist_B"] = self.l1_loss(recB, realB[:B // 2])
        
        if self.opt.lambda_inner_sem > 0.0:
            losses["Inner_sem"] = (self.crossentropy_loss(sempredA, semA_small) +
                                   self.crossentropy_loss(sempredB, semB_small)) * self.opt.lambda_inner_sem
        
        # with torch.no_grad():
            # outer_preds_mix, _ = torch.max(self.outer_segmenter(mix), dim=1)
        outer_preds_mix = self.outer_segmenter(mix)
            
            
        # print("__________________________________________")
        # print(f"size of outer_preds_mix: {outer_preds_mix.size()}")
        # print(f"type of outer_preds_mix: {type(outer_preds_mix)}")
        # print(f"size of semA: {semA.size()}")
        
        
        metrics["outer_sem_mix"] = self.outer_loss(outer_preds_mix, semA)
        
        if self.opt.lambda_outer_sem > 0.0:
            losses["outer_sem_loss"] = metrics["outer_sem_mix"] * self.opt.lambda_outer_sem
        
        if self.opt.lambda_L1 > 0.0:
            # losses["G_L1"] = metrics["L1_dist"] * self.opt.lambda_L1
            losses["G_L1"] = (metrics["L1_dist_A"] + metrics["L1_dist_B"]) * self.opt.lambda_L1    # SOLT

        if self.opt.lambda_GAN > 0.0:
            losses["G_GAN_rec"] = loss.gan_loss(
                self.D(rec),
                should_be_classified_as_real=True
            ) * (self.opt.lambda_GAN * 0.5)

            losses["G_GAN_mix"] = loss.gan_loss(
                self.D(mix),
                should_be_classified_as_real=True
            ) * (self.opt.lambda_GAN * 2.0)     # SOLT : 1.0->2.0 because there are only half of the amount of mixed images

        if self.opt.lambda_PatchGAN > 0.0:
            # real_feat = self.Dpatch.extract_features(
            #     self.get_random_crops(real),
            #     aggregate=self.opt.patch_use_aggregation).detach()
            
            real_feat = self.Dpatch.extract_features(   # SOLT : real->realB
                self.get_random_crops(realB),
                aggregate=self.opt.patch_use_aggregation).detach()
            mix_feat = self.Dpatch.extract_features(self.get_random_crops(mix))

            losses["G_mix"] = loss.gan_loss(
                self.Dpatch.discriminate_features(real_feat, mix_feat),
                should_be_classified_as_real=True,
            ) * self.opt.lambda_PatchGAN * 2.0  # SOLT : 1.0->2.0 because there are only half of the amount of mixed images

        return losses, metrics

    def get_visuals_for_snapshot(self, real):
        if self.opt.isTrain:
            # avoid the overhead of generating too many visuals during training
            real = real[:2] if self.opt.num_gpus > 1 else real[:4]
        
        # sp, gl = self.E(real)
        realA = real[0]  # SOLT
        realB = real[1]
        real = torch.cat((realA, realB), 0)

        spA, glA = self.E(realA)  # SOLT
        spB, glB = self.E(realB)
        sp, gl = torch.cat((spA, spB), 0), torch.cat((glA, glB), 0)
        layout = util.resize2d_tensor(util.visualize_spatial_code(sp), real)
        rec = self.G(sp, gl)
        # mix = self.G(sp, self.swap(gl))
        mix = self.G(spA, glB)

        visuals = {"real": real, "layout": layout, "rec": rec, "mix": mix}

        return visuals

    def fix_noise(self, sample_image=None):
        """ The generator architecture is stochastic because of the noise
        input at each layer (StyleGAN2 architecture). It could lead to
        flickering of the outputs even when identical inputs are given.
        Prevent flickering by fixing the noise injection of the generator.
        """
        if sample_image is not None:
            # The generator should be run at least once,
            # so that the noise dimensions could be computed
            sp, gl = self.E(sample_image)
            self.G(sp, gl)
        noise_var = self.G.fix_and_gather_noise_parameters()
        return noise_var

    # def encode(self, image, extract_features=False):  # SOLT modification
    #     return self.E(image, extract_features=extract_features)
    
    def encode(self, image):  # SOLT modification
        return self.E(image)

    def decode(self, spatial_code, global_code):
        return self.G(spatial_code, global_code)

    def get_parameters_for_mode(self, mode):
        if mode == "generator":
            return list(self.G.parameters()) + list(self.E.parameters())
        elif mode == "discriminator":
            Dparams = []
            if self.opt.lambda_GAN > 0.0:
                Dparams += list(self.D.parameters())
            if self.opt.lambda_PatchGAN > 0.0:
                Dparams += list(self.Dpatch.parameters())
            return Dparams
