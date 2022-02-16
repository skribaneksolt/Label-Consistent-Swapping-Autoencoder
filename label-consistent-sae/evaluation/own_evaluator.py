import os
import torchvision.transforms as transforms
from PIL import Image
from evaluation import BaseEvaluator
from data.base_dataset import get_transform
import util
from glob import glob
from tqdm import tqdm

class OwnEvaluator(BaseEvaluator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--input_structure_folder", required=True, type=str)
        parser.add_argument("--input_texture_folder", required=True, type=str)
        parser.add_argument("--texture_mix_alphas", type=float, nargs='+',
                            default=[1.0],
                            help="Performs interpolation of the texture image."
                                 "If set to 1.0, it performs full swapping."
                                 "If set to 0.0, it performs direct reconstruction"
                            )
        
        opt, _ = parser.parse_known_args()
        # dataroot = os.path.dirname(opt.input_structure_image)
        dataroot = os.path.dirname(opt.input_structure_folder)
        # dataroot and dataset_mode are ignored in SimpleSwapplingEvaluator.
        # Just set it to the directory that contains the input structure image.
        parser.set_defaults(dataroot=dataroot, dataset_mode="imagefolder")
        
        return parser
    
    def load_image(self, path):
        path = os.path.expanduser(path)
        img = Image.open(path).convert('RGB')
        transform = get_transform(self.opt)
        tensor = transform(img).unsqueeze(0)
        return tensor
    
    def evaluate(self, model, dataset, nsteps=None):
        os.makedirs(self.output_dir(), exist_ok=True)
        
        structure_paths = glob(self.opt.input_structure_folder + '/*.png')
        texture_paths = glob(self.opt.input_texture_folder + '/*.png')
        print(f"num of texture images: {len(texture_paths)}")
        print(f"num of structure images: {len(structure_paths)}")
        for i, texture_path in enumerate(texture_paths):
            for _, structure_path in tqdm(enumerate(structure_paths)):
                structure_image = self.load_image(structure_path)
                texture_image = self.load_image(texture_path)
                
                model(sample_image=structure_image, command="fix_noise")
                structure_code, source_texture_code = model(
                    structure_image, command="encode")
                _, target_texture_code = model(texture_image, command="encode")
                
                # alphas = self.opt.texture_mix_alphas
                # for alpha in alphas:
                texture_code = util.lerp(
                    source_texture_code, target_texture_code, 1.0)
                
                output_image = model(structure_code, texture_code, command="decode")
                output_image = transforms.ToPILImage()(
                    (output_image[0].clamp(-1.0, 1.0) + 1.0) * 0.5)
                

                output_name = "%s_%d.png" % (
                    os.path.splitext(os.path.basename(structure_path))[0][:-7],
                    i
                )

                output_path = os.path.join(self.output_dir(),  output_name)
                
                output_image.save(output_path)
                # for cycle end
        
        return {}
