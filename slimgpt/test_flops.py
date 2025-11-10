import argparse
import torch
from ptflops import get_model_complexity_info

from models.llama.modeling_llama import LlamaForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_version = int(torch.__version__.split('.')[1])

def main(args):
        
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        low_cpu_mem_usage=True if torch_version >=9 else False
    )

    def input_constructor(x):
        return {'input_ids': torch.ones(x).long().to(device)}

    if device == "cuda":
        model.half()
        model = model.cuda()
    
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(model, (1, 64,), as_strings=True,
                                                    input_constructor = input_constructor,
                                                    print_per_layer_stat=True, verbose=True,
                                                    )
    else:
        model.float()
        macs, params = get_model_complexity_info(model, (1, 64,), as_strings=True,
                                                    input_constructor = input_constructor,
                                                    print_per_layer_stat=True, verbose=True,
                                                    )

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print("GPU Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default="", help='base model name')
    args = parser.parse_args()
    main(args)
