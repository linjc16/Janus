import json
import glob
import argparse
import pdb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='medqa_usmle', choices=['medqa_usmle'], help='dataset')
    args = parser.parse_args()
    filepath = f'data/{args.dataset}/graph/raw/*.json'

    
    out_dict = {}
    for filename in glob.glob(filepath):
        with open(filename, 'r') as f:
            data = json.load(f)
            out_dict.update(data)
    
    # filepath = f'data/{args.dataset}/graph/raw/cache/*.json'
    # for filename in glob.glob(filepath):
    #     with open(filename, 'r') as f:
    #         data = json.load(f)
    #         out_dict.update(data)

    for key, value in out_dict.items():
        
        for i in range(len(value)):
            # refine the value[i] by: (1) if length greater than 40 words, truncate it to 40 words;
            # (2) replace '\n' with ';'; replace '- ' with ''; replace '-' with ''
            value[i] = value[i].replace('\n', '; ').replace('- ', '').replace('-', '').replace('_', ' ')
            if len(value[i].split()) > 50:
                value[i] = ' '.join(value[i].split()[:50])
        
    

    with open(f'data/{args.dataset}/graph/train.graph2text.json', 'w') as f:
        json.dump(out_dict, f, indent=4)