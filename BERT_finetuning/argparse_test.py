import argparse


parser = argparse.ArgumentParser(description='Finetune BERT model')
parser.add_argument('--fold', type=int, nargs=1,
                    help='fold indice')

args = parser.parse_args()
print(args.fold[0])