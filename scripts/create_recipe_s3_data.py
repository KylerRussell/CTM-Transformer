"""Create the S3 recipe-confirmation data: fresh words per seed, seeds disjoint from every earlier study and probe."""
from ctm_transformer.group_suite import create_group_suite

ROOT='research/data/recipe_s3_v1'
SEEDS=(347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521)

if __name__=='__main__':
    create_group_suite(ROOT,'S3',SEEDS,320000,16,512,1024,32,'recipe_s3_v1')
