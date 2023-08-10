import argparse
import oci
import json
import subprocess
import yaml

from mcli import RunConfig, create_run


parser = argparse.ArgumentParser()
parser.add_argument('-b', '--bucket', default='mosaicml-internal-checkpoints', type=str, help='Checkpoint bucket')
parser.add_argument('-p', '--prefix', default=None, type=str, help='Checkpoint prefix')
parser.add_argument('-r', '--remote', default=None, type=str, help='Remote dataset to eval on')
parser.add_argument('--subsets',
                    default=['painting', 'people', 'photography', 'face'],
                    type=str,
                    nargs='*',
                    help='Subsets to evaluate on')
parser.add_argument('-i',
                    '--iterations',
                    default=[550000],
                    type=int,
                    nargs='*',
                    help='Iterations to evaluate at')
parser.add_argument('-s', '--size', default=256, type=int, help='Image size')
parser.add_argument('--seed', default=17, type=int, help='Random seed')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('--num_samples', default=10000, type=int, help='Number of samples to eval')
parser.add_argument('--project', default=None, type=str, help='Wandb project')
parser.add_argument('--name', default='', type=str, help='Wandb base run name')
args = parser.parse_args()


# Get the list of all checkpoints
oci_config = oci.config.from_file()
object_storage_client = oci.object_storage.ObjectStorageClient(oci_config)
process = subprocess.Popen('oci os ns get', stdout=subprocess.PIPE, shell=True)
namespace, _ = process.communicate()
namespace = json.loads(namespace.decode('utf-8'))['data']
object_list = object_storage_client.list_objects(namespace, args.bucket, prefix=args.prefix , fields="name,timeCreated,size")
# Get the checkpoints corresponding to the desired iterations
eval_checkpoints = []
for o in object_list.data.objects:
    checkpoint_name = o.name.split('/')[-1]
    checkpoint_name = '-'.join(checkpoint_name.split('-')[:-1])
    if 'latest' not in checkpoint_name:
        batch_number = int(checkpoint_name.split('-')[1][2:])
        if batch_number in args.iterations:
            eval_checkpoints.append({"checkpoint": o.name, "checkpoint_name": checkpoint_name.split('.')[0]})


# mcli integrations
integrations = [
                    {'integration_type': 'git_repo',
                     'git_repo': 'mosaicml/diffusion',
                     'git_branch': 'main',
                     'pip_install': '.'},
                    {'integration_type': 'wandb',
                     'project': args.project,
                     'entity': 'mosaic-ml'}
               ]
# Scheduling
scheduling = {'priority': 'medium', 'resumable': False}

command = "cd diffusion \n"
command += "HYDRA_FULL_ERROR=1 composer run_eval.py --config-path /mnt/config --config-name parameters"

for eval_checkpoint in eval_checkpoints:
    for subset in args.subsets:
        load_path = f"oci://{args.bucket}/{eval_checkpoint['checkpoint']}"
        # Open the template yaml
        with open('template.yaml', 'r') as f:
            template = yaml.safe_load(f)
        # Fill out the template
        template['image_size'] = args.size
        template['batch_size'] = args.batch_size
        template['num_samples'] = args.num_samples
        if args.prefix is not None:
            template['load_path'] = load_path
        template['seed'] = args.seed
        template['name'] = f"{args.name}-{subset}-{eval_checkpoint['checkpoint_name']}-seed-{args.seed}"
        template['project'] = args.project
        template['remote'] = args.remote + subset
        template['local'] = f"/tmp/mds-cache-cc-{subset}"

        # Create a run config
        run_config = RunConfig(name='eval-sweep',
                                gpu_num=8,
                                cluster='r1z1',
                                image='mosaicml/pytorch_vision:1.13.1_cu117-python3.10-ubuntu20.04',
                                integrations=integrations,
                                command=command,
                                scheduling=scheduling,
                                parameters=template
                                )

        # Create the run
        print("Evaluating checkpoint: ", eval_checkpoint['checkpoint'])
        run = create_run(run_config)
