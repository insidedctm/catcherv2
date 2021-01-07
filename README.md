# Catcher Version 2

## Setup AWS training infrastructure
Open 'Cloud Formation' console, select Create Stack and then 'with new resources'. In the 'Specify Template' section select 'Upload a template file',
click on 'Upload File' and select `cf-training.yaml` from the repository root directory. Click 'Next', then enter/select the following values for
the parameters:

* Stack Name: CatcherV2TrainingStack
* EC2 type: p3.2xlarge
* Key Pair: id_rsa
* Subnet: ECS default - Public Subnet 1
* VPC: ECS default - VPC

Click Next, Next, Create Stack

When the stack has been created copy the IP address from the 'Outputs' section of the stack and then ssh into the instance:

`ssh -i ~/.ssh/id_rsa ubuntu@<ip-address>`

Then run the following commands
```bash
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 571043958073.dkr.ecr.eu-west-1.amazonaws.com
docker pull 571043958073.dkr.ecr.eu-west-1.amazonaws.com/catcher-file-upload
docker run --gpus all -it -p 6006:6006 --entrypoint /bin/bash 571043958073.dkr.ecr.eu-west-1.amazonaws.com/catcher-file-upload:latest
```

This takes you into the docker container in the `/projects/catcher` directory. Run the following commands to finalise setup:

```bash
cd ..
export PYTHONPATH=$PYTHONPATH:/projects/catcher
git clone https://github.com/insidedctm/catcherv2.git
cd catcherv2
```

## Training
### 3D Convolution Model
To run training for the 3D Convolution model (requires something with more than 16GB memory)

```bash
python3 train_conv3d.py --num_epochs 5 --batch_size 20
```

Amend `num_epochs` and `batch_size` appropriately. NB 20  for `batch_size` works on machines with 32GB/64GB

### Naive Transfer Learning Model
To run training for the Naive Transfer Learning model

```bash
python3 catcher-v2.py --num_epochs 5 --batch_size 150
```

Amend `num_epochs` and `batch_size` appropriately. NB 150 seems a reasonable value for `batch_size`; on a g4dn.xlarge instance 
`batch_size=400` resulted in out of memory errors.

### Running Tensorboard
```bash
tensorboard --logdir tf_logs/ > /dev/null 2>&1 &
```
