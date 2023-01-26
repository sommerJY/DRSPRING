
# wandb 쓰는법 
# 근데 이쁘게 wrapping 을 다 해두고 그걸 기반으로 python py 를 굴려서 보는게 메커니즘임 


conda install -c conda-forge wandb
wandb login


# 그냥 코드에 뭐뭐 넣어줘야하는지만 확인 

wandb init 

import wandb

# main() 함수보다 먼저 
wandb.init(project="project-name", reinit=True)
wandb.run.name = 'your-run-name'
혹은
wandb.run.name = wandb.run.id

# config 정보를 넣고 싶은 경우 
wandb.init(config={"epochs": 4, "batch_size": 32})



# argparse 쓰는 경우 
wandb.init()
wandb.config.epochs = 4

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',
                     help='input batch size for training (default: 8)')
args = parser.parse_args()
wandb.config.update(args) # adds all of the arguments as config variables






# args 변수 선언하고 나서 
wandb.config.update(args)


# model 선언하고 나서 
wandb.watch(model)



# loss 확인하고 싶은 경우 
wandb.log({
        "Test Accuracy": 100. * correct / len(test_loader.dataset),
        "Test Loss": test_loss})

# histogram 으로 보고싶은 경우 
wandb.log({"gradients": wandb.Histogram(numpy_array_or_sequence)})
wandb.run.summary.update({"gradients": wandb.Histogram(np_histogram=np.histogram(data))})
