# Introduction

Transfer learning Accuracy

Type           | inf+qdr+rel →skt | inf+rel+skt →qdr | qdr+rel+skt →inf | inf+qdr+skt →rel
--------------|:-----:|:-----:|:----:|:--------------------:
Weak    | 23.1% |  11.8% |    8.2% | 41.8%
Strong    | 33.7% |  13.3% |  13.0% | 53.1%
Ours  | 35.09% | 12.23% |  13.2% | 49.97%    

# How to use it?
   1. use the <bash get_dataset.sh> to get the dataset
   2. make sure the dataset in the same file with "train_multi_D_*.py" file
   3. make sure the directories "save" and "submission" have been create.
   4. open the "Dataset.py" the modify the input image size for resnet152 and resnet50
      (usually 64\*64,128\*128,224\*224,256\*256)
   5. <python3 train_multi_D_*.py> to start training model
   6. collect all of csv output in a single folder
   7. <python3 vote.py your_folder_name> vote.csv as output in current path
> Some note for "*"
>   - rel is for inf+qdr+skt →rel
>   - skt is for inf+qdr+rel →skt
>   - qdr is for inf+rel+skt →qdr
>   - inf is for qdr+rel+skt →inf


# Training Tips:
If you want to train the model, here is some tips you have to follow.
   1. Step 1: train the model as input image size 64\*64 and batch size = 100
   2. Step 2: Get the highest test acc model in step 1
   3. Step 3: load the model in Step 2 and train the model as input image size 128\*128 and batch size = 50
   4. Step 4:load the highest test acc in Step 3 and train the model as input image size 244\*244 and batch size = 20
   4. Step 5:load the highest test acc in Step 4 and train the model as input image size 256\*256 and batch size = 10

>  *If you don't follow the tips, you probably may not get good Accuracy after training.*


# How to predict?
### predict for test dataset
   1. Download the model file from Google drive (sorry my dropbox is filled with hw2/hw3/hw4)
      "https://drive.google.com/drive/u/0/folders/1cZtetDDiQX5qr_yW3PQziwKCSJx5YBSB"
   2. For predict the test dataset ,you should download *-rel-best.pth
   3. Important! Make sure image size in Dataset_Pred is 256 (desired_size = 256)
   4. python3 predict.py <encoder_path> <classifier_path>
   Ex: python3 predict.py ./save/256/encoder--0.4162.pth ./save/256/classifier--0.4162.pth
   5. You will receive a "pred.csv" file in submission

### predict for inf, skt, qdr dataset
   1. Download the model file from Google drive (sorry my dropbox is filled with hw2/hw3/hw4)
      "https://drive.google.com/drive/u/0/folders/1cZtetDDiQX5qr_yW3PQziwKCSJx5YBSB"
   2. For predict the skt dataset ,you should download *-skt-best.pth
   3. Important! Don't change the desired_size in Dataset_pred directory
   4. python3 predict_skt/inf/qdr.py <encoder_path> <classifier_path>    
   Ex: python3 predict_skt.py ./save/256/encoder--0.4162.pth ./save/256/classifier--0.4162.pth
   5. You will receive a "pred_skt/inf/qdr.csv" file in submission
### Why voting?
   1. The image size is various in test set
   2. According to out tsne plot on model of differnt image size, domain matching outcome somehow depends on size of image
   3. So implement voting can improve the outcome by combining the result of differnt image size
### Special parts on loading image
   1. H,W of some image is rather differnt, so reshape it into a square image may cause info loss
   2. Pading reshape is implemented to solve this problem
# VisDA2019-Visual-Domain-Adaptation-Challenge
