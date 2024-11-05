# Info: About the proposed framework
In this work we have proposed an attention based distillation framework for the occludeded 3D human pose estimation. The sturcture of the proposed model is following.  
## About the pose filling network
**The Structure**
the filling network is composed with an attention filling model and several global attention finetuning models.  
![filling_network](https://github.com/user-attachments/assets/06703f06-f32c-4c0c-903b-9e912ed6b706)
**Code**
[Model Code](https://github.com/XuyangHao123/3D_Occluded_HPE/blob/main/models.py#L411)
[Train Process](https://github.com/XuyangHao123/3D_Occluded_HPE/blob/main/pre_train_filling_network.py#L38)

## About the pose lifting network
**The Structure**
The pose lifting network is composed by several residual linear layers and one graph convolution layer.
**Code**
[Model Code](https://github.com/XuyangHao123/3D_Occluded_HPE/blob/main/models.py#L30)
[Train Process](https://github.com/XuyangHao123/3D_Occluded_HPE/blob/main/pre_train_lifting_network.py#L41)

## About the distillation framework
**The Strcture**
The teacher network is pretrained on the none occluded poses, the filling network is pretrained on the occluded poses and the student network is trained with the occluded poses.  
Note: the teacher and student networks have the same structure.
**The training process**
![total_process](https://github.com/user-attachments/assets/c355bfa2-a855-478b-91f5-a8be4a6b18d3)
**Code**
[Train Process](https://github.com/XuyangHao123/3D_Occluded_HPE/blob/main/org_main.py#L50)


# Main File description
  3D_Occluded_HPE  
&emsp;&emsp;|-pre_train_filling_network.py #pretrain the filling network, this will save the filling network to the dic: fillnet  
&emsp;&emsp;|-pre_train_lifting_network.py #pretrain the lifting network, this will save the teacher network to the dic: liftnet  
&emsp;&emsp;|-pre_train_inn2d_on_mpii.py #pretrain the normalizing flow network on other dataset  
&emsp;&emsp;|-org_main.py #train the student network, this will save the proposed mothed to the dic: total_model  
&emsp;&emsp;|-models.py #the proposed networks 
&emsp;&emsp;|-requirements.txt #the required packages of our expriments env
# 1.Clone the code
> git clone https://github.com/XuyangHao123/3D_Occluded_HPE.git  
# 2.Download the dataset
[dataset](https://pan.baidu.com/s/1w5J1l6AeYBVyxSPIn7b1jA?pwd=rvvx)  
put the dataset dic under the root. (This is the dataset is P2 subject of Human3.6 dataset)  
# 3.Install requirements
> cd 3D_occluded_HPE  
> pip install -r requirements.txt  
# 4.Pretrain the tearcher and filling network
> python pre_train_filling_network.py  
> python pre_train_lifting_network.py  
# 5.Train
> python org_main.py  
# Appendix
If you want to train on the other dataset, please pretrain the normalizing flow network; change the attributes in the class: Config of each python file; change the [bl_prior](https://github.com/XuyangHao123/3D_Occluded_HPE/blob/main/pre_train_lifting_network.py#L48) in the org_main.py and [bl_prior](https://github.com/XuyangHao123/3D_Occluded_HPE/blob/main/org_main.py#L59) in the org_main.py  (not must but will influence the performance).
