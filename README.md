# Main File description
  3D_Occluded_HPE
      |-pre_train_filling_network.py #pretrain the filling network, this will save the filling network to the dic: fillnet  
      |-pre_train_lifting_network.py #pretrain the lifting network, this will save the teacher network to the dic: liftnet  
      |-pre_train_inn2d_on_mpii.py #pretrain the normalizing flow network on other dataset  
      |-org_main.py #train the student network, this will save the proposed mothed to the dic: total_model  
      |-models.py #the proposed networks  
# 1.Clone the code
> git clone https://github.com/XuyangHao123/3D_Occluded_HPE.git  
# 2.Download the dataset
[link](https://pan.baidu.com/s/1w5J1l6AeYBVyxSPIn7b1jA?pwd=rvvx)  
put the dataset dic under the root  
# 3.Install requirements
> cd 3D_occluded_HPE  
> pip install requirements.txt  
# 4.Pretrain the tearcher and filling network
> python pre_train_filling_network.py  
> python pre_train_lifting_network.py  
# 5.Train
> python org_main.py  
# Appendix
If you want to train on the other dataset, please pretrain the normalizing flow network and change the attributes in the class: Config of each python file.  
