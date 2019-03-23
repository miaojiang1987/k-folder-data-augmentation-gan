mkdir results/celeba_1/WGAN_GP/sliced
mkdir results/celeba_2/WGAN_GP/sliced
mkdir results/celeba_3/WGAN_GP/sliced
mkdir results/celeba_4/WGAN_GP/sliced
mkdir results/celeba_5/WGAN_GP/sliced
#mkdir results/celeba_6/WGAN_GP/sliced
#mkdir results/celeba_7/WGAN_GP/sliced
#mkdir results/celeba_8/WGAN_GP/sliced
#mkdir results/celeba_9/WGAN_GP/sliced
#mkdir results/celeba_10/WGAN_GP/sliced
python main.py --dataset celeba_1 --gan_type WGAN_GP --epoch 100 --batch_size 64 --repeat 0 --input_size 64
python main.py --dataset celeba_2 --gan_type WGAN_GP --epoch 100 --batch_size 64 --repeat 0 --input_size 64
python main.py --dataset celeba_3 --gan_type WGAN_GP --epoch 100 --batch_size 64 --repeat 0 --input_size 64
python main.py --dataset celeba_4 --gan_type WGAN_GP --epoch 100 --batch_size 64 --repeat 0 --input_size 64
python main.py --dataset celeba_5 --gan_type WGAN_GP --epoch 100 --batch_size 64 --repeat 0 --input_size 64
#python main.py --dataset celeba_6 --gan_type WGAN_GP --epoch 300 --batch_size 64
#python main.py --dataset celeba_7 --gan_type WGAN_GP --epoch 300 --batch_size 64
#python main.py --dataset celeba_8 --gan_type WGAN_GP --epoch 300 --batch_size 64
#python main.py --dataset celeba_9 --gan_type WGAN_GP --epoch 300 --batch_size 64
#python main.py --dataset celeba_10 --gan_type WGAN_GP --epoch 300 --batch_size 64

cp results/slice_add.py results/celeba_1/WGAN_GP/sliced 
cp results/slice_add.py results/celeba_2/WGAN_GP/sliced
cp results/slice_add.py results/celeba_3/WGAN_GP/sliced
cp results/slice_add.py results/celeba_4/WGAN_GP/sliced
cp results/slice_add.py results/celeba_5/WGAN_GP/sliced
#cp results/slice_add.py results/celeba_6/WGAN_GP/sliced
#cp results/slice_add.py results/celeba_7/WGAN_GP/sliced
#cp results/slice_add.py results/celeba_8/WGAN_GP/sliced
#cp results/slice_add.py results/celeba_9/WGAN_GP/sliced
#cp results/slice_add.py results/celeba_10/WGAN_GP/sliced


cd results/celeba_1/WGAN_GP/sliced/
python slice_add.py 1
cd ../../../../results/celeba_2/WGAN_GP/sliced/
python slice_add.py 2
cd ../../../../results/celeba_3/WGAN_GP/sliced/
python slice_add.py 3
cd ../../../../results/celeba_4/WGAN_GP/sliced/
python slice_add.py 4
cd ../../../../results/celeba_5/WGAN_GP/sliced/
python slice_add.py 5
#cd ../../../../results/celeba_6/WGAN_GP/sliced/
#python slice_add.py 6
#cd ../../../../results/celeba_7/WGAN_GP/sliced/
#python slice_add.py 7
#cd ../../../../results/celeba_8/WGAN_GP/sliced/
#python slice_add.py 8
#cd ../../../../results/celeba_9/WGAN_GP/sliced/
#python slice_add.py 9
#cd ../../../../results/celeba_10/WGAN_GP/sliced/
#python slice_add.py 10
cd ../../../../

cp -r results/celeba_1 results/5_folder_results/5000_images/iteration1
cp -r results/celeba_2 results/5_folder_results/5000_images/iteration1
cp -r results/celeba_3 results/5_folder_results/5000_images/iteration1
cp -r results/celeba_4 results/5_folder_results/5000_images/iteration1
cp -r results/celeba_5 results/5_folder_results/5000_images/iteration1
cp -r model_result_distributed_1 results/5_folder_results/5000_images/iteration1
cp -r model_result_distributed_2 results/5_folder_results/5000_images/iteration1
cp -r model_result_distributed_3 results/5_folder_results/5000_images/iteration1
cp -r model_result_distributed_4 results/5_folder_results/5000_images/iteration1
cp -r model_result_distributed_5 results/5_folder_results/5000_images/iteration1

python main.py --dataset celeba_1 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_1/discriminator_epoch_099.pth --netG_path model_result_distributed_1/generator_epoch_099.pth --input_size 64
python main.py --dataset celeba_2 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_2/discriminator_epoch_099.pth --netG_path model_result_distributed_2/generator_epoch_099.pth --input_size 64
python main.py --dataset celeba_3 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_3/discriminator_epoch_099.pth --netG_path model_result_distributed_3/generator_epoch_099.pth --input_size 64
python main.py --dataset celeba_4 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_4/discriminator_epoch_099.pth --netG_path model_result_distributed_4/generator_epoch_099.pth --input_size 64
python main.py --dataset celeba_5 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_5/discriminator_epoch_099.pth --netG_path model_result_distributed_5/generator_epoch_099.pth --input_size 64
#python main.py --dataset celeba_6 --gan_type WGAN_GP --epoch 300 --batch_size 64
#python main.py --dataset celeba_7 --gan_type WGAN_GP --epoch 300 --batch_size 64
#python main.py --dataset celeba_8 --gan_type WGAN_GP --epoch 300 --batch_size 64
#python main.py --dataset celeba_9 --gan_type WGAN_GP --epoch 300 --batch_size 64
#python main.py --dataset celeba_10 --gan_type WGAN_GP --epoch 300 --batch_size 64

cd results/celeba_1/WGAN_GP/sliced/
python slice_add.py 1
cd ../../../../results/celeba_2/WGAN_GP/sliced/
python slice_add.py 2
cd ../../../../results/celeba_3/WGAN_GP/sliced/
python slice_add.py 3
cd ../../../../results/celeba_4/WGAN_GP/sliced/
python slice_add.py 4
cd ../../../../results/celeba_5/WGAN_GP/sliced/
python slice_add.py 5
#cd ../../../../results/celeba_6/WGAN_GP/sliced/
#python slice_add.py 6
#cd ../../../../results/celeba_7/WGAN_GP/sliced/
#python slice_add.py 7
#cd ../../../../results/celeba_8/WGAN_GP/sliced/
#python slice_add.py 8
#cd ../../../../results/celeba_9/WGAN_GP/sliced/
#python slice_add.py 9
#cd ../../../../results/celeba_10/WGAN_GP/sliced/
#python slice_add.py 10
cd ../../../../

cp -r results/celeba_1 results/5_folder_results/5000_images/iteration2
cp -r results/celeba_2 results/5_folder_results/5000_images/iteration2
cp -r results/celeba_3 results/5_folder_results/5000_images/iteration2
cp -r results/celeba_4 results/5_folder_results/5000_images/iteration2
cp -r results/celeba_5 results/5_folder_results/5000_images/iteration2
cp -r model_result_distributed_1  results/5_folder_results/5000_images/iteration2
cp -r model_result_distributed_2  results/5_folder_results/5000_images/iteration2
cp -r model_result_distributed_3  results/5_folder_results/5000_images/iteration2
cp -r model_result_distributed_4  results/5_folder_results/5000_images/iteration2
cp -r model_result_distributed_5  results/5_folder_results/5000_images/iteration2

python main.py --dataset celeba_1 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_1/discriminator_epoch_099.pth --netG_path model_result_distributed_1/generator_epoch_099.pth --input_size 64
python main.py --dataset celeba_2 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_2/discriminator_epoch_099.pth --netG_path model_result_distributed_2/generator_epoch_099.pth --input_size 64
python main.py --dataset celeba_3 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_3/discriminator_epoch_099.pth --netG_path model_result_distributed_3/generator_epoch_099.pth --input_size 64
python main.py --dataset celeba_4 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_4/discriminator_epoch_099.pth --netG_path model_result_distributed_4/generator_epoch_099.pth --input_size 64
python main.py --dataset celeba_5 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_5/discriminator_epoch_099.pth --netG_path model_result_distributed_5/generator_epoch_099.pth --input_size 64
#python main.py --dataset celeba_6 --gan_type WGAN_GP --epoch 300 --batch_size 64
#python main.py --dataset celeba_7 --gan_type WGAN_GP --epoch 300 --batch_size 64
#python main.py --dataset celeba_8 --gan_type WGAN_GP --epoch 300 --batch_size 64
#python main.py --dataset celeba_9 --gan_type WGAN_GP --epoch 300 --batch_size 64
#python main.py --dataset celeba_10 --gan_type WGAN_GP --epoch 300 --batch_size 64

cd results/celeba_1/WGAN_GP/sliced/
python slice_add.py 1
cd ../../../../results/celeba_2/WGAN_GP/sliced/
python slice_add.py 2
cd ../../../../results/celeba_3/WGAN_GP/sliced/
python slice_add.py 3
cd ../../../../results/celeba_4/WGAN_GP/sliced/
python slice_add.py 4
cd ../../../../results/celeba_5/WGAN_GP/sliced/
python slice_add.py 5
#cd ../../../../results/celeba_6/WGAN_GP/sliced/
#python slice_add.py 6
#cd ../../../../results/celeba_7/WGAN_GP/sliced/
#python slice_add.py 7
#cd ../../../../results/celeba_8/WGAN_GP/sliced/
#python slice_add.py 8
#cd ../../../../results/celeba_9/WGAN_GP/sliced/
#python slice_add.py 9
#cd ../../../../results/celeba_10/WGAN_GP/sliced/
#python slice_add.py 10
cd ../../../../
'''
cp -r results/celeba_1 results/5_folder_results/5000_images/iteration3
cp -r results/celeba_2 results/5_folder_results/5000_images/iteration3
cp -r results/celeba_3 results/5_folder_results/5000_images/iteration3
cp -r results/celeba_4 results/5_folder_results/5000_images/iteration3
cp -r results/celeba_5 results/5_folder_results/5000_images/iteration3
cp -r model_result_distributed_1 results/5_folder_results/5000_images/iteration3
cp -r model_result_distributed_2 results/5_folder_results/5000_images/iteration3
cp -r model_result_distributed_3 results/5_folder_results/5000_images/iteration3
cp -r model_result_distributed_4 results/5_folder_results/5000_images/iteration3
cp -r model_result_distributed_5 results/5_folder_results/5000_images/iteration3

python main.py --dataset celeba_1 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_1/discriminator_epoch_099.pth --netG_path model_result_distributed_1/generator_epoch_099.pth --input_size 64
python main.py --dataset celeba_2 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_2/discriminator_epoch_099.pth --netG_path model_result_distributed_2/generator_epoch_099.pth --input_size 64
python main.py --dataset celeba_3 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_3/discriminator_epoch_099.pth --netG_path model_result_distributed_3/generator_epoch_099.pth --input_size 64
python main.py --dataset celeba_4 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_4/discriminator_epoch_099.pth --netG_path model_result_distributed_4/generator_epoch_099.pth --input_size 64
python main.py --dataset celeba_5 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_5/discriminator_epoch_099.pth --netG_path model_result_distributed_5/generator_epoch_099.pth --input_size 64
#python main.py --dataset celeba_6 --gan_type WGAN_GP --epoch 300 --batch_size 64
#python main.py --dataset celeba_7 --gan_type WGAN_GP --epoch 300 --batch_size 64
#python main.py --dataset celeba_8 --gan_type WGAN_GP --epoch 300 --batch_size 64
#python main.py --dataset celeba_9 --gan_type WGAN_GP --epoch 300 --batch_size 64
#python main.py --dataset celeba_10 --gan_type WGAN_GP --epoch 300 --batch_size 64

cd results/celeba_1/WGAN_GP/sliced/
python slice_add.py 1
cd ../../../../results/celeba_2/WGAN_GP/sliced/
python slice_add.py 2
cd ../../../../results/celeba_3/WGAN_GP/sliced/
python slice_add.py 3
cd ../../../../results/celeba_4/WGAN_GP/sliced/
python slice_add.py 4
cd ../../../../results/celeba_5/WGAN_GP/sliced/
python slice_add.py 5
#cd ../../../../results/celeba_6/WGAN_GP/sliced/
#python slice_add.py 6
#cd ../../../../results/celeba_7/WGAN_GP/sliced/
#python slice_add.py 7
#cd ../../../../results/celeba_8/WGAN_GP/sliced/
#python slice_add.py 8
#cd ../../../../results/celeba_9/WGAN_GP/sliced/
#python slice_add.py 9
#cd ../../../../results/celeba_10/WGAN_GP/sliced/
#python slice_add.py 10
cd ../../../../

cp -r results/celeba_1 results/5_folder_results/5000_images/iteration4
cp -r results/celeba_2 results/5_folder_results/5000_images/iteration4
cp -r results/celeba_3 results/5_folder_results/5000_images/iteration4
cp -r results/celeba_4 results/5_folder_results/5000_images/iteration4
cp -r results/celeba_5 results/5_folder_results/5000_images/iteration4
cp -r model_result_distributed_1 results/5_folder_results/5000_images/iteration4
cp -r model_result_distributed_2 results/5_folder_results/5000_images/iteration4
cp -r model_result_distributed_3 results/5_folder_results/5000_images/iteration4
cp -r model_result_distributed_4 results/5_folder_results/5000_images/iteration4
cp -r model_result_distributed_5 results/5_folder_results/5000_images/iteration4

python main.py --dataset celeba_1 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_1/discriminator_epoch_099.pth --netG_path model_result_distributed_1/generator_epoch_099.pth --input_size 64
python main.py --dataset celeba_2 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_2/discriminator_epoch_099.pth --netG_path model_result_distributed_2/generator_epoch_099.pth --input_size 64
python main.py --dataset celeba_3 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_3/discriminator_epoch_099.pth --netG_path model_result_distributed_3/generator_epoch_099.pth --input_size 64
python main.py --dataset celeba_4 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_4/discriminator_epoch_099.pth --netG_path model_result_distributed_4/generator_epoch_099.pth --input_size 64
python main.py --dataset celeba_5 --gan_type WGAN_GP --epoch 100 --batch_size 64 --netD_path model_result_distributed_5/discriminator_epoch_099.pth --netG_path model_result_distributed_5/generator_epoch_099.pth --input_size 64
#python main.py --dataset celeba_6 --gan_type WGAN_GP --epoch 300 --batch_size 64
#python main.py --dataset celeba_7 --gan_type WGAN_GP --epoch 300 --batch_size 64
#python main.py --dataset celeba_8 --gan_type WGAN_GP --epoch 300 --batch_size 64
#python main.py --dataset celeba_9 --gan_type WGAN_GP --epoch 300 --batch_size 64
#python main.py --dataset celeba_10 --gan_type WGAN_GP --epoch 300 --batch_size 64

cd results/celeba_1/WGAN_GP/sliced/
python slice_add.py 1
cd ../../../../results/celeba_2/WGAN_GP/sliced/
python slice_add.py 2
cd ../../../../results/celeba_3/WGAN_GP/sliced/
python slice_add.py 3
cd ../../../../results/celeba_4/WGAN_GP/sliced/
python slice_add.py 4
cd ../../../../results/celeba_5/WGAN_GP/sliced/
python slice_add.py 5
#cd ../../../../results/celeba_6/WGAN_GP/sliced/
#python slice_add.py 6
#cd ../../../../results/celeba_7/WGAN_GP/sliced/
#python slice_add.py 7
#cd ../../../../results/celeba_8/WGAN_GP/sliced/
#python slice_add.py 8
#cd ../../../../results/celeba_9/WGAN_GP/sliced/
#python slice_add.py 9
#cd ../../../../results/celeba_10/WGAN_GP/sliced/
#python slice_add.py 10
cd ../../../../

cp -r results/celeba_1 results/5_folder_results/5000_images/iteration5
cp -r results/celeba_2 results/5_folder_results/5000_images/iteration5
cp -r results/celeba_3 results/5_folder_results/5000_images/iteration5
cp -r results/celeba_4 results/5_folder_results/5000_images/iteration5
cp -r results/celeba_5 results/5_folder_results/5000_images/iteration5
cp -r model_result_distributed_1 results/5_folder_results/5000_images/iteration5
cp -r model_result_distributed_2 results/5_folder_results/5000_images/iteration5
cp -r model_result_distributed_3 results/5_folder_results/5000_images/iteration5
cp -r model_result_distributed_4 results/5_folder_results/5000_images/iteration5
cp -r model_result_distributed_5 results/5_folder_results/5000_images/iteration5

'''

